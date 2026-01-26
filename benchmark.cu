#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_SIMD
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using byte = uint8_t;

const int KERNEL_ITERS = 1;

inline __host__ __device__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Pixel {
    byte r, g, b;
    __device__ __host__ bool operator==(const Pixel& o) const { return r == o.r && g == o.g && b == o.b; }
};

struct Image { int width, height; std::vector<Pixel> pixels; };

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() { return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count(); }
};

struct DetailedPrep { double alloc_host = 0; double alloc_device = 0; double ctx_init = 0; };
struct PipelineResult { DetailedPrep prep; double h2d = 0, blur = 0, d2h = 0, rle_time = 0, total = 0; size_t rle_size = 0; };
struct KernelStats { 
    int regs_per_thread = 0; 
    size_t shared_static = 0; 
    size_t local_per_thread = 0; 
    int max_threads_per_block = 0; 
};

struct PixelAnalysis {
    int x, y;
    double time_ms;
    size_t total_ops;
    size_t unique_ops;
    size_t redundant_ops;
};


size_t cpuRLE(const Pixel* data, int w, int h, int y0, int y1) {
    size_t total_size = 0;
    for (int y = y0; y < y1; y++) {
        int count = 1;
        for (int x = 1; x < w; x++) {
            if (data[y * w + x] == data[y * w + x - 1] && count < 255) count++;
            else { total_size += 4; count = 1; }
        }
        total_size += 4;
    }
    return total_size;
}

PixelAnalysis analyzePixel(int target_x, int target_y, int w, int h, const std::vector<Pixel>& pixels, int radius) {
    PixelAnalysis stats = {target_x, target_y, 0.0, 0, 0, 0};
    stats.total_ops = (2 * radius + 1) * (2 * radius + 1);
    int valid_w = std::min(target_x + radius, w - 1) - std::max(target_x - radius, 0) + 1;
    int valid_h = std::min(target_y + radius, h - 1) - std::max(target_y - radius, 0) + 1;
    stats.unique_ops = (size_t)valid_w * valid_h;
    stats.redundant_ops = stats.total_ops - stats.unique_ops;
    Timer t;
    long r=0, g=0, b=0;
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int ny = clampi(target_y + ky, 0, h - 1);
            int nx = clampi(target_x + kx, 0, w - 1);
            const Pixel& p = pixels[ny * w + nx];
            r += p.r; g += p.g; b += p.b;
        }
    }
    stats.time_ms = t.elapsed();
    return stats;
}


__global__ void rleKernel(const Pixel* src, uint32_t* rowSizes, int w, int h) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;
    int count = 1;
    uint32_t current_row_bytes = 0;
    for (int x = 1; x < w; x++) {
        if (src[y * w + x] == src[y * w + x - 1] && count < 255) count++;
        else { current_row_bytes += 4; count = 1; }
    }
    current_row_bytes += 4;
    rowSizes[y] = current_row_bytes;
}

__global__ void blurKernelGlobal(const Pixel* __restrict__ src, Pixel* __restrict__ dst, int w, int h, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    long r = 0, g = 0, b = 0; 
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int ny = clampi(y + ky, 0, h - 1);
            int nx = clampi(x + kx, 0, w - 1);
            const Pixel& p = src[ny * w + nx];
            r += p.r; g += p.g; b += p.b;
        }
    }
    int cnt = (2 * radius + 1) * (2 * radius + 1);
    dst[y * w + x] = { (byte)(r/cnt), (byte)(g/cnt), (byte)(b/cnt) };
}

__global__ void blurKernelShared(const Pixel* __restrict__ src, Pixel* __restrict__ dst, int w, int h, int radius, int block_dim) {
    extern __shared__ Pixel tile[];
    int tile_w = block_dim + 2 * radius;
    int tx = threadIdx.x, ty = threadIdx.y;
    int x_base = blockIdx.x * block_dim, y_base = blockIdx.y * block_dim;

    for (int i = ty; i < tile_w; i += block_dim) {
        for (int j = tx; j < tile_w; j += block_dim) {
            int lx = clampi(x_base - radius + j, 0, w - 1);
            int ly = clampi(y_base - radius + i, 0, h - 1);
            tile[i * tile_w + j] = src[ly * w + lx];
        }
    }
    __syncthreads();

    int x = x_base + tx, y = y_base + ty;
    if (x >= w || y >= h) return;
    long r = 0, g = 0, b = 0;
    int sx = tx + radius, sy = ty + radius;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            Pixel p = tile[(sy + ky) * tile_w + (sx + kx)];
            r += p.r; g += p.g; b += p.b;
        }
    }
    int cnt = (2 * radius + 1) * (2 * radius + 1);
    dst[y * w + x] = {(byte)(r / cnt), (byte)(g / cnt), (byte)(b / cnt)};
}


void saveFullReport(std::ofstream& f, const std::string& label, int w, int h, int radius, int block_size, int threads, 
                    double loadT, double saveT, const PipelineResult& st, const PipelineResult& mt, 
                    const PipelineResult& cu_global, const PipelineResult& cu_shared, 
                    const KernelStats& ks, const PixelAnalysis& corner, const PixelAnalysis& center) {
    auto line = [&]() { f << std::string(110, '-') << "\n"; };
    f << "\nSCENARIO: " << label << "\n";
    f << "Image: " << w << "x" << h << " | Threads: " << threads << " | Radius: " << radius << " | Block: " << block_size << "\n";
    line();
    f << "[Disk I/O]\n" << "Load: " << loadT << " ms\n" << "Save: " << saveT << " ms\n";
    line();
    f << std::left << std::setw(25) << "Stage" << std::setw(20) << "Single" << std::setw(20) << "Multi" << std::setw(20) << "CUDA (Global)" << "CUDA (Shared)\n";
    line();
    f << std::setw(25) << "Host alloc" << std::setw(20) << st.prep.alloc_host << std::setw(20) << mt.prep.alloc_host << std::setw(20) << cu_global.prep.alloc_host << cu_shared.prep.alloc_host << "\n";
    f << std::setw(25) << "Device alloc" << std::setw(20) << "-" << std::setw(20) << "-" << std::setw(20) << cu_global.prep.alloc_device << cu_shared.prep.alloc_device << "\n";
    f << std::setw(25) << "Context init" << std::setw(20) << "-" << std::setw(20) << "-" << std::setw(20) << cu_global.prep.ctx_init << cu_shared.prep.ctx_init << "\n";
    line();
    f << std::setw(25) << "H2D" << std::setw(20) << "-" << std::setw(20) << "-" << std::setw(20) << cu_global.h2d << cu_shared.h2d << "\n";
    f << std::setw(25) << "Blur" << std::setw(20) << st.blur << std::setw(20) << mt.blur << std::setw(20) << cu_global.blur << cu_shared.blur << "\n";
    f << std::setw(25) << "D2H" << std::setw(20) << "-" << std::setw(20) << "-" << std::setw(20) << cu_global.d2h << cu_shared.d2h << "\n";
    f << std::setw(25) << "RLE Compression" << std::setw(20) << st.rle_time << std::setw(20) << mt.rle_time << std::setw(20) << cu_global.rle_time << cu_shared.rle_time << "\n";
    line();
    f << std::setw(25) << "TOTAL" << std::setw(20) << st.total << std::setw(20) << mt.total << std::setw(20) << cu_global.total << cu_shared.total << "\n";
    line();
    f << "[ROW-BASED RLE RESULTS]\n";
    f << "Single RLE size:   " << st.rle_size << " B\n";
    f << "Multi RLE size:    " << mt.rle_size << " B\n";
    f << "CUDA (Global) RLE: " << cu_global.rle_size << " B\n";
    f << "CUDA (Shared) RLE: " << cu_shared.rle_size << " B\n";
    line();
    f << "[CUDA KERNEL STATS]\n" << "Regs: " << ks.regs_per_thread << " | Shared static: " << ks.shared_static << " B\n";
    line();
    f << "[SINGLE PIXEL WORKLOAD ANALYSIS]\n";
    f << "Pixel " << std::setw(15) << "(0,0) [Corner]" << "(w/2, h/2) [Center]\n";
    f << "Total Memory Ops: " << std::setw(11) << corner.total_ops << center.total_ops << "\n";
    f << "Unique Mem Ops:   " << std::setw(11) << corner.unique_ops << center.unique_ops << "\n";
    f << "Redundant Clamps: " << std::setw(11) << corner.redundant_ops << center.redundant_ops << "\n";
    f << "CPU Eval Time:    " << corner.time_ms << " ms" << std::string(5, ' ') << center.time_ms << " ms\n";
    line();
}


void executeScenario(std::ofstream& reportFile, const std::string& imgName, int radius, int block_sz, const std::string& label) {
    int w, h, c;
    Timer tLoad; unsigned char* raw = stbi_load(imgName.c_str(), &w, &h, &c, 3);
    if(!raw) { std::cerr << "Failed to load " << imgName << "\n"; return; }
    double loadT = tLoad.elapsed();
    Image img{ w, h }; img.pixels.assign((Pixel*)raw, (Pixel*)raw + w * h); stbi_image_free(raw);
    int ths = std::thread::hardware_concurrency();
    PipelineResult st, mt, cu_global, cu_shared;

    {
        Timer t; std::vector<Pixel> out = img.pixels; st.prep.alloc_host = t.elapsed();
        Timer tb;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                long r=0,g=0,b=0;
                for (int ky=-radius; ky<=radius; ky++)
                    for (int kx=-radius; kx<=radius; kx++) {
                        Pixel& p = img.pixels[clampi(y+ky,0,h-1)*w+clampi(x+kx,0,w-1)];
                        r+=p.r; g+=p.g; b+=p.b;
                    }
                int cnt = (2*radius+1)*(2*radius+1);
                out[y*w+x] = {(byte)(r/cnt),(byte)(g/cnt),(byte)(b/cnt)};
            }
        st.blur = tb.elapsed();
        Timer tr; st.rle_size = cpuRLE(out.data(), w, h, 0, h); st.rle_time = tr.elapsed();
        st.total = st.prep.alloc_host + st.blur + st.rle_time;
    }

    {
        Timer t; std::vector<Pixel> out = img.pixels; mt.prep.alloc_host = t.elapsed();
        Timer tb; std::vector<std::thread> pool;
        for (int i=0; i<ths; i++) {
            pool.emplace_back([&, i, radius] {
                int y0 = h*i/ths, y1 = h*(i+1)/ths;
                for (int y=y0; y<y1; y++)
                    for (int x=0; x<w; x++) {
                        long r=0,g=0,b=0;
                        for (int ky=-radius; ky<=radius; ky++)
                            for (int kx=-radius; kx<=radius; kx++) {
                                Pixel& p = img.pixels[clampi(y+ky,0,h-1)*w+clampi(x+kx,0,w-1)];
                                r+=p.r; g+=p.g; b+=p.b;
                            }
                        int cnt = (2*radius+1)*(2*radius+1);
                        out[y*w+x] = {(byte)(r/cnt),(byte)(g/cnt),(byte)(b/cnt)};
                    }
            });
        }
        for (auto& th : pool) th.join();
        mt.blur = tb.elapsed();
        Timer tr; std::vector<size_t> pRLE(ths); std::vector<std::thread> rpool;
        for (int i=0; i<ths; i++) rpool.emplace_back([&, i]{ pRLE[i] = cpuRLE(out.data(), w, h, h*i/ths, h*(i+1)/ths); });
        for (auto& th : rpool) th.join();
        for (auto s : pRLE) mt.rle_size += s;
        mt.rle_time = tr.elapsed();
        mt.total = mt.prep.alloc_host + mt.blur + mt.rle_time;
    }

    KernelStats ks;
    cudaFree(0); Timer tc; cudaDeviceSynchronize(); 
    cu_global.prep.ctx_init = cu_shared.prep.ctx_init = tc.elapsed();

    Timer thAlloc; std::vector<Pixel> res(w*h); 
    cu_global.prep.alloc_host = cu_shared.prep.alloc_host = thAlloc.elapsed();

    Timer ta; Pixel *ds, *dd; size_t sz = w*h*sizeof(Pixel);
    cudaMalloc(&ds, sz); cudaMalloc(&dd, sz); 
    cu_global.prep.alloc_device = cu_shared.prep.alloc_device = ta.elapsed();

    Timer th2d; cudaMemcpy(ds, img.pixels.data(), sz, cudaMemcpyHostToDevice); 
    cu_global.h2d = cu_shared.h2d = th2d.elapsed();

    dim3 blk(block_sz, block_sz), grd((w+block_sz-1)/block_sz, (h+block_sz-1)/block_sz);
    uint32_t *dr; cudaMalloc(&dr, h*sizeof(uint32_t));
    std::vector<uint32_t> hr(h);

    Timer tbg; for(int i=0; i<KERNEL_ITERS; i++) blurKernelGlobal<<<grd, blk>>>(ds, dd, w, h, radius);
    cudaDeviceSynchronize(); cu_global.blur = tbg.elapsed() / KERNEL_ITERS;

    int tile_w = block_sz + 2 * radius;
    size_t shm = tile_w * tile_w * sizeof(Pixel);
    cudaFuncAttributes attr{}; cudaFuncGetAttributes(&attr, blurKernelShared);
    ks.regs_per_thread = attr.numRegs; ks.shared_static = attr.sharedSizeBytes;
    
    Timer tbs; for(int i=0; i<KERNEL_ITERS; i++) blurKernelShared<<<grd, blk, shm>>>(ds, dd, w, h, radius, block_sz);
    cudaDeviceSynchronize(); cu_shared.blur = tbs.elapsed() / KERNEL_ITERS;

    Timer tr; rleKernel<<<(h+255)/256, 256>>>(dd, dr, w, h); cudaDeviceSynchronize(); 
    cu_global.rle_time = cu_shared.rle_time = tr.elapsed();
    cudaMemcpy(hr.data(), dr, h*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for (auto s : hr) cu_global.rle_size = (cu_shared.rle_size += s);

    Timer td2h; cudaMemcpy(res.data(), dd, sz, cudaMemcpyDeviceToHost); 
    cu_global.d2h = cu_shared.d2h = td2h.elapsed();

    cu_global.total = cu_global.prep.ctx_init + cu_global.prep.alloc_device + cu_global.h2d + cu_global.blur + cu_global.rle_time + cu_global.d2h;
    cu_shared.total = cu_shared.prep.ctx_init + cu_shared.prep.alloc_device + cu_shared.h2d + cu_shared.blur + cu_shared.rle_time + cu_shared.d2h;

    PixelAnalysis pCorner = analyzePixel(0, 0, w, h, img.pixels, radius);
    PixelAnalysis pCenter = analyzePixel(w/2, h/2, w, h, img.pixels, radius);

    saveFullReport(reportFile, label, w, h, radius, block_sz, ths, loadT, 0, st, mt, cu_global, cu_shared, ks, pCorner, pCenter);
    
    cudaFree(ds); cudaFree(dd); cudaFree(dr);
}

int main() {
    std::ofstream reportFile("performance_report_benchmark.txt", std::ios::app);
    
    for (int iter = 1; iter <= 5; iter++) {
        reportFile << "\n" << std::string(40, '=') << " ITERATION " << iter << " " << std::string(40, '=') << "\n";
        
        executeScenario(reportFile, "obraz_maly.jpg", 32, 8, "SCENARIO 1: Small Image");
        executeScenario(reportFile, "obraz_sredni.jpg", 32, 8, "SCENARIO 1: Medium Image");
        executeScenario(reportFile, "obraz_duzy.jpg", 32, 8, "SCENARIO 1: Large Image");

        executeScenario(reportFile, "obraz_sredni.jpg", 16, 8, "SCENARIO 2: Radius 16");
        executeScenario(reportFile, "obraz_sredni.jpg", 32, 8, "SCENARIO 2: Radius 32");
        executeScenario(reportFile, "obraz_sredni.jpg", 64, 8, "SCENARIO 2: Radius 64");

        executeScenario(reportFile, "obraz_sredni.jpg", 32, 16, "SCENARIO 3: Block 16x16");
        executeScenario(reportFile, "obraz_sredni.jpg", 32, 32, "SCENARIO 3: Block 32x32");
    }

    std::cout << "Testy zakończone. Wszystkie raporty dopisane do performance_report_full.txt\n";
    return 0;
}