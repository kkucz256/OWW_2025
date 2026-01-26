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

constexpr int BLUR_RADIUS = 32;
constexpr int BLOCK = 8;
constexpr int TILE = BLOCK + 2 * BLUR_RADIUS;
const int ITERS = 1;

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

__global__ void blurKernelShared(const Pixel* __restrict__ src, Pixel* __restrict__ dst, int w, int h) {
    __shared__ Pixel tile[TILE][TILE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x_base = blockIdx.x * BLOCK;
    int y_base = blockIdx.y * BLOCK;

    for (int i = ty; i < TILE; i += BLOCK) {
        for (int j = tx; j < TILE; j += BLOCK) {
            int load_x = clampi(x_base - BLUR_RADIUS + j, 0, w - 1);
            int load_y = clampi(y_base - BLUR_RADIUS + i, 0, h - 1);
            tile[i][j] = src[load_y * w + load_x];
        }
    }

    __syncthreads();

    int x = x_base + tx;
    int y = y_base + ty;
    if (x >= w || y >= h) return;

    long r = 0, g = 0, b = 0;
    int sx = tx + BLUR_RADIUS;
    int sy = ty + BLUR_RADIUS;

    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++) {
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++) {
            Pixel p = tile[sy + ky][sx + kx];
            r += p.r; g += p.g; b += p.b;
        }
    }
    
    int cnt = (2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1);
    dst[y * w + x] = {(byte)(r / cnt), (byte)(g / cnt), (byte)(b / cnt)};
}

void saveReport(const std::string& fname, int w, int h, int threads, double loadT, double saveT, const PipelineResult& st, const PipelineResult& mt, const PipelineResult& cu, const KernelStats& ks) {
    std::ofstream f(fname);
    auto line = [&]() { f << std::string(90, '-') << "\n"; };
    f << "PERFORMANCE REPORT (MANUAL TILE CACHING + ROW-RLE)\n";
    f << "Image: " << w << "x" << h << " | Threads: " << threads << " | Radius: " << BLUR_RADIUS << "\n";
    line();
    f << "[Disk I/O]\n" << "Load: " << loadT << " ms\n" << "Save: " << saveT << " ms\n";
    line();
    f << std::left << std::setw(30) << "Stage" << std::setw(20) << "Single" << std::setw(20) << "Multi" << "CUDA\n";
    line();
    f << std::setw(30) << "Host alloc" << std::setw(20) << st.prep.alloc_host << std::setw(20) << mt.prep.alloc_host << cu.prep.alloc_host << "\n";
    f << std::setw(30) << "Device alloc" << std::setw(20) << "-" << std::setw(20) << "-" << cu.prep.alloc_device << "\n";
    f << std::setw(30) << "Context init" << std::setw(20) << "-" << std::setw(20) << "-" << cu.prep.ctx_init << "\n";
    line();
    f << std::setw(30) << "H2D" << std::setw(20) << "-" << std::setw(20) << "-" << cu.h2d << "\n";
    f << std::setw(30) << "Blur" << std::setw(20) << st.blur << std::setw(20) << mt.blur << cu.blur << "\n";
    f << std::setw(30) << "D2H" << std::setw(20) << "-" << std::setw(20) << "-" << cu.d2h << "\n";
    f << std::setw(30) << "RLE Compression" << std::setw(20) << st.rle_time << std::setw(20) << mt.rle_time << cu.rle_time << "\n";
    line();
    f << std::setw(30) << "TOTAL" << std::setw(20) << st.total << std::setw(20) << mt.total << cu.total << "\n";
    line();
    f << "[ROW-BASED RLE RESULTS (After Blur)]\n";
    f << "Original raw size: " << (size_t)w*h*3 << " B\n";
    f << "Single RLE size:   " << st.rle_size << " B\n";
    f << "Multi RLE size:    " << mt.rle_size << " B\n";
    f << "CUDA RLE size:     " << cu.rle_size << " B\n";
    line();
    f << "[CUDA KERNEL STATS]\n" 
      << "Registers per thread: " << ks.regs_per_thread << "\n" 
      << "Shared memory (static): " << ks.shared_static << " B\n" 
      << "Local memory per thread: " << ks.local_per_thread << " B\n" 
      << "Max threads per block: " << ks.max_threads_per_block << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    int w, h, c;
    Timer tLoad; unsigned char* raw = stbi_load(argv[1], &w, &h, &c, 3);
    double loadT = tLoad.elapsed();
    Image img{ w, h }; img.pixels.assign((Pixel*)raw, (Pixel*)raw + w * h); stbi_image_free(raw);
    int ths = std::thread::hardware_concurrency();
    PipelineResult st, mt, cu;

    { // Single
        Timer t; Image out = img; st.prep.alloc_host = t.elapsed();
        Timer tb;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                long r=0,g=0,b=0;
                for (int ky=-BLUR_RADIUS; ky<=BLUR_RADIUS; ky++)
                    for (int kx=-BLUR_RADIUS; kx<=BLUR_RADIUS; kx++) {
                        Pixel& p = img.pixels[clampi(y+ky,0,h-1)*w+clampi(x+kx,0,w-1)];
                        r+=p.r; g+=p.g; b+=p.b;
                    }
                int cnt = (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1);
                out.pixels[y*w+x] = {(byte)(r/cnt),(byte)(g/cnt),(byte)(b/cnt)};
            }
        st.blur = tb.elapsed();
        Timer tr; st.rle_size = cpuRLE(out.pixels.data(), w, h, 0, h); st.rle_time = tr.elapsed();
        st.total = st.prep.alloc_host + st.blur + st.rle_time;
    }

    { // Multi
        Timer t; Image out = img; mt.prep.alloc_host = t.elapsed();
        Timer tb; std::vector<std::thread> pool;
        for (int i=0; i<ths; i++) {
            pool.emplace_back([&, i] {
                int y0 = h*i/ths, y1 = h*(i+1)/ths;
                for (int y=y0; y<y1; y++)
                    for (int x=0; x<w; x++) {
                        long r=0,g=0,b=0;
                        for (int ky=-BLUR_RADIUS; ky<=BLUR_RADIUS; ky++)
                            for (int kx=-BLUR_RADIUS; kx<=BLUR_RADIUS; kx++) {
                                Pixel& p = img.pixels[clampi(y+ky,0,h-1)*w+clampi(x+kx,0,w-1)];
                                r+=p.r; g+=p.g; b+=p.b;
                            }
                        int cnt = (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1);
                        out.pixels[y*w+x] = {(byte)(r/cnt),(byte)(g/cnt),(byte)(b/cnt)};
                    }
            });
        }
        for (auto& th : pool) th.join();
        mt.blur = tb.elapsed();
        Timer tr; std::vector<size_t> parts(ths); std::vector<std::thread> rpool;
        for (int i=0; i<ths; i++) rpool.emplace_back([&, i]{ parts[i] = cpuRLE(out.pixels.data(), w, h, h*i/ths, h*(i+1)/ths); });
        for (auto& th : rpool) th.join();
        for (auto s : parts) mt.rle_size += s;
        mt.rle_time = tr.elapsed();
        mt.total = mt.prep.alloc_host + mt.blur + mt.rle_time;
    }

    KernelStats ks;
    { // CUDA
        cudaFree(0); Timer tc; cudaDeviceSynchronize(); cu.prep.ctx_init = tc.elapsed();
        Timer th; std::vector<Pixel> res(w*h); cu.prep.alloc_host = th.elapsed();
        Timer ta; Pixel *ds, *dd; size_t sz = w*h*sizeof(Pixel);
        cudaMalloc(&ds, sz); cudaMalloc(&dd, sz); cu.prep.alloc_device = ta.elapsed();
        Timer th2d; cudaMemcpy(ds, img.pixels.data(), sz, cudaMemcpyHostToDevice); cu.h2d = th2d.elapsed();
        
        dim3 blk(BLOCK, BLOCK), grd((w+BLOCK-1)/BLOCK, (h+BLOCK-1)/BLOCK);
        cudaFuncAttributes attr{}; cudaFuncGetAttributes(&attr, blurKernelShared);
        ks.regs_per_thread = attr.numRegs; 
        ks.shared_static = attr.sharedSizeBytes;
        ks.local_per_thread = attr.localSizeBytes;
        ks.max_threads_per_block = attr.maxThreadsPerBlock;

        Timer tb; for (int i=0; i<ITERS; i++) blurKernelShared<<<grd, blk>>>(ds, dd, w, h);
        cudaDeviceSynchronize(); cu.blur = tb.elapsed() / ITERS;

        uint32_t *dr; cudaMalloc(&dr, h*sizeof(uint32_t));
        Timer tr; rleKernel<<<(h+255)/256, 256>>>(dd, dr, w, h); cudaDeviceSynchronize(); cu.rle_time = tr.elapsed();
        std::vector<uint32_t> hr(h); cudaMemcpy(hr.data(), dr, h*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (auto s : hr) cu.rle_size += s;

        Timer td2h; cudaMemcpy(res.data(), dd, sz, cudaMemcpyDeviceToHost); cu.d2h = td2h.elapsed();
        cu.total = cu.prep.ctx_init + cu.prep.alloc_device + cu.h2d + (cu.blur * ITERS) + cu.rle_time + cu.d2h;
        cudaFree(ds); cudaFree(dd); cudaFree(dr);
        stbi_write_png("final_result.png", w, h, 3, res.data(), w*3);
    }

    saveReport("performance_report_full.txt", w, h, ths, loadT, 0, st, mt, cu, ks);
    std::cout << "Raport wygenerowany";
    return 0;
}