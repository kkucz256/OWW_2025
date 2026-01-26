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

inline __host__ __device__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Pixel
{
    byte r, g, b;
    __device__ __host__ bool operator==(const Pixel &o) const { return r == o.r && g == o.g && b == o.b; }
};

struct Image
{
    int width, height;
    std::vector<Pixel> pixels;
};

class Timer
{
    std::chrono::high_resolution_clock::time_point start;

public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() { return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count(); }
};

struct DetailedPrep
{
    double alloc_host = 0;
    double alloc_device = 0;
    double ctx_init = 0;
};
struct PipelineResult
{
    DetailedPrep prep;
    double h2d = 0, blur = 0, d2h = 0, rle_time = 0, total = 0;
    size_t rle_size = 0;
};
struct KernelStats
{
    int regs_per_thread = 0;
    size_t shared_static = 0;
    size_t local_per_thread = 0;
    int max_threads_per_block = 0;
};

struct PixelAnalysis
{
    int x, y;
    double time_ms;
    size_t total_ops;
    size_t unique_ops;
    size_t redundant_ops;
};

size_t cpuRLE(const Pixel *data, int w, int h, int y0, int y1)
{
    size_t total_size = 0;
    for (int y = y0; y < y1; y++)
    {
        int count = 1;
        for (int x = 1; x < w; x++)
        {
            if (data[y * w + x] == data[y * w + x - 1] && count < 255)
                count++;
            else
            {
                total_size += 4;
                count = 1;
            }
        }
        total_size += 4;
    }
    return total_size;
}

__global__ void rleKernel(const Pixel *src, uint32_t *rowSizes, int w, int h)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h)
        return;
    int count = 1;
    uint32_t current_row_bytes = 0;
    for (int x = 1; x < w; x++)
    {
        if (src[y * w + x] == src[y * w + x - 1] && count < 255)
            count++;
        else
        {
            current_row_bytes += 4;
            count = 1;
        }
    }
    current_row_bytes += 4;
    rowSizes[y] = current_row_bytes;
}

__global__ void blurKernelGlobal(const Pixel *__restrict__ src, Pixel *__restrict__ dst, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    long r = 0, g = 0, b = 0;
    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky)
    {
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx)
        {
            int ny = clampi(y + ky, 0, h - 1);
            int nx = clampi(x + kx, 0, w - 1);
            const Pixel &p = src[ny * w + nx];
            r += p.r;
            g += p.g;
            b += p.b;
        }
    }
    int cnt = (2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1);
    dst[y * w + x] = {(byte)(r / cnt), (byte)(g / cnt), (byte)(b / cnt)};
}

__global__ void blurKernelShared(const Pixel *__restrict__ src, Pixel *__restrict__ dst, int w, int h)
{
    __shared__ Pixel tile[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x_base = blockIdx.x * BLOCK, y_base = blockIdx.y * BLOCK;

    for (int i = ty; i < TILE; i += BLOCK)
    {
        for (int j = tx; j < TILE; j += BLOCK)
        {
            int load_x = clampi(x_base - BLUR_RADIUS + j, 0, w - 1);
            int load_y = clampi(y_base - BLUR_RADIUS + i, 0, h - 1);
            tile[i][j] = src[load_y * w + load_x];
        }
    }
    __syncthreads();

    int x = x_base + tx, y = y_base + ty;
    if (x >= w || y >= h)
        return;

    long r = 0, g = 0, b = 0;
    int sx = tx + BLUR_RADIUS, sy = ty + BLUR_RADIUS;

    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++)
    {
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++)
        {
            Pixel p = tile[sy + ky][sx + kx];
            r += p.r;
            g += p.g;
            b += p.b;
        }
    }
    int cnt = (2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1);
    dst[y * w + x] = {(byte)(r / cnt), (byte)(g / cnt), (byte)(b / cnt)};
}

PixelAnalysis analyzePixel(int target_x, int target_y, int w, int h, const std::vector<Pixel> &pixels)
{
    PixelAnalysis stats = {target_x, target_y, 0.0, 0, 0, 0};

    int radius = BLUR_RADIUS;
    stats.total_ops = (2 * radius + 1) * (2 * radius + 1);

    int start_x = target_x - radius;
    int end_x = target_x + radius;
    int start_y = target_y - radius;
    int end_y = target_y + radius;

    int valid_w = std::min(end_x, w - 1) - std::max(start_x, 0) + 1;
    int valid_h = std::min(end_y, h - 1) - std::max(start_y, 0) + 1;

    stats.unique_ops = valid_w * valid_h;
    stats.redundant_ops = stats.total_ops - stats.unique_ops;

    Timer t;
    long r = 0, g = 0, b = 0;
    for (int ky = -radius; ky <= radius; ++ky)
    {
        for (int kx = -radius; kx <= radius; ++kx)
        {
            int ny = clampi(target_y + ky, 0, h - 1);
            int nx = clampi(target_x + kx, 0, w - 1);
            const Pixel &p = pixels[ny * w + nx];
            r += p.r;
            g += p.g;
            b += p.b;
        }
    }
    stats.time_ms = t.elapsed();
    return stats;
}
void saveReport(const std::string &fname, int w, int h, int threads, double loadT, double saveT,
                const PipelineResult &st, const PipelineResult &mt,
                const PipelineResult &cu_global, const PipelineResult &cu_shared,
                const KernelStats &ks, const PixelAnalysis &corner, const PixelAnalysis &center)
{
    std::ofstream f(fname);
    auto line = [&]()
    { f << std::string(110, '-') << "\n"; };
    f << "PERFORMANCE REPORT: GLOBAL vs SHARED CACHING\n";
    f << "Image: " << w << "x" << h << " | Threads: " << threads << " | Radius: " << BLUR_RADIUS << "\n";
    line();
    f << "[Disk I/O]\n"
      << "Load: " << loadT << " ms\n"
      << "Save: " << saveT << " ms\n";
    line();
    f << std::left << std::setw(25) << "Stage"
      << std::setw(20) << "Single"
      << std::setw(20) << "Multi"
      << std::setw(20) << "CUDA (Global)"
      << "CUDA (Shared)\n";
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
    f << "[ROW-BASED RLE RESULTS (After Blur)]\n";
    f << "Original raw size: " << (size_t)w * h * 3 << " B\n";
    f << "Single RLE size:   " << st.rle_size << " B\n";
    f << "Multi RLE size:    " << mt.rle_size << " B\n";
    f << "CUDA (Global) RLE: " << cu_global.rle_size << " B\n";
    f << "CUDA (Shared) RLE: " << cu_shared.rle_size << " B\n";
    line();
    f << "[CUDA SHARED KERNEL STATS]\n"
      << "Registers per thread: " << ks.regs_per_thread << "\n"
      << "Shared memory (static): " << ks.shared_static << " B\n"
      << "Local memory per thread: " << ks.local_per_thread << " B\n"
      << "Max threads per block: " << ks.max_threads_per_block << "\n";
    line();
    f << "[SINGLE PIXEL WORKLOAD ANALYSIS (Corner vs Center)]\n";
    f << "Pixel " << std::setw(15) << "(0,0) [Corner]" << "(100,100) [Center]\n";
    f << "Total Memory Ops: " << std::setw(11) << corner.total_ops << center.total_ops << "\n";
    f << "Unique Mem Ops:   " << std::setw(11) << corner.unique_ops << center.unique_ops << "\n";
    f << "Redundant Clamps: " << std::setw(11) << corner.redundant_ops << center.redundant_ops << "\n";
    f << "CPU Eval Time:    " << corner.time_ms << " ms" << std::string(5, ' ') << center.time_ms << " ms\n";
}

int main(int argc, char *argv[])
{
    if (argc < 2)
        return 1;
    int w, h, c;
    Timer tLoad;
    unsigned char *raw = stbi_load(argv[1], &w, &h, &c, 3);
    double loadT = tLoad.elapsed();
    Image img{w, h};
    img.pixels.assign((Pixel *)raw, (Pixel *)raw + w * h);
    stbi_image_free(raw);
    int ths = std::thread::hardware_concurrency();
    PipelineResult st, mt, cu_global, cu_shared;

    { // Single
        Timer t;
        Image out = img;
        st.prep.alloc_host = t.elapsed();
        Timer tb;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                long r = 0, g = 0, b = 0;
                for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++)
                    for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++)
                    {
                        Pixel &p = img.pixels[clampi(y + ky, 0, h - 1) * w + clampi(x + kx, 0, w - 1)];
                        r += p.r;
                        g += p.g;
                        b += p.b;
                    }
                int cnt = (2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1);
                out.pixels[y * w + x] = {(byte)(r / cnt), (byte)(g / cnt), (byte)(b / cnt)};
            }
        st.blur = tb.elapsed();
        Timer tr;
        st.rle_size = cpuRLE(out.pixels.data(), w, h, 0, h);
        st.rle_time = tr.elapsed();
        st.total = st.prep.alloc_host + st.blur + st.rle_time;
    }

    { // Multi
        Timer t;
        Image out = img;
        mt.prep.alloc_host = t.elapsed();
        Timer tb;
        std::vector<std::thread> pool;
        for (int i = 0; i < ths; i++)
        {
            pool.emplace_back([&, i]
                              {
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
                    } });
        }
        for (auto &th : pool)
            th.join();
        mt.blur = tb.elapsed();
        Timer tr;
        std::vector<size_t> parts(ths);
        std::vector<std::thread> rpool;
        for (int i = 0; i < ths; i++)
            rpool.emplace_back([&, i]
                               { parts[i] = cpuRLE(out.pixels.data(), w, h, h * i / ths, h * (i + 1) / ths); });
        for (auto &th : rpool)
            th.join();
        for (auto s : parts)
            mt.rle_size += s;
        mt.rle_time = tr.elapsed();
        mt.total = mt.prep.alloc_host + mt.blur + mt.rle_time;
    }

    KernelStats ks;
    cudaFree(0);
    Timer tc;
    cudaDeviceSynchronize();
    cu_global.prep.ctx_init = cu_shared.prep.ctx_init = tc.elapsed();

    Timer th;
    std::vector<Pixel> res(w * h);
    cu_global.prep.alloc_host = cu_shared.prep.alloc_host = th.elapsed();

    Timer ta;
    Pixel *ds, *dd;
    size_t sz = w * h * sizeof(Pixel);
    cudaMalloc(&ds, sz);
    cudaMalloc(&dd, sz);
    cu_global.prep.alloc_device = cu_shared.prep.alloc_device = ta.elapsed();

    Timer th2d;
    cudaMemcpy(ds, img.pixels.data(), sz, cudaMemcpyHostToDevice);
    cu_global.h2d = cu_shared.h2d = th2d.elapsed();

    dim3 blk(BLOCK, BLOCK), grd((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);
    uint32_t *dr;
    cudaMalloc(&dr, h * sizeof(uint32_t));
    std::vector<uint32_t> hr(h);

    {
        Timer tb;
        for (int i = 0; i < ITERS; i++)
            blurKernelGlobal<<<grd, blk>>>(ds, dd, w, h);
        cudaDeviceSynchronize();
        cu_global.blur = tb.elapsed() / ITERS;

        Timer tr;
        rleKernel<<<(h + 255) / 256, 256>>>(dd, dr, w, h);
        cudaDeviceSynchronize();
        cu_global.rle_time = tr.elapsed();
        cudaMemcpy(hr.data(), dr, h * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (auto s : hr)
            cu_global.rle_size += s;

        Timer td2h;
        cudaMemcpy(res.data(), dd, sz, cudaMemcpyDeviceToHost);
        cu_global.d2h = td2h.elapsed();
        cu_global.total = cu_global.prep.ctx_init + cu_global.prep.alloc_device + cu_global.h2d + (cu_global.blur * ITERS) + cu_global.rle_time + cu_global.d2h;
    }

    {
        cudaFuncAttributes attr{};
        cudaFuncGetAttributes(&attr, blurKernelShared);
        ks.regs_per_thread = attr.numRegs;
        ks.shared_static = attr.sharedSizeBytes;
        ks.local_per_thread = attr.localSizeBytes;
        ks.max_threads_per_block = attr.maxThreadsPerBlock;

        Timer tb;
        for (int i = 0; i < ITERS; i++)
            blurKernelShared<<<grd, blk>>>(ds, dd, w, h);
        cudaDeviceSynchronize();
        cu_shared.blur = tb.elapsed() / ITERS;

        Timer tr;
        rleKernel<<<(h + 255) / 256, 256>>>(dd, dr, w, h);
        cudaDeviceSynchronize();
        cu_shared.rle_time = tr.elapsed();
        cu_shared.rle_size = 0;
        cudaMemcpy(hr.data(), dr, h * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (auto s : hr)
            cu_shared.rle_size += s;

        Timer td2h;
        cudaMemcpy(res.data(), dd, sz, cudaMemcpyDeviceToHost);
        cu_shared.d2h = td2h.elapsed();
        cu_shared.total = cu_shared.prep.ctx_init + cu_shared.prep.alloc_device + cu_shared.h2d + (cu_shared.blur * ITERS) + cu_shared.rle_time + cu_shared.d2h;
    }

    cudaFree(ds);
    cudaFree(dd);
    cudaFree(dr);
    stbi_write_png("final_result.png", w, h, 3, res.data(), w * 3);

    PixelAnalysis corner = analyzePixel(0, 0, w, h, img.pixels);
    PixelAnalysis center = analyzePixel(100, 100, w, h, img.pixels);

    saveReport("performance_report_full.txt", w, h, ths, loadT, 0, st, mt, cu_global, cu_shared, ks, corner, center);
    std::cout << "Raport gotowy.\n";
    return 0;
}