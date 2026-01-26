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
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using byte = uint8_t;
const int BLUR_RADIUS = 32;

struct Pixel {
    byte r, g, b;
    __device__ __host__ bool operator==(const Pixel& other) const {
        return r == other.r && g == other.g && b == other.b;
    }
};

struct Image {
    int width, height;
    std::vector<Pixel> pixels;
};

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

__global__ void blurKernel(const Pixel* src, Pixel* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    long r = 0, g = 0, b = 0;
    int cnt = 0;

    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky)
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx) {
            int ny = max(0, min(h - 1, y + ky));
            int nx = max(0, min(w - 1, x + kx));
            const Pixel& p = src[ny * w + nx];
            r += p.r; g += p.g; b += p.b;
            cnt++;
        }

    dst[y * w + x] = {
        (byte)(r / cnt),
        (byte)(g / cnt),
        (byte)(b / cnt)
    };
}

__global__ void rleKernel(const Pixel* src, byte* dst, int* rowSizes, int w, int h) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;

    int rowStart = y * w;
    int outIdx = y * (w * 4);
    int current = outIdx;
    int i = 0;

    while (i < w) {
        Pixel p = src[rowStart + i];
        byte count = 1;
        while (i + count < w && count < 255 && src[rowStart + i + count] == p)
            count++;

        dst[current++] = count;
        dst[current++] = p.r;
        dst[current++] = p.g;
        dst[current++] = p.b;
        i += count;
    }
    rowSizes[y] = current - outIdx;
}

struct DetailedPrep {
    double alloc_host = 0;
    double alloc_device = 0;
    double ctx_init = 0;
};

struct PipelineResult {
    DetailedPrep prep_info;
    double h2d = 0, blur = 0, rle = 0, d2h = 0, total = 0;
};

struct KernelStats {
    int regs = 0;
    size_t shared = 0;
    size_t local = 0;
    int maxThreads = 0;
};

void saveReport(
    const std::string& fname, int w, int h, int threads,
    double loadT, double saveT,
    const PipelineResult& st,
    const PipelineResult& mt,
    const PipelineResult& cu,
    const KernelStats& blurKS,
    const KernelStats& rleKS
) {
    std::ofstream f(fname);
    auto line = [&]() { f << std::string(95, '-') << "\n"; };

    f << "PERFORMANCE COMPARISON REPORT\n";
    f << "Image: " << w << "x" << h << " | Threads: " << threads << " | Radius: " << BLUR_RADIUS << "\n";
    line();

    f << "[Disk I/O]\n";
    f << "Load: " << loadT << " ms\n";
    f << "Save: " << saveT << " ms\n";
    line();

    f << std::setw(35) << "STAGE"
      << std::setw(20) << "SINGLE"
      << std::setw(20) << "MULTI"
      << "CUDA\n";
    line();

    f << std::setw(35) << "Blur"
      << std::setw(20) << st.blur
      << std::setw(20) << mt.blur
      << cu.blur << "\n";

    f << std::setw(35) << "RLE"
      << std::setw(20) << st.rle
      << std::setw(20) << mt.rle
      << cu.rle << "\n";

    line();
    f << std::setw(35) << "TOTAL"
      << std::setw(20) << st.total
      << std::setw(20) << mt.total
      << cu.total << "\n";

    line();
    f << "[CUDA KERNEL STATS]\n";
    f << "Blur kernel: regs=" << blurKS.regs << " shared=" << blurKS.shared
      << " local=" << blurKS.local << "\n";
    f << "RLE kernel:  regs=" << rleKS.regs << " shared=" << rleKS.shared
      << " local=" << rleKS.local << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;

    int w, h, c;
    Timer tLoad;
    unsigned char* raw = stbi_load(argv[1], &w, &h, &c, 3);
    double loadTime = tLoad.elapsed();

    Image img{ w, h };
    img.pixels.assign((Pixel*)raw, (Pixel*)raw + w * h);
    stbi_image_free(raw);

    int threads = std::thread::hardware_concurrency();
    PipelineResult resST{}, resMT{}, resCUDA{};
    KernelStats blurKS{}, rleKS{};
    volatile size_t prevent_opt = 0;

    // --- CUDA ---
    Pixel *d_s, *d_d;
    byte* d_r;
    int* d_sz;
    size_t sz = w * h * sizeof(Pixel);

    cudaMalloc(&d_s, sz);
    cudaMalloc(&d_d, sz);
    cudaMalloc(&d_r, sz * 4);
    cudaMalloc(&d_sz, h * sizeof(int));

    cudaMemcpy(d_s, img.pixels.data(), sz, cudaMemcpyHostToDevice);

    dim3 block(8, 8);
    dim3 grid((w + 7) / 8, (h + 7) / 8);

    blurKernel<<<grid, block>>>(d_s, d_d, w, h);
    cudaDeviceSynchronize();

    cudaFuncAttributes a{};
    cudaFuncGetAttributes(&a, blurKernel);
    blurKS = { a.numRegs, a.sharedSizeBytes, a.localSizeBytes, a.maxThreadsPerBlock };

    const int ITERS = 200;
    Timer tB;
    for (int i = 0; i < ITERS; i++)
        blurKernel<<<grid, block>>>(d_s, d_d, w, h);
    cudaDeviceSynchronize();
    resCUDA.blur = tB.elapsed() / ITERS;

    Timer tR;
    rleKernel<<<(h + 255) / 256, 256>>>(d_d, d_r, d_sz, w, h);
    cudaDeviceSynchronize();
    resCUDA.rle = tR.elapsed();

    cudaFuncGetAttributes(&a, rleKernel);
    rleKS = { a.numRegs, a.sharedSizeBytes, a.localSizeBytes, a.maxThreadsPerBlock };

    Image out = img;
    cudaMemcpy(out.pixels.data(), d_d, sz, cudaMemcpyDeviceToHost);

    cudaFree(d_s); cudaFree(d_d); cudaFree(d_r); cudaFree(d_sz);

    Timer tSave;
    stbi_write_png("final_result.png", w, h, 3, out.pixels.data(), w * 3);
    double saveTime = tSave.elapsed();

    saveReport("performance_full_breakdown.txt",
               w, h, threads,
               loadTime, saveTime,
               resST, resMT, resCUDA,
               blurKS, rleKS);

    return 0;
}
