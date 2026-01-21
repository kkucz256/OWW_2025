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
    long r = 0, g = 0, b = 0; int cnt = 0;
    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky) {
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx) {
            int ny = max(0, min(h - 1, y + ky)), nx = max(0, min(w - 1, x + kx));
            const Pixel& p = src[ny * w + nx];
            r += p.r; g += p.g; b += p.b; cnt++;
        }
    }
    dst[y * w + x] = { (byte)(r/cnt), (byte)(g/cnt), (byte)(b/cnt) };
}

__global__ void rleKernel(const Pixel* src, byte* dst, int* rowSizes, int w, int h) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;
    int rowStart = y * w, outIdx = y * (w * 4), current = outIdx, i = 0;
    while (i < w) {
        Pixel p = src[rowStart + i]; byte count = 1;
        while (i + count < w && count < 255 && src[rowStart + i + count] == p) count++;
        dst[current++] = count; dst[current++] = p.r; dst[current++] = p.g; dst[current++] = p.b;
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

void saveReport(const std::string& fname, int w, int h, int threads, 
                double loadT, double saveT,
                const PipelineResult& st, const PipelineResult& mt, const PipelineResult& cuda) {
    std::ofstream f(fname);
    auto line = [&]() { f << std::string(95, '-') << "\n"; };
    
    f << "PERFORMANCE COMPARISON REPORT (Extreme Breakdown + Disk I/O)\n";
    f << "Image: " << w << "x" << h << " | Threads: " << threads << " | Radius: " << BLUR_RADIUS << "\n";
    line();

    f << "[Disk I/O Latency]\n";
    f << std::left << std::setw(35) << "  - Image Load (Disk -> RAM):" << loadT << " ms\n";
    f << std::left << std::setw(35) << "  - Image Save (RAM -> Disk):" << saveT << " ms\n";
    line();

    f << std::left << std::setw(35) << "STAGE" 
      << std::setw(20) << "SINGLE (ms)" 
      << std::setw(20) << "MULTI (ms)" 
      << "CUDA (ms)\n";
    line();

    f << "[Preparation Breakdown]\n";
    f << std::left << std::setw(35) << "  - Host Alloc/Copy" << std::setw(20) << st.prep_info.alloc_host << std::setw(20) << mt.prep_info.alloc_host << cuda.prep_info.alloc_host << "\n";
    f << std::left << std::setw(35) << "  - Device Alloc" << std::setw(20) << "-" << std::setw(20) << "-" << cuda.prep_info.alloc_device << "\n";
    f << std::left << std::setw(35) << "  - Context/Sync Init" << std::setw(20) << "-" << std::setw(20) << "-" << cuda.prep_info.ctx_init << "\n";
    line();
    f << std::left << std::setw(35) << "H2D Transfer" << std::setw(20) << "-" << std::setw(20) << "-" << cuda.h2d << "\n";
    f << std::left << std::setw(35) << "Blur Computation" << std::setw(20) << st.blur << std::setw(20) << mt.blur << cuda.blur << "\n";
    f << std::left << std::setw(35) << "RLE Computation" << std::setw(20) << st.rle << std::setw(20) << mt.rle << cuda.rle << "\n";
    f << std::left << std::setw(35) << "D2H Transfer" << std::setw(20) << "-" << std::setw(20) << "-" << cuda.d2h << "\n";
    line();
    f << std::left << std::setw(35) << "TOTAL PIPELINE" << std::setw(20) << st.total << std::setw(20) << mt.total << cuda.total << "\n";
    f.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cout << "Gdzie sciezka pliku?\n"; return 1; }
    
    int w, h, c;
    std::cout << "Loading image from disk...\n";
    Timer tLoad;
    unsigned char* raw_data = stbi_load(argv[1], &w, &h, &c, 3);
    if (!raw_data) return 1;
    double loadTime = tLoad.elapsed();

    Image img{ w, h }; img.pixels.assign((Pixel*)raw_data, (Pixel*)raw_data + w * h);
    stbi_image_free(raw_data);

    int threads = std::thread::hardware_concurrency();
    PipelineResult resST, resMT, resCUDA;
    volatile size_t prevent_opt;

    // {
    //     std::cout << "Executing Single Thread...\n";
    //     Timer tP; Image blurred = img; resST.prep_info.alloc_host = tP.elapsed();
    //     Timer tB;
    //     for (int y = 0; y < h; ++y) {
    //         for (int x = 0; x < w; ++x) {
    //             long r = 0, g = 0, b = 0; int cnt = 0;
    //             for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky) {
    //                 for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx) {
    //                     int ny = std::max(0, std::min(h - 1, y + ky)), nx = std::max(0, std::min(w - 1, x + kx));
    //                     const Pixel& p = img.pixels[ny * w + nx];
    //                     r += p.r; g += p.g; b += p.b; cnt++;
    //                 }
    //             }
    //             blurred.pixels[y * w + x] = { (byte)(r/cnt), (byte)(g/cnt), (byte)(b/cnt) };
    //         }
    //     }
    //     resST.blur = tB.elapsed();
    //     Timer tR;
    //     size_t s = 0;
    //     for (size_t i = 0; i < blurred.pixels.size(); ) {
    //         Pixel p = blurred.pixels[i]; byte count = 1;
    //         while (i + count < blurred.pixels.size() && count < 255 && blurred.pixels[i + count] == p) count++;
    //         s += 4; i += count;
    //     }
    //     prevent_opt = s; resST.rle = tR.elapsed();
    //     resST.total = resST.prep_info.alloc_host + resST.blur + resST.rle;
    // }

    // {
    //     std::cout << "Executing Multi Thread...\n";
    //     Timer tP; Image blurred = img; resMT.prep_info.alloc_host = tP.elapsed();
        
    //     Timer tB;
    //     std::vector<std::thread> pool;
    //     for (int i = 0; i < threads; ++i) {
    //         pool.emplace_back([&, i, threads, h, w]() {
    //             int rows = h / threads, sY = i * rows, eY = (i == threads - 1) ? h : (i + 1) * rows;
    //             for (int y = sY; y < eY; ++y) {
    //                 for (int x = 0; x < w; ++x) {
    //                     long r = 0, g = 0, b = 0; int cnt = 0;
    //                     for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky) {
    //                         for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx) {
    //                             int ny = std::max(0, std::min(h - 1, y + ky)), nx = std::max(0, std::min(w - 1, x + kx));
    //                             const Pixel& p = img.pixels[ny * w + nx];
    //                             r += p.r; g += p.g; b += p.b; cnt++;
    //                         }
    //                     }
    //                     blurred.pixels[y * w + x] = { (byte)(r/cnt), (byte)(g/cnt), (byte)(b/cnt) };
    //                 }
    //             }
    //         });
    //     }
    //     for (auto& t : pool) t.join();
    //     resMT.blur = tB.elapsed();

    //     Timer tR;
    //     std::vector<std::thread> rleThreads;
    //     std::vector<size_t> partialResults(threads, 0);
    //     int chunk = (int)blurred.pixels.size() / threads;
    //     for (int i = 0; i < threads; ++i) {
    //         int start = i * chunk;
    //         int end = (i == threads - 1) ? (int)blurred.pixels.size() : (i + 1) * chunk;
    //         rleThreads.emplace_back([=, &blurred, &partialResults]() {
    //             size_t s = 0;
    //             for (int j = start; j < end; ) {
    //                 Pixel p = blurred.pixels[j]; byte c = 1;
    //                 while (j + c < end && c < 255 && blurred.pixels[j + c] == p) c++;
    //                 s += 4; j += c;
    //             }
    //             partialResults[i] = s;
    //         });
    //     }
    //     for (auto& t : rleThreads) t.join();
    //     size_t total_s = 0;
    //     for (size_t val : partialResults) total_s += val;
    //     prevent_opt = total_s; 
    //     resMT.rle = tR.elapsed();

    //     resMT.total = resMT.prep_info.alloc_host + resMT.blur + resMT.rle;
    // }

    Image finalGpuImg = img;
    {
        std::cout << "Executing CUDA...\n";
        Timer tCtx; cudaDeviceSynchronize(); resCUDA.prep_info.ctx_init = tCtx.elapsed();
        Timer tAllocH;
        Image blurred = img; resCUDA.prep_info.alloc_host = tAllocH.elapsed();

        Timer tAllocD;
        Pixel *d_s, *d_d; byte* d_r; int* d_sz;
        size_t sz = w * h * sizeof(Pixel);
        cudaMalloc(&d_s, sz); cudaMalloc(&d_d, sz);
        cudaMalloc(&d_r, sz * 4); cudaMalloc(&d_sz, h * sizeof(int));
        resCUDA.prep_info.alloc_device = tAllocD.elapsed();

        Timer tH2D; cudaMemcpy(d_s, img.pixels.data(), sz, cudaMemcpyHostToDevice); resCUDA.h2d = tH2D.elapsed();
        const int BLOCK = 8;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);

        blurKernel<<<grid, block>>>(d_s, d_d, w, h);
        cudaDeviceSynchronize();

        const int ITERS = 200;

        Timer tB;
        for (int i = 0; i < ITERS; i++) {
            blurKernel<<<grid, block>>>(d_s, d_d, w, h);
        }
        cudaDeviceSynchronize();

        resCUDA.blur = tB.elapsed() / ITERS;

        Timer tR;
        rleKernel<<<(h+255)/256, 256>>>(d_d, d_r, d_sz, w, h);
        cudaDeviceSynchronize();
        resCUDA.rle = tR.elapsed();

        Timer tD2H;
        cudaMemcpy(finalGpuImg.pixels.data(), d_d, sz, cudaMemcpyDeviceToHost);
        resCUDA.d2h = tD2H.elapsed();

        resCUDA.total = resCUDA.prep_info.ctx_init + resCUDA.prep_info.alloc_host + resCUDA.prep_info.alloc_device + resCUDA.h2d + resCUDA.blur + resCUDA.rle + resCUDA.d2h;
        cudaFree(d_s); cudaFree(d_d); cudaFree(d_r); cudaFree(d_sz);
    }



    std::cout << "Saving final image...\n";
    Timer tSave;
    stbi_write_png("final_result.png", w, h, 3, finalGpuImg.pixels.data(), w * 3);
    double saveTime = tSave.elapsed();

    saveReport("performance_full_breakdown.txt", w, h, threads, loadTime, saveTime, resST, resMT, resCUDA);
    std::cout << "Done. Size flag: " << prevent_opt << "\n";
    return 0;
}