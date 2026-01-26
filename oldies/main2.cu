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

constexpr int BLUR_RADIUS = 32;
constexpr int BLOCK = 8;
constexpr int TILE = BLOCK + 2 * BLUR_RADIUS;

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Pixel {
    byte r, g, b;
    __device__ __host__ bool operator==(const Pixel& o) const {
        return r == o.r && g == o.g && b == o.b;
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
        auto e = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(e - start).count();
    }
};


__global__ void blurKernelShared(
    const Pixel* __restrict__ src,
    Pixel* __restrict__ dst,
    int w, int h
) {
    __shared__ Pixel tile[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK + tx;
    int y = blockIdx.y * BLOCK + ty;
    int sx = tx + BLUR_RADIUS;
    int sy = ty + BLUR_RADIUS;

    if (x < w && y < h)
        tile[sy][sx] = src[y * w + x];

    if (ty < BLUR_RADIUS && x < w) {
        int y0 = max(y - BLUR_RADIUS, 0);
        int y1 = min(y + BLOCK, h - 1);
        tile[ty][sx] = src[y0 * w + x];
        tile[sy + BLOCK][sx] = src[y1 * w + x];
    }

    if (tx < BLUR_RADIUS && y < h) {
        int x0 = max(x - BLUR_RADIUS, 0);
        int x1 = min(x + BLOCK, w - 1);
        tile[sy][tx] = src[y * w + x0];
        tile[sy][sx + BLOCK] = src[y * w + x1];
    }

    if (tx < BLUR_RADIUS && ty < BLUR_RADIUS) {
        int x0 = max(x - BLUR_RADIUS, 0);
        int x1 = min(x + BLOCK, w - 1);
        int y0 = max(y - BLUR_RADIUS, 0);
        int y1 = min(y + BLOCK, h - 1);

        tile[ty][tx] = src[y0 * w + x0];
        tile[ty][sx + BLOCK] = src[y0 * w + x1];
        tile[sy + BLOCK][tx] = src[y1 * w + x0];
        tile[sy + BLOCK][sx + BLOCK] = src[y1 * w + x1];
    }

    __syncthreads();

    if (x >= w || y >= h) return;

    long r = 0, g = 0, b = 0;
    int cnt = 0;

    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++)
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++) {
            Pixel p = tile[sy + ky][sx + kx];
            r += p.r; g += p.g; b += p.b;
            cnt++;
        }

    dst[y * w + x] = {
        (byte)(r / cnt),
        (byte)(g / cnt),
        (byte)(b / cnt)
    };
}


struct DetailedPrep {
    double alloc_host = 0;
    double alloc_device = 0;
    double ctx_init = 0;
};

struct PipelineResult {
    DetailedPrep prep;
    double h2d = 0, blur = 0, d2h = 0, total = 0;
};


void saveReport(
    const std::string& fname,
    int w, int h, int threads,
    double loadT, double saveT,
    const PipelineResult& st,
    const PipelineResult& mt,
    const PipelineResult& cu
) {
    std::ofstream f(fname);
    auto line = [&]() { f << std::string(90, '-') << "\n"; };

    f << "PERFORMANCE COMPARISON REPORT (CUDA SHARED MEMORY BLUR)\n";
    f << "Image: " << w << "x" << h
      << " | Threads: " << threads
      << " | Radius: " << BLUR_RADIUS << "\n";
    line();

    f << "[Disk I/O]\n";
    f << "Load: " << loadT << " ms\n";
    f << "Save: " << saveT << " ms\n";
    line();

    f << std::left
      << std::setw(30) << "Stage"
      << std::setw(20) << "Single"
      << std::setw(20) << "Multi"
      << "CUDA\n";
    line();

    f << std::setw(30) << "Host alloc"
      << std::setw(20) << st.prep.alloc_host
      << std::setw(20) << mt.prep.alloc_host
      << cu.prep.alloc_host << "\n";

    f << std::setw(30) << "Device alloc"
      << std::setw(20) << "-"
      << std::setw(20) << "-"
      << cu.prep.alloc_device << "\n";

    f << std::setw(30) << "Context init"
      << std::setw(20) << "-"
      << std::setw(20) << "-"
      << cu.prep.ctx_init << "\n";

    line();

    f << std::setw(30) << "H2D"
      << std::setw(20) << "-"
      << std::setw(20) << "-"
      << cu.h2d << "\n";

    f << std::setw(30) << "Blur"
      << std::setw(20) << st.blur
      << std::setw(20) << mt.blur
      << cu.blur << "\n";

    f << std::setw(30) << "D2H"
      << std::setw(20) << "-"
      << std::setw(20) << "-"
      << cu.d2h << "\n";

    line();

    f << std::setw(30) << "TOTAL"
      << std::setw(20) << st.total
      << std::setw(20) << mt.total
      << cu.total << "\n";
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "podaj obraz\n";
        return 1;
    }

    int w, h, c;
    Timer tLoad;
    unsigned char* raw = stbi_load(argv[1], &w, &h, &c, 3);
    double loadT = tLoad.elapsed();

    Image img{ w, h };
    img.pixels.assign((Pixel*)raw, (Pixel*)raw + w * h);
    stbi_image_free(raw);

    int threads = std::thread::hardware_concurrency();
    PipelineResult st, mt, cu;

    {
        Timer t; Image out = img;
        st.prep.alloc_host = t.elapsed();

        Timer tb;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                long r=0,g=0,b=0; int c=0;
                for (int ky=-BLUR_RADIUS; ky<=BLUR_RADIUS; ky++)
                    for (int kx=-BLUR_RADIUS; kx<=BLUR_RADIUS; kx++) {
                        int ny = clampi(y+ky,0,h-1);
                        int nx = clampi(x+kx,0,w-1);
                        Pixel& p = img.pixels[ny*w+nx];
                        r+=p.r; g+=p.g; b+=p.b; c++;
                    }
                out.pixels[y*w+x] = {(byte)(r/c),(byte)(g/c),(byte)(b/c)};
            }
        st.blur = tb.elapsed();
        st.total = st.prep.alloc_host + st.blur;
    }

    {
        Timer t; Image out = img;
        mt.prep.alloc_host = t.elapsed();

        Timer tb;
        std::vector<std::thread> pool;
        for (int i=0;i<threads;i++) {
            pool.emplace_back([&,i]{
                int y0 = h*i/threads;
                int y1 = h*(i+1)/threads;
                for (int y=y0;y<y1;y++)
                    for (int x=0;x<w;x++) {
                        long r=0,g=0,b=0; int c=0;
                        for (int ky=-BLUR_RADIUS;ky<=BLUR_RADIUS;ky++)
                            for (int kx=-BLUR_RADIUS;kx<=BLUR_RADIUS;kx++) {
                                int ny=clampi(y+ky,0,h-1);
                                int nx=clampi(x+kx,0,w-1);
                                Pixel& p=img.pixels[ny*w+nx];
                                r+=p.r; g+=p.g; b+=p.b; c++;
                            }
                        out.pixels[y*w+x]={(byte)(r/c),(byte)(g/c),(byte)(b/c)};
                    }
            });
        }
        for (auto& t : pool) t.join();
        mt.blur = tb.elapsed();
        mt.total = mt.prep.alloc_host + mt.blur;
    }

    Image gpuOut = img;
    {
        Timer tc; 
        cudaDeviceSynchronize();
        cu.prep.ctx_init = tc.elapsed();

        Timer th; 
        cu.prep.alloc_host = th.elapsed();

        Timer td;
        Pixel *ds, *dd;
        size_t sz = w * h * sizeof(Pixel);
        cudaMalloc(&ds, sz);
        cudaMalloc(&dd, sz);
        cu.prep.alloc_device = td.elapsed();

        Timer th2d;
        cudaMemcpy(ds, img.pixels.data(), sz, cudaMemcpyHostToDevice);
        cu.h2d = th2d.elapsed();

        dim3 block(BLOCK, BLOCK);
        dim3 grid((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK);

        blurKernelShared<<<grid, block>>>(ds, dd, w, h);
        cudaDeviceSynchronize();

        const int ITERS = 200;

        Timer tB;
        for (int i = 0; i < ITERS; i++) {
            blurKernelShared<<<grid, block>>>(ds, dd, w, h);
        }
        cudaDeviceSynchronize();

        cu.blur = tB.elapsed() / ITERS;

        Timer td2h;
        cudaMemcpy(gpuOut.pixels.data(), dd, sz, cudaMemcpyDeviceToHost);
        cu.d2h = td2h.elapsed();

        cu.total = cu.prep.ctx_init +
                cu.prep.alloc_host +
                cu.prep.alloc_device +
                cu.h2d +
                cu.blur +
                cu.d2h;

        cudaFree(ds);
        cudaFree(dd);
    }

    Timer tSave;
    stbi_write_png("final_result.png", w, h, 3, gpuOut.pixels.data(), w * 3);
    double saveT = tSave.elapsed();

    saveReport("performance_shared_report.txt",
               w, h, threads,
               loadT, saveT,
               st, mt, cu);

    std::cout << "DONE\n";
    return 0;
}
