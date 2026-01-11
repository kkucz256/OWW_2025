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
const int BLUR_RADIUS = 15; 

struct Pixel {
    byte r, g, b;
    bool operator==(const Pixel& other) const {
        return r == other.r && g == other.g && b == other.b;
    }
};

struct Image {
    int width;
    int height;
    std::vector<Pixel> pixels;
};

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

class ImageIO {
public:
    static Image load(const std::string& filename) {
        int w, h, c;
        unsigned char* data = stbi_load(filename.c_str(), &w, &h, &c, 3);
        if (!data) throw std::runtime_error("Cannot load image");
        Image img{w, h};
        img.pixels.resize(w * h);
        for (size_t i = 0; i < w * h; ++i) {
            img.pixels[i] = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
        }
        stbi_image_free(data);
        return img;
    }

    static void save(const std::string& filename, const Image& img) {
        std::vector<byte> raw;
        raw.reserve(img.width * img.height * 3);
        for (const auto& p : img.pixels) {
            raw.push_back(p.r);
            raw.push_back(p.g);
            raw.push_back(p.b);
        }
        stbi_write_png(filename.c_str(), img.width, img.height, 3, raw.data(), img.width * 3);
    }
};

class Processor {
    static void blurRegion(const Image& src, Image& dst, int startY, int endY) {
        int w = src.width;
        int h = src.height;
        for (int y = startY; y < endY; ++y) {
            for (int x = 0; x < w; ++x) {
                long r = 0, g = 0, b = 0;
                int count = 0;
                for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky) {
                    for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx) {
                        int ny = y + ky;
                        int nx = x + kx;
                        if (ny < 0) ny = 0;
                        if (ny >= h) ny = h - 1;
                        if (nx < 0) nx = 0;
                        if (nx >= w) nx = w - 1;

                        const Pixel& p = src.pixels[ny * w + nx];
                        r += p.r; g += p.g; b += p.b;
                        count++;
                    }
                }
                dst.pixels[y * w + x] = {(byte)(r/count), (byte)(g/count), (byte)(b/count)};
            }
        }
    }

public:
    static void applyBlur(const Image& src, Image& dst, int threads, double& outSpawn, double& outWork) {
        dst = src;
        std::vector<std::thread> pool;
        int rows = src.height / threads;
        
        Timer tSpawn;
        for (int i = 0; i < threads; ++i) {
            pool.emplace_back([&, i]() {
                blurRegion(src, dst, i * rows, (i == threads - 1) ? src.height : (i + 1) * rows);
            });
        }
        outSpawn = tSpawn.elapsedMs();

        Timer tWork;
        for (auto& t : pool) t.join();
        outWork = tWork.elapsedMs();
    }
};

__global__ void blurKernel(const Pixel* src, Pixel* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    long r = 0, g = 0, b = 0;
    int count = 0;
    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky) {
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx) {
            int ny = max(0, min(h - 1, y + ky));
            int nx = max(0, min(w - 1, x + kx));
            const Pixel& p = src[ny * w + nx];
            r += p.r; g += p.g; b += p.b;
            count++;
        }
    }
    dst[y * w + x] = {(byte)(r / count), (byte)(g / count), (byte)(b / count)};
}

class CudaProcessor {
public:
    static void applyBlur(const Image& src, Image& dst, double& outH2D, double& outKernel, double& outD2H) {
        dst = src; 
        size_t size = src.width * src.height * sizeof(Pixel);
        Pixel *d_src, *d_dst;
        cudaMalloc(&d_src, size);
        cudaMalloc(&d_dst, size);

        Timer tH2D;
        cudaMemcpy(d_src, src.pixels.data(), size, cudaMemcpyHostToDevice);
        outH2D = tH2D.elapsedMs();

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((src.width + 15) / 16, (src.height + 15) / 16);

        Timer tKernel;
        blurKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, src.width, src.height);
        cudaDeviceSynchronize();
        outKernel = tKernel.elapsedMs();

        Timer tD2H;
        cudaMemcpy(dst.pixels.data(), d_dst, size, cudaMemcpyDeviceToHost);
        outD2H = tD2H.elapsedMs();

        cudaFree(d_src); cudaFree(d_dst);
    }
};

class RLE {
public:
    static std::vector<byte> compress(const Image& img) {
        std::vector<byte> out;
        out.reserve(img.pixels.size() * 2);
        size_t n = img.pixels.size();
        size_t i = 0;
        while (i < n) {
            Pixel p = img.pixels[i];
            byte count = 1;
            while (i + count < n && count < 255 && img.pixels[i + count] == p) count++;
            out.push_back(count);
            out.push_back(p.r); out.push_back(p.g); out.push_back(p.b);
            i += count;
        }
        return out;
    }
    static Image decompress(const std::vector<byte>& data, int w, int h) {
        Image img{w, h};
        img.pixels.reserve(w * h);
        size_t i = 0;
        while (i < data.size()) {
            byte count = data[i++];
            Pixel p = {data[i++], data[i++], data[i++]};
            for (int k = 0; k < count; ++k) img.pixels.push_back(p);
        }
        return img;
    }
    static void saveToFile(const std::string& fname, int w, int h, const std::vector<byte>& data) {
        std::ofstream f(fname, std::ios::binary);
        f.write((char*)&w, sizeof(w)); f.write((char*)&h, sizeof(h));
        f.write((char*)data.data(), data.size());
    }
    static std::vector<byte> loadFromFile(const std::string& fname, int& w, int& h) {
        std::ifstream f(fname, std::ios::binary | std::ios::ate);
        size_t sz = f.tellg(); f.seekg(0);
        f.read((char*)&w, sizeof(w)); f.read((char*)&h, sizeof(h));
        std::vector<byte> data(sz - 8); f.read((char*)data.data(), data.size());
        return data;
    }
};

void saveFullReport(const std::string& fname, int w, int h, int threads, 
                    double s_total, double m_spawn, double m_work, 
                    double c_h2d, double c_kernel, double c_d2h, double rle_time) {
    std::ofstream r(fname);
    r << "PERFORMANCE ANALYSIS\n" << std::string(50, '=') << "\n";
    r << "Image: " << w << "x" << h << " | Threads: " << threads << " | Radius: " << BLUR_RADIUS << "\n\n";

    auto printStat = [&](std::string name, double time, std::string note = "") {
        r << std::left << std::setw(25) << name << ": " << std::fixed << std::setprecision(2) << std::setw(10) << time << " ms | " << note << "\n";
    };

    printStat("CPU Single-Thread", s_total, "Baseline");
    r << "\n[CPU Multi-Thread Analysis]\n";
    double m_total = m_spawn + m_work;
    printStat("Total Multi-Thread", m_total, "Speedup: " + std::to_string(s_total / m_total).substr(0,4) + "x");
    printStat("  > Thread Spawning", m_spawn, "OS Bottleneck");
    printStat("  > Actual Computing", m_work, "Pure logic");

    r << "\n[CUDA GPU Analysis]\n";
    double c_total = c_h2d + c_kernel + c_d2h;
    printStat("Total CUDA Time", c_total, "Speedup: " + std::to_string(s_total / c_total).substr(0,4) + "x");
    printStat("  > Host to Device", c_h2d, "PCIe Bottleneck!");
    printStat("  > GPU Kernel", c_kernel, "Calculation power");
    printStat("  > Device to Host", c_d2h, "Transfer back");

    r << "\n[Bonus: RLE Analysis]\n";
    printStat("RLE Compression", rle_time, "Single-threaded lag");

    r << "\nDIAGNOSIS:\n";
    if (m_spawn > m_work * 0.1) r << "- WARNING: Wątki systemowe tworzą się za długo względem pracy.\n";
    if (c_h2d + c_d2h > c_kernel) r << "- WARNING: Szyna PCIe zabija zysk z GPU. Algorytm zbyt prosty.\n";
    if (rle_time > c_total) r << "- WARNING: Kompresja RLE trwa dłużej niż cały blur na GPU!\n";

    r.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    try {
        std::string path = argv[1];
        unsigned int threads = std::thread::hardware_concurrency();
        Image input = ImageIO::load(path);
        Image blurred = input;

        double s_spawn, s_work, m_spawn, m_work, c_h2d, c_kernel, c_d2h, rle_time;

        std::cout << "[1] CPU Single...\n";
        Processor::applyBlur(input, blurred, 1, s_spawn, s_work);
        double s_total = s_work;

        std::cout << "[2] CPU Multi...\n";
        Processor::applyBlur(input, blurred, threads, m_spawn, m_work);

        std::cout << "[3] CUDA...\n";
        CudaProcessor::applyBlur(input, blurred, c_h2d, c_kernel, c_d2h);

        std::cout << "[4] RLE...\n";
        Timer tRle;
        auto compData = RLE::compress(blurred);
        rle_time = tRle.elapsedMs();
        RLE::saveToFile("output.rle", blurred.width, blurred.height, compData);

        std::cout << "Saving report...\n";
        saveFullReport("performance_report.txt", input.width, input.height, threads, s_total, m_spawn, m_work, c_h2d, c_kernel, c_d2h, rle_time);

        int w, h;
        auto loadedData = RLE::loadFromFile("output.rle", w, h);
        Image finalImg = RLE::decompress(loadedData, w, h);
        ImageIO::save("final_output.png", finalImg);
        std::cout << "Success.\n";

    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}