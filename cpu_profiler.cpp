#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <random>

using byte = uint8_t;

constexpr int BLUR_RADIUS = 32;

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Pixel {
    byte r, g, b;
    bool operator==(const Pixel& o) const { return r == o.r && g == o.g && b == o.b; }
};

struct Image { int width, height; std::vector<Pixel> pixels; };

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() { return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count(); }
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

int main() {
    int w = 2048, h = 2048;
    Image img{ w, h }; img.pixels.resize(w * h);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& p : img.pixels) p = {(byte)dist(rng), (byte)dist(rng), (byte)dist(rng)};

    int ths = std::thread::hardware_concurrency();
    std::cout << "[INFO] Start profilowania CPU (" << ths << " watkow) dla obrazu " << w << "x" << h << "\n";

    Timer t_total;

    Image out = img;
    std::vector<std::thread> pool;
    for (int i=0; i<ths; i++) {
        pool.emplace_back([&, i] {
            int y0 = h*i/ths, y1 = h*(i+1)/ths;
            for (int y=y0; y<y1; y++) {
                for (int x=0; x<w; x++) {
                    long r=0,g=0,b=0;
                    for (int ky=-BLUR_RADIUS; ky<=BLUR_RADIUS; ky++) {
                        for (int kx=-BLUR_RADIUS; kx<=BLUR_RADIUS; kx++) {
                            int ny = clampi(y+ky,0,h-1);
                            int nx = clampi(x+kx,0,w-1);
                            const Pixel& p = img.pixels[ny*w+nx];
                            r+=p.r; g+=p.g; b+=p.b;
                        }
                    }
                    int cnt = (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1);
                    out.pixels[y*w+x] = {(byte)(r/cnt),(byte)(g/cnt),(byte)(b/cnt)};
                }
            }
        });
    }
    for (auto& th : pool) th.join();

    std::cout << "[INFO] Blur CPU zakonczony. Czas: " << t_total.elapsed() << " ms\n";
    return 0;
}