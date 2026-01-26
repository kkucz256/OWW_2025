// #include <iostream>
// #include <vector>
// #include <string>
// #include <chrono>
// #include <thread>
// #include <fstream>
// #include <algorithm>
// #include <iomanip>

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

// using byte = uint8_t;
// const int BLUR_RADIUS = 15; 

// struct Pixel {
//     byte r, g, b;
//     bool operator==(const Pixel& other) const {
//         return r == other.r && g == other.g && b == other.b;
//     }
// };

// struct Image {
//     int width;
//     int height;
//     std::vector<Pixel> pixels;
// };

// class ImageIO {
// public:
//     static Image load(const std::string& filename) {
//         int w, h, c;
//         unsigned char* data = stbi_load(filename.c_str(), &w, &h, &c, 3);
//         if (!data) throw std::runtime_error("Cannot load image");
//         Image img{w, h};
//         img.pixels.resize(w * h);
//         for (size_t i = 0; i < w * h; ++i) {
//             img.pixels[i] = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
//         }
//         stbi_image_free(data);
//         return img;
//     }

//     static void save(const std::string& filename, const Image& img) {
//         std::vector<byte> raw;
//         raw.reserve(img.width * img.height * 3);
//         for (const auto& p : img.pixels) {
//             raw.push_back(p.r);
//             raw.push_back(p.g);
//             raw.push_back(p.b);
//         }
//         stbi_write_png(filename.c_str(), img.width, img.height, 3, raw.data(), img.width * 3);
//     }
// };

// class Processor {
//     static void blurRegion(const Image& src, Image& dst, int startY, int endY) {
//         int w = src.width;
//         int h = src.height;
//         for (int y = startY; y < endY; ++y) {
//             for (int x = 0; x < w; ++x) {
//                 long r = 0, g = 0, b = 0;
//                 int count = 0;
//                 for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ++ky) {
//                     for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; ++kx) {
//                         int ny = std::clamp(y + ky, 0, h - 1);
//                         int nx = std::clamp(x + kx, 0, w - 1);
//                         const Pixel& p = src.pixels[ny * w + nx];
//                         r += p.r; g += p.g; b += p.b;
//                         count++;
//                     }
//                 }
//                 dst.pixels[y * w + x] = {(byte)(r/count), (byte)(g/count), (byte)(b/count)};
//             }
//         }
//     }

// public:
//     static void applyBlur(const Image& src, Image& dst, int threads) {
//         dst = src;
//         std::vector<std::thread> pool;
//         int rows = src.height / threads;
//         for (int i = 0; i < threads; ++i) {
//             pool.emplace_back([&, i]() {
//                 blurRegion(src, dst, i * rows, (i == threads - 1) ? src.height : (i + 1) * rows);
//             });
//         }
//         for (auto& t : pool) t.join();
//     }
// };

// class RLE {
// public:
//     static std::vector<byte> compress(const Image& img) {
//         std::vector<byte> out;
//         out.reserve(img.pixels.size() * 2);
//         size_t n = img.pixels.size();
//         size_t i = 0;
//         while (i < n) {
//             Pixel p = img.pixels[i];
//             byte count = 1;
//             while (i + count < n && count < 255 && img.pixels[i + count] == p) count++;
//             out.push_back(count);
//             out.push_back(p.r); out.push_back(p.g); out.push_back(p.b);
//             i += count;
//         }
//         return out;
//     }

//     static Image decompress(const std::vector<byte>& data, int w, int h) {
//         Image img{w, h};
//         img.pixels.reserve(w * h);
//         size_t i = 0;
//         while (i < data.size()) {
//             byte count = data[i++];
//             Pixel p = {data[i++], data[i++], data[i++]};
//             for (int k = 0; k < count; ++k) img.pixels.push_back(p);
//         }
//         return img;
//     }

//     static void saveToFile(const std::string& fname, int w, int h, const std::vector<byte>& data) {
//         std::ofstream f(fname, std::ios::binary);
//         f.write((char*)&w, sizeof(w));
//         f.write((char*)&h, sizeof(h));
//         f.write((char*)data.data(), data.size());
//     }

//     static std::vector<byte> loadFromFile(const std::string& fname, int& w, int& h) {
//         std::ifstream f(fname, std::ios::binary | std::ios::ate);
//         size_t sz = f.tellg();
//         f.seekg(0);
//         f.read((char*)&w, sizeof(w));
//         f.read((char*)&h, sizeof(h));
//         std::vector<byte> data(sz - 8);
//         f.read((char*)data.data(), data.size());
//         return data;
//     }
// };

// int main(int argc, char* argv[]) {
//     if (argc < 2) return 1;
//     try {
//         std::string path = argv[1];
//         unsigned int threads = std::thread::hardware_concurrency();
//         if (threads == 0) threads = 4;

//         std::cout << "Loading: " << path << "\n";
//         Image input = ImageIO::load(path);
//         Image blurred = input;

//         std::cout << "Image size: " << input.width << "x" << input.height << "\n";
//         std::cout << "Operation: Heavy Blur (Radius " << BLUR_RADIUS << ")\n\n";

        
//         std::cout << "[1] Running Single Thread (Reference)...\n";
//         auto tStart1 = std::chrono::high_resolution_clock::now();
//         Processor::applyBlur(input, blurred, 1);
//         auto tEnd1 = std::chrono::high_resolution_clock::now();
//         double time1 = std::chrono::duration<double, std::milli>(tEnd1 - tStart1).count();
//         std::cout << "Single Thread Time: " << time1 << " ms\n\n";

//         std::cout << "[2] Running Multi Thread (" << threads << " threads)...\n";
//         auto tStart2 = std::chrono::high_resolution_clock::now();
//         Processor::applyBlur(input, blurred, threads);
//         auto tEnd2 = std::chrono::high_resolution_clock::now();
//         double time2 = std::chrono::duration<double, std::milli>(tEnd2 - tStart2).count();
//         std::cout << "Multi Thread Time:  " << time2 << " ms\n";

//         std::cout << "--------------------------------\n";
//         std::cout << "SPEEDUP: x" << std::fixed << std::setprecision(2) << (time1 / time2) << "\n";
//         std::cout << "--------------------------------\n\n";

//         std::cout << "Compressing RLE...\n";
//         auto compData = RLE::compress(blurred);
//         RLE::saveToFile("output.rle", blurred.width, blurred.height, compData);
//         std::cout << "Saved output.rle (" << compData.size() / 1024 << " KB)\n";

//         std::cout << "Decompressing verify...\n";
//         int w, h;
//         auto loadedData = RLE::loadFromFile("output.rle", w, h);
//         Image finalImg = RLE::decompress(loadedData, w, h);
        
//         ImageIO::save("final_output.png", finalImg);
//         std::cout << "Saved final_output.png. Success.\n";

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << "\n";
//         return 1;
//     }
//     return 0;
// }