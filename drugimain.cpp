#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <thread>
#include <numeric>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using byte = uint8_t;

const int SCALE_FACTOR = 50; 

std::string removeExtension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

struct Image {
    int width;
    int height;
    std::vector<byte> pixels;
};

class ImageProcessor {
public:
    static Image loadImage(const std::string& filename) {
        int w, h, channels;
        unsigned char* data = stbi_load(filename.c_str(), &w, &h, &channels, 3);
        if (!data) throw std::runtime_error("STB Error: Could not load image");

        Image img;
        img.width = w;
        img.height = h;
        img.pixels.assign(data, data + (w * h * 3));
        stbi_image_free(data);
        return img;
    }

    static void saveRLE(const std::string& filename, int width, int height, const std::vector<byte>& compressedData) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("File error");
        file.write(reinterpret_cast<const char*>(&width), sizeof(width));
        file.write(reinterpret_cast<const char*>(&height), sizeof(height));
        file.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());
    }
};

class RLECodec {
private:
    static void compressRowToBuffer(const Image& img, int y, std::vector<byte>& buffer) {
        size_t offset = (size_t)y * img.width * 3;
        const byte* row = &img.pixels[offset];
        
        int x = 0;
        int width = img.width;

        while (x < width) {
            byte r = row[x*3];
            byte g = row[x*3+1];
            byte b = row[x*3+2];
            byte count = 1;

            while ((x + count) < width && count < 255) {
                int next = (x + count) * 3;
                if (row[next] == r && row[next+1] == g && row[next+2] == b) count++;
                else break;
            }
            buffer.push_back(count);
            buffer.push_back(r);
            buffer.push_back(g);
            buffer.push_back(b);
            x += count;
        }
    }

public:
    static std::vector<byte> compressSingleThread(const Image& img) {
        std::vector<byte> output;
        output.reserve(img.pixels.size() / 2); 
        for (int y = 0; y < img.height; y++) {
            compressRowToBuffer(img, y, output);
        }
        return output;
    }

    static std::vector<byte> compressMultiThreadOptimized(const Image& img, unsigned int numThreads, double& executionTimeMs) {
        int height = img.height;
        int rowsPerThread = height / numThreads;
        
        std::vector<std::vector<byte>> threadBuffers(numThreads);
        std::vector<std::thread> threads;

        auto start = std::chrono::high_resolution_clock::now();

        for (unsigned int i = 0; i < numThreads; i++) {
            threads.emplace_back([&, i]() {
                int startY = i * rowsPerThread;
                int endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
                int myRows = endY - startY;
                threadBuffers[i].reserve((size_t)myRows * img.width * 3 / 2);

                for (int y = startY; y < endY; y++) {
                    compressRowToBuffer(img, y, threadBuffers[i]);
                }
            });
        }

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        executionTimeMs = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<byte> finalOutput;
        size_t totalSize = 0;
        for (const auto& buf : threadBuffers) totalSize += buf.size();
        
        finalOutput.reserve(totalSize);
        for (const auto& buf : threadBuffers) {
            finalOutput.insert(finalOutput.end(), buf.begin(), buf.end());
        }

        return finalOutput;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image>\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string baseName = removeExtension(inputFile);
    std::string rleFile = baseName + "_optimized_x" + std::to_string(SCALE_FACTOR) + ".rle";
    
    try {
        Image original = ImageProcessor::loadImage(inputFile);
        
        std::cout << "Generating x" << SCALE_FACTOR << " payload...\n";
        Image megaImage;
        megaImage.width = original.width;
        megaImage.height = original.height * SCALE_FACTOR;
        
        size_t originalSize = original.pixels.size();
        megaImage.pixels.reserve(originalSize * SCALE_FACTOR);
        
        for(int i = 0; i < SCALE_FACTOR; i++) {
            megaImage.pixels.insert(megaImage.pixels.end(), original.pixels.begin(), original.pixels.end());
        }

        unsigned int threads = std::thread::hardware_concurrency(); 
        if(threads == 0) threads = 4;

        std::cout << "Threads: " << threads << "\n";
        std::cout << "Workload: " << megaImage.width << "x" << megaImage.height 
                  << " (" << (megaImage.pixels.size() / (1024*1024)) << " MB)\n";
        std::cout << "Method: Static Partitioning (Block-based, Pre-allocated)\n\n";

        std::cout << "Running Single Thread...\n";
        auto start1 = std::chrono::high_resolution_clock::now();
        RLECodec::compressSingleThread(megaImage);
        auto end1 = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration<double, std::milli>(end1 - start1).count();
        std::cout << "Single-thread time: " << time1 << " ms\n";

        std::cout << "Running Multi Thread...\n";
        double time2 = 0.0;
        std::vector<byte> compressed2 = RLECodec::compressMultiThreadOptimized(megaImage, threads, time2);
        
        std::cout << "Multi-thread EXEC time: " << time2 << " ms\n";

        std::cout << "--------------------------------\n";
        std::cout << "SPEEDUP: x" << std::fixed << std::setprecision(2) << (time1/time2) << "\n";
        std::cout << "--------------------------------\n";

        ImageProcessor::saveRLE(rleFile, megaImage.width, megaImage.height, compressed2);
        std::cout << "Saved: " << rleFile << "\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}