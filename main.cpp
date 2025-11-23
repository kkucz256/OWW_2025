#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using byte = uint8_t;

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
        
        if (!data) {
            const char* reason = stbi_failure_reason();
            std::string msg = "BŁĄD STB: [" + std::string(reason) + "] plik: " + filename;
            throw std::runtime_error(msg);
        }

        Image img;
        img.width = w;
        img.height = h;
        img.pixels.assign(data, data + (w * h * 3));

        stbi_image_free(data);
        std::cout << "[LOAD] Wczytano: " << filename << " (" << w << "x" << h << ")\n";
        return img;
    }

    static void saveToPPM(const Image& img, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return;

        file << "P6\n" << img.width << " " << img.height << "\n255\n";
        file.write(reinterpret_cast<const char*>(img.pixels.data()), img.pixels.size());
        
        std::cout << "[PPM] Zapisano obraz: " << filename << "\n";
    }
    
    static void saveRLE(const std::string& filename, int width, int height, const std::vector<byte>& compressedData) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Nie mozna zapisac pliku RLE");
        
        file.write(reinterpret_cast<const char*>(&width), sizeof(width));
        file.write(reinterpret_cast<const char*>(&height), sizeof(height));
        file.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());
        std::cout << "[RLE] Zapisano wynik: " << filename << "\n";
    }

    static bool loadRLE(const std::string& filename, int& width, int& height, std::vector<byte>& data) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size < 8) return false;

        file.read(reinterpret_cast<char*>(&width), sizeof(width));
        file.read(reinterpret_cast<char*>(&height), sizeof(height));

        std::streamsize dataSize = size - sizeof(int) * 2;
        data.resize(dataSize);
        file.read(reinterpret_cast<char*>(data.data()), dataSize);
        
        return true;
    }
};

class RLECodec {
public:
    static std::vector<byte> compress(const Image& img) {
        std::vector<byte> output;
        output.reserve(img.pixels.size() / 2);

        for (int y = 0; y < img.height; y++) {
            const byte* row = &img.pixels[y * img.width * 3];
            int x = 0;
            while (x < img.width) {
                byte r = row[x*3];
                byte g = row[x*3+1];
                byte b = row[x*3+2];
                byte count = 1;

                while ((x + count) < img.width && count < 255) {
                    int next = (x + count) * 3;
                    if (row[next] == r && row[next+1] == g && row[next+2] == b) count++;
                    else break;
                }
                output.push_back(count);
                output.push_back(r);
                output.push_back(g);
                output.push_back(b);
                x += count;
            }
        }
        return output;
    }

    static Image decompress(const std::vector<byte>& rleData, int width, int height) {
        Image img;
        img.width = width;
        img.height = height;
        img.pixels.reserve(width * height * 3);

        size_t idx = 0;
        size_t n = rleData.size();

        while (idx < n) {
            if (idx + 4 > n) break;

            byte count = rleData[idx++];
            byte r = rleData[idx++];
            byte g = rleData[idx++];
            byte b = rleData[idx++];

            for (int i = 0; i < count; i++) {
                img.pixels.push_back(r);
                img.pixels.push_back(g);
                img.pixels.push_back(b);
            }
        }
        return img;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Uzycie: " << argv[0] << " <obraz.jpg/png>\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string baseName = removeExtension(inputFile);
    std::string rleFile = baseName + ".rle";
    std::string decodedFile = baseName + "_decoded.ppm";

    try {
        Image original = ImageProcessor::loadImage(inputFile);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<byte> compressed = RLECodec::compress(original);
        auto end = std::chrono::high_resolution_clock::now();
        
        ImageProcessor::saveRLE(rleFile, original.width, original.height, compressed);

        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        double ratio = (double)compressed.size() / (double)original.pixels.size() * 100.0;
        
        std::cout << "\n--- WYNIKI KOMPRESJI ---\n";
        std::cout << "Czas: " << duration << " ms\n";
        std::cout << "Ratio: " << ratio << "% (oryginał: " << original.pixels.size()/1024 << "KB -> RLE: " << compressed.size()/1024 << "KB)\n";

        std::cout << "\n--- WERYFIKACJA ---\n";
        std::cout << "Otwieram plik .rle i dekoduję...\n";
        
        int w, h;
        std::vector<byte> dataFromFile;
        if (!ImageProcessor::loadRLE(rleFile, w, h, dataFromFile)) {
            throw std::runtime_error("Nie udalo sie wczytac stworzonego pliku RLE!");
        }

        Image decoded = RLECodec::decompress(dataFromFile, w, h);
        ImageProcessor::saveToPPM(decoded, decodedFile);

        std::cout << "SUKCES! Zapisano odzyskany obraz jako: " << decodedFile << "\n";
        std::cout << "Teraz mozesz otworzyc plik _decoded.ppm i porownac go z oryginalem.\n";

    } catch (const std::exception& e) {
        std::cerr << "BŁĄD: " << e.what() << "\n";
        return 1;
    }

    return 0;
}