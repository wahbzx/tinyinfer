#include <array>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

int main() {
    // Detect path of model present in models directory - Set in CMakeLists.txt
    const auto path = std::filesystem::path(TINYINFER_SOURCE_DIR)
                    / "models" / "tinyllama-1.1b"
                    / "model.safetensors";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Could not open file\n";
        return 1;
    }

    // Get the first 8 bytes and parse as a unit64_t - the size of the json header
    std::array<std::uint8_t, 8> bytes{};
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
        std::cerr << "File is shorter than 8 bytes\n";
        return 1;
    }

    std::uint64_t headerSize = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        headerSize |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    std::cout << "Header size: " << headerSize << " bytes\n";
    
    if (headerSize > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamsize>::max())) {
        std::cerr << "Header is too large\n";
        return 1;
    }

    // Now using the header size, we read the entire json header
    std::string header(static_cast<std::size_t>(headerSize), '\0');
    
    if (!file.read(header.data(), static_cast<std::streamsize>(headerSize))) {
        std::cerr << "Could not read the complete JSON header\n";
        return 1;
    }

    // Save the json header to a file for inspection temporarily
    std::ofstream header_json("header.json");
    header_json << header;

    std::cout << header << "\nCurrent position in file: "<< file.tellg() << '\n';
    // Model Weights begin from this point on

    return 0;
}