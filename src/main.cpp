#include <array>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>

nlohmann::json read_safetensors_header(std::ifstream& file) {
    // Get the first 8 bytes and parse as a uint64_t - the size of the JSON header
    std::array<std::uint8_t, 8> bytes{};
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
        throw std::runtime_error("File is shorter than 8 bytes");
    }

    std::uint64_t header_size = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        header_size |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    if (header_size > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Header is too large");
    }

    // Now using the header size, we read the entire JSON header
    std::string header(static_cast<std::size_t>(header_size), '\0');
    if (!file.read(header.data(), static_cast<std::streamsize>(header_size))) {
        throw std::runtime_error("Could not read the complete JSON header");
    }

    // Save the JSON header to a file for inspection temporarily
    std::ofstream header_json("header.json");
    if (!header_json || !(header_json << header)) {
        throw std::runtime_error("Could not write header.json");
    }

    // Parse the JSON header so callers can use it to locate the model weights
    try {
        return nlohmann::json::parse(header);
    } catch (const nlohmann::json::parse_error& error) {
        throw std::runtime_error(std::string("Invalid SafeTensors JSON header: ") + error.what());
    }
}

int tokeniser_experiment() {
    // load the tokeniser from the model folder and use sentencepiece to encode some input string
    const auto tokenizer_path = std::filesystem::path(TINYINFER_SOURCE_DIR)
                          / "models" / "tinyllama-1.1b" / "tokenizer.model";

    sentencepiece::SentencePieceProcessor tokenizer;
    const auto load_status = tokenizer.Load(tokenizer_path.string());
    if (!load_status.ok()) {
        std::cerr << "Could not load tokenizer: " << load_status.ToString() << '\n';
        return 1;
    }

    const std::string input = "Hello, world! Hello , world !";
    std::vector<int> token_ids;
    const auto encode_status = tokenizer.Encode(input, &token_ids);
    if (!encode_status.ok()) {
        std::cerr << "Could not tokenize input: " << encode_status.ToString() << '\n';
        return 1;
    }
    std::cout << '"' << input << "\"= ";
    // Print out all the token_ids
    for (int token_id : token_ids) {
        std::cout << token_id << ' ';
    }

    //TODO: We need to load in the model.embed_tokens.weight section so we can perform a lookup
    // to retrieve the embedding vector that these token ids map to
    std::cout << '\n';
    return 0;
}


int main() {
    // Detect path of model present in models directory - Set in CMakeLists.txt
    const auto path = std::filesystem::path(TINYINFER_SOURCE_DIR)
                    / "models" / "tinyllama-1.1b"
                    / "model.safetensors";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Could not open " << path << '\n';
        return 1;
    }

    nlohmann::json parsed_header;
    try {
        parsed_header = read_safetensors_header(file);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    // Model weights begin from this point on; data_offsets are relative to it
    const std::streampos tensor_data_start = file.tellg();
    std::cout << std::string(40, '=') << "Model Header" << std::string(40, '=') << '\n';
    // Loop through the parsed JSON header to print out the tensors
    for (auto it = parsed_header.begin(); it != parsed_header.end(); ++it) {
        if (it.key() == "__metadata__") continue;

        const auto& tensor = it.value();
        std::cout << it.key()
                << " dtype=" << tensor.at("dtype").get<std::string>()
                << " shape=" << tensor.at("shape").dump()
                << " offsets=" << tensor.at("data_offsets").dump()
                << '\n';
    }

    std::cout << "Tensor data starts at byte: " << tensor_data_start << '\n';
    return tokeniser_experiment();
}
