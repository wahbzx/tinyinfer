#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sentencepiece_processor.h>
#include "model.hpp"

int tokeniser_experiment(Model& model) {
    // load the tokeniser from the model folder and use sentencepiece to encode some input string
    const auto tokenizer_path = std::filesystem::path(TINYINFER_SOURCE_DIR)
                          / "models" / "tinyllama-1.1b" / "tokenizer.model";

    sentencepiece::SentencePieceProcessor tokenizer;
    const auto load_status = tokenizer.Load(tokenizer_path.string());
    if (!load_status.ok()) {
        std::cerr << "Could not load tokenizer: " << load_status.ToString() << '\n';
        return 1;
    }

    const std::string input = "Hello, world!";
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

    std::cout << '\n';
    // Look up tokens in the embedding table
    for (int token_id : token_ids) {
        const auto embedding = model.load_row("model.embed_tokens.weight",
                                              static_cast<std::size_t>(token_id));
        std::cout << "Token " << token_id << " embedding: " << embedding.shape[0]
                  << " " << embedding.dtype << " values (" << embedding.bytes.size() << " bytes)\n";
    }
    return 0;
}


int main() {
    // Detect path of model present in models directory - Set in CMakeLists.txt
    const auto path = std::filesystem::path(TINYINFER_SOURCE_DIR)
                    / "models" / "tinyllama-1.1b"
                    / "model.safetensors";
    try {
        Model model(path);
        std::cout << "Vocabulary size: " << model.config().at("vocab_size") << '\n';
        return tokeniser_experiment(model);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
