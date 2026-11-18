#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct LoadedTensor {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::byte> bytes;
};

class Model {
public:
    explicit Model(const std::filesystem::path& model_path);

    const nlohmann::json& header() const { return header_; }
    const nlohmann::json& config() const { return config_; }
    LoadedTensor load_tensor(const std::string& name);
    LoadedTensor load_row(const std::string& name, std::size_t row);

private:
    static std::size_t dtype_size(const std::string& dtype);

    struct TensorInfo {
        std::string dtype;
        std::vector<std::size_t> shape;
        std::uint64_t start;
        std::uint64_t end;
    };

    TensorInfo tensor_info(const std::string& name) const;
    std::vector<std::byte> read_bytes(std::uint64_t start, std::uint64_t size);

    std::ifstream file_;
    std::uint64_t data_start_ = 0;
    std::uint64_t file_size_ = 0;
    nlohmann::json header_;
    nlohmann::json config_;
};
