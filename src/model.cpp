#include "model.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

// For now we only support BF16 as thats the only tensor type used for our TinyLlama model
std::size_t Model::dtype_size(const std::string& dtype) {
    if (dtype == "BF16") return 2;
    throw std::runtime_error("Unsupported tensor dtype: " + dtype);
}


// Model Constructor using the model path
Model::Model(const std::filesystem::path& model_path)
    : file_(model_path, std::ios::binary) {
    if (!file_) throw std::runtime_error("Could not open " + model_path.string());
    
    // Store the total file size
    file_size_ = std::filesystem::file_size(model_path);
    if (file_size_ < 8)
        throw std::runtime_error("SafeTensors file is shorter than 8 bytes");
    
    
    // Check if the model header has 8 bytes reserved for the header size, otherwise this file is invalid 
    std::array<unsigned char, 8> size_bytes{};
    if (!file_.read(reinterpret_cast<char*>(size_bytes.data()), size_bytes.size()))
        throw std::runtime_error("SafeTensors file is shorter than 8 bytes");
    
    // Attempt to read the first 8 bytes which contain the size of the model's header size
    // Need to use a uint64 in case the file is above 4GiB otherwise we may have an integer overflow
    std::uint64_t header_size = 0;
    for (std::size_t i = 0; i < size_bytes.size(); ++i)
        header_size |= std::uint64_t{size_bytes[i]} << (8 * i);

    // Now we attempt to read the header json based on the header size specified
    std::string header_text(static_cast<std::size_t>(header_size), '\0');
    if (!file_.read(header_text.data(), static_cast<std::streamsize>(header_size)))
        throw std::runtime_error("Could not read SafeTensors header");
    header_ = nlohmann::json::parse(header_text);
    
    // Record the number of bytes into the file the data starts
    data_start_ = size_bytes.size() + header_size;

    // Also load the config.json that comes with the model that stores important parameters
    const auto config_path = model_path.parent_path() / "config.json";
    std::ifstream config_file(config_path);
    if (!config_file) throw std::runtime_error("Could not open " + config_path.string());
    config_ = nlohmann::json::parse(config_file);
}

// Load the relevant tensor info on demand to load the data
Model::TensorInfo Model::tensor_info(const std::string& name) const {
    if (!header_.contains(name) || name == "__metadata__")
        throw std::out_of_range("Tensor not found: " + name);
    
    const auto& entry = header_.at(name);
    TensorInfo info{entry.at("dtype").get<std::string>(),
                    entry.at("shape").get<std::vector<std::size_t>>(),
                    entry.at("data_offsets").at(0).get<std::uint64_t>(),
                    entry.at("data_offsets").at(1).get<std::uint64_t>()};

    if (info.start > info.end || info.end > file_size_ - data_start_)
        throw std::runtime_error("Invalid offsets for tensor: " + name);

    return info;
}

std::vector<std::byte> Model::read_bytes(std::uint64_t start, std::uint64_t size) {
    if (start > file_size_ || size > file_size_ - start)
        throw std::runtime_error("Tensor loaded in exceeds the size of file");

    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(start));
    file_.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));

    // In case the read fails
    if (!file_)
        throw std::runtime_error("Could not read tensor data");
    return bytes;
}

// Load in the raw tensor data
LoadedTensor Model::load_tensor(const std::string& name) {
    auto info = tensor_info(name);
    return {std::move(info.dtype), std::move(info.shape),
            read_bytes(data_start_ + info.start, info.end - info.start)};
}

// Load a single row of a tensor which helps us easily lookup the embeddings
LoadedTensor Model::load_row(const std::string& name, std::size_t row) {
    auto info = tensor_info(name);
    if (info.shape.size() != 2 || row >= info.shape[0])
        throw std::out_of_range("Invalid row for tensor: " + name);

    const std::uint64_t row_bytes = info.shape[1] * dtype_size(info.dtype);
    return {std::move(info.dtype), {info.shape[1]},
            read_bytes(data_start_ + info.start + row * row_bytes, row_bytes)};
}
