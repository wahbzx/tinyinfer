cmake_minimum_required(VERSION 3.25)

project(tinyinfer LANGUAGES CXX CUDA)

# Language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)

# Dependencies
include(FetchContent)

# nlohmann/json - For parsing json (not too interested in writing my own json parser)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(nlohmann_json)

# SentencePiece - For Tokenisation (Maybe replace in the future - maybe not :) )
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(sentencepiece)

# Executable
add_executable(tinyinfer
    src/main.cpp
    src/kernels/hello.cu
)

# Dependencies and libraries
target_link_libraries(tinyinfer PRIVATE
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
    sentencepiece-static
)

# Include directories
target_include_directories(tinyinfer PRIVATE
    ${sentencepiece_SOURCE_DIR}/src
    ${sentencepiece_BINARY_DIR}/src
)

# Compile definitions
target_compile_definitions(tinyinfer PRIVATE
    TINYINFER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)