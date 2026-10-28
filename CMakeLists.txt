cmake_minimum_required(VERSION 3.25)

project(tinyinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(tinyinfer
    src/main.cpp
    src/kernels/hello.cu
)

target_link_libraries(tinyinfer PRIVATE CUDA::cudart CUDA::cublas)
