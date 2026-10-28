#include <cstdio>
#include <cuda_runtime.h>
#include "hello.hpp"

__global__ void helloKernel() {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    printf("From GeForce RTX 5070 Ti: Global thread ID: %d, blockid: %d, blockDim: %d, threadIdx: %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
}

void launchHelloKernel() {
    // Launch kernel with 2 blocks, 4 threads each
    helloKernel<<<2, 4>>>();
}
