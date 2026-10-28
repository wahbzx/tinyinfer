#include <stdio.h>
#include <cuda_runtime.h>
#include "kernels/hello.hpp"

int main() {
    launchHelloKernel();
    
    // Force CPU to wait for GPU to finish and flush printf buffer
    // If you don't program will quit before GPU can flush the buffer
    cudaDeviceSynchronize();
    
    return 0;
}
