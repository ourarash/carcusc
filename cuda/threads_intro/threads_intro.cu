#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_threads() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Launching 1 block with 8 threads\n");
    hello_threads<<<1, 8>>>();
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    return 0;
}
