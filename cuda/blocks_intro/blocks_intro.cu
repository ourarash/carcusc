#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_blocks() {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block=%d thread=%d global=%d\n", blockIdx.x, threadIdx.x, global_index);
}

int main() {
    const int blocks = 3;
    const int threads_per_block = 4;
    printf("Launching %d blocks with %d threads each\n", blocks, threads_per_block);
    hello_blocks<<<blocks, threads_per_block>>>();
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    return 0;
}
