#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill_with_grid_stride(int* output, int count) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = global_index; index < count; index += stride) {
        output[index] = global_index;
    }
}

int main() {
    const int count = 20;
    const int blocks = 2;
    const int threads_per_block = 4;
    int host_output[count] = {0};
    int* device_output = nullptr;

    cudaError_t error = cudaMalloc(&device_output, sizeof(int) * count);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    fill_with_grid_stride<<<blocks, threads_per_block>>>(device_output, count);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(error));
        cudaFree(device_output);
        return 1;
    }

    error = cudaMemcpy(host_output, device_output, sizeof(int) * count, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(error));
        cudaFree(device_output);
        return 1;
    }

    std::printf("Grid-stride example with %d blocks and %d threads per block\n", blocks, threads_per_block);
    std::printf("Stride = blocks * threads_per_block = %d\n", blocks * threads_per_block);
    for (int i = 0; i < count; ++i) {
        std::printf("output[%2d] written by global thread %d\n", i, host_output[i]);
    }

    cudaFree(device_output);
    return 0;
}
