#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {

constexpr int kNumElements = 1 << 20;
constexpr int kThreadsPerBlock = 256;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void reduce_sum(const float* input, float* partial_sums, int count) {
    __shared__ float shared[kThreadsPerBlock];

    unsigned int tid = threadIdx.x;
    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (global_index < count) ? input[global_index] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

}  // namespace

int main() {
    const size_t bytes = static_cast<size_t>(kNumElements) * sizeof(float);
    const int blocks = (kNumElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const size_t partial_bytes = static_cast<size_t>(blocks) * sizeof(float);

    float* host_input = static_cast<float*>(std::malloc(bytes));
    float* host_partials = static_cast<float*>(std::malloc(partial_bytes));
    if (!host_input || !host_partials) {
        std::fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < kNumElements; ++i) {
        host_input[i] = 1.0f + static_cast<float>(i % 5) * 0.25f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_sum = 0.0;
    for (int i = 0; i < kNumElements; ++i) {
        cpu_sum += host_input[i];
    }
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    float* device_input = nullptr;
    float* device_partials = nullptr;
    check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(device_input)");
    check_cuda(cudaMalloc(&device_partials, partial_bytes), "cudaMalloc(device_partials)");
    check_cuda(cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D input");

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)");

    reduce_sum<<<blocks, kThreadsPerBlock>>>(device_input, device_partials, kNumElements);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");

    float gpu_kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    check_cuda(cudaMemcpy(host_partials, device_partials, partial_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H partials");

    double gpu_sum = 0.0;
    for (int i = 0; i < blocks; ++i) {
        gpu_sum += host_partials[i];
    }

    std::printf("Reduction example\n");
    std::printf("  Elements           : %d\n", kNumElements);
    std::printf("  Blocks             : %d\n", blocks);
    std::printf("  Threads per block  : %d\n", kThreadsPerBlock);
    std::printf("  CPU sum            : %.3f\n", cpu_sum);
    std::printf("  GPU sum            : %.3f\n", gpu_sum);
    std::printf("  CPU time           : %.3f ms\n", cpu_ms);
    std::printf("  GPU kernel time    : %.3f ms\n", gpu_kernel_ms);
    std::printf("  Validation         : %s\n", std::fabs(cpu_sum - gpu_sum) < 1e-2 ? "PASS" : "FAIL");

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(device_input);
    cudaFree(device_partials);
    std::free(host_input);
    std::free(host_partials);
    return std::fabs(cpu_sum - gpu_sum) < 1e-2 ? 0 : 1;
}
