#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {

constexpr int kNumElements = 1 << 24;
constexpr int kThreadsPerBlock = 256;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        c[index] = a[index] + b[index];
    }
}

}  // namespace

int main() {
    const size_t bytes = static_cast<size_t>(kNumElements) * sizeof(float);

    float* host_a = static_cast<float*>(std::malloc(bytes));
    float* host_b = static_cast<float*>(std::malloc(bytes));
    float* host_cpu = static_cast<float*>(std::malloc(bytes));
    float* host_gpu = static_cast<float*>(std::malloc(bytes));
    if (!host_a || !host_b || !host_cpu || !host_gpu) {
        std::fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < kNumElements; ++i) {
        host_a[i] = static_cast<float>(i % 1000) * 0.5f;
        host_b[i] = static_cast<float>((i * 7) % 1000) * 0.25f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kNumElements; ++i) {
        host_cpu[i] = host_a[i] + host_b[i];
    }
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    check_cuda(cudaMalloc(&dev_a, bytes), "cudaMalloc(dev_a)");
    check_cuda(cudaMalloc(&dev_b, bytes), "cudaMalloc(dev_b)");
    check_cuda(cudaMalloc(&dev_c, bytes), "cudaMalloc(dev_c)");

    cudaEvent_t total_start;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    cudaEvent_t total_stop;
    check_cuda(cudaEventCreate(&total_start), "cudaEventCreate(total_start)");
    check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate(kernel_start)");
    check_cuda(cudaEventCreate(&kernel_stop), "cudaEventCreate(kernel_stop)");
    check_cuda(cudaEventCreate(&total_stop), "cudaEventCreate(total_stop)");

    check_cuda(cudaEventRecord(total_start), "cudaEventRecord(total_start)");
    check_cuda(cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
    check_cuda(cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D b");

    const int blocks = (kNumElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord(kernel_start)");
    vector_add_kernel<<<blocks, kThreadsPerBlock>>>(dev_a, dev_b, dev_c, kNumElements);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventRecord(kernel_stop), "cudaEventRecord(kernel_stop)");
    check_cuda(cudaEventSynchronize(kernel_stop), "cudaEventSynchronize(kernel_stop)");

    check_cuda(cudaMemcpy(host_gpu, dev_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");
    check_cuda(cudaEventRecord(total_stop), "cudaEventRecord(total_stop)");
    check_cuda(cudaEventSynchronize(total_stop), "cudaEventSynchronize(total_stop)");

    float gpu_kernel_ms = 0.0f;
    float gpu_total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_kernel_ms, kernel_start, kernel_stop), "cudaEventElapsedTime(kernel)");
    check_cuda(cudaEventElapsedTime(&gpu_total_ms, total_start, total_stop), "cudaEventElapsedTime(total)");

    int mismatches = 0;
    for (int i = 0; i < kNumElements; ++i) {
        if (std::fabs(host_cpu[i] - host_gpu[i]) > 1e-5f) {
            ++mismatches;
            if (mismatches < 5) {
                std::fprintf(stderr, "Mismatch at %d: cpu=%f gpu=%f\n", i, host_cpu[i], host_gpu[i]);
            }
        }
    }

    int device = 0;
    cudaDeviceProp prop{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::printf("Vector add benchmark\n");
    std::printf("  Elements          : %d\n", kNumElements);
    std::printf("  GPU               : %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("  CPU time          : %.3f ms\n", cpu_ms);
    std::printf("  GPU kernel time   : %.3f ms\n", gpu_kernel_ms);
    std::printf("  GPU total time    : %.3f ms (includes copies)\n", gpu_total_ms);
    if (gpu_kernel_ms > 0.0f) {
        std::printf("  Speedup (kernel)  : %.2fx\n", cpu_ms / gpu_kernel_ms);
    }
    if (gpu_total_ms > 0.0f) {
        std::printf("  Speedup (total)   : %.2fx\n", cpu_ms / gpu_total_ms);
    }
    std::printf("  Validation        : %s\n", mismatches == 0 ? "PASS" : "FAIL");

    cudaEventDestroy(total_start);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(total_stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    std::free(host_a);
    std::free(host_b);
    std::free(host_cpu);
    std::free(host_gpu);

    return mismatches == 0 ? 0 : 1;
}
