#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void saxpy_kernel(float alpha, const float* x, const float* y, float* out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = alpha * x[index] + y[index];
    }
}

static void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    const int num_elements = 1 << 24;
    const size_t num_bytes = static_cast<size_t>(num_elements) * sizeof(float);
    const int iterations = 200;
    const float alpha = 2.5f;

    int device = 0;
    cudaDeviceProp prop{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::printf("[GPU] %s | compute %d.%d | SMs %d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("[INFO] elements=%d bytes=%zu iterations=%d\n",
                num_elements, num_bytes, iterations);

    float* host_x = static_cast<float*>(std::malloc(num_bytes));
    float* host_y = static_cast<float*>(std::malloc(num_bytes));
    float* host_out = static_cast<float*>(std::malloc(num_bytes));
    if (!host_x || !host_y || !host_out) {
        std::fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < num_elements; ++i) {
        host_x[i] = static_cast<float>(i % 1000) * 0.5f;
        host_y[i] = static_cast<float>((i * 3) % 1000) * 0.25f;
    }

    float *dev_x = nullptr, *dev_y = nullptr, *dev_out = nullptr;
    check_cuda(cudaMalloc(&dev_x, num_bytes), "cudaMalloc(dev_x)");
    check_cuda(cudaMalloc(&dev_y, num_bytes), "cudaMalloc(dev_y)");
    check_cuda(cudaMalloc(&dev_out, num_bytes), "cudaMalloc(dev_out)");

    check_cuda(cudaMemcpy(dev_x, host_x, num_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D x");
    check_cuda(cudaMemcpy(dev_y, host_y, num_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D y");

    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)");
    for (int iter = 0; iter < iterations; ++iter) {
        saxpy_kernel<<<blocks, threads_per_block>>>(alpha, dev_x, dev_y, dev_out, num_elements);
    }
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");

    check_cuda(cudaMemcpy(host_out, dev_out, num_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H out");

    const int sample_index = 12345;
    const float expected = alpha * host_x[sample_index] + host_y[sample_index];
    const float actual = host_out[sample_index];
    std::printf("[RESULT] sample[%d] expected=%.3f actual=%.3f\n", sample_index, expected, actual);
    std::printf("[TIMING] total kernel time: %.3f ms\n", elapsed_ms);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_out);
    std::free(host_x);
    std::free(host_y);
    std::free(host_out);

    return (std::abs(expected - actual) < 1e-5f) ? 0 : 1;
}
