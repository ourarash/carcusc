#!/usr/bin/env bash
# =============================================================================
# test_cuda.sh — Write, compile, and run a minimal CUDA program to verify
#                the GPU environment is working correctly.
#
# Usage:
#   bash /path/to/carcusc/setup/test_cuda.sh
#
# Must be run inside a GPU allocation with modules already loaded:
#   source /path/to/carcusc/setup/module.sh
# =============================================================================

set -euo pipefail

WORKDIR=$(mktemp -d /tmp/cuda_test_XXXXXX)
SOURCE="$WORKDIR/hello_gpu.cu"
BINARY="$WORKDIR/hello_gpu"
PASS="✓"
FAIL="✗"
WARN="!"

log_section() { echo ""; echo "──────────────────────────────────────────"; echo "  $1"; echo "──────────────────────────────────────────"; }
log_ok()      { echo "  $PASS  $1"; }
log_fail()    { echo "  $FAIL  $1"; }
log_warn()    { echo "  $WARN  $1"; }
log_info()    { echo "      $1"; }

# =============================================================================
log_section "Environment Check"
# =============================================================================

# Check nvcc
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $NF}' | tr -d ',')
    log_ok "nvcc found: $NVCC_VER"
else
    log_fail "nvcc not found — did you run: source ~/setup/module.sh ?"
    exit 1
fi

# Check nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    log_ok "nvidia-smi found: $GPU_NAME ($GPU_MEM)"
else
    log_warn "nvidia-smi not found — continuing anyway"
fi

log_info "Work directory: $WORKDIR"

# Force nvcc to use the system GCC 8.5 toolchain.
# nvhpc 24.5 is compatible with GCC 8.5, while the cluster's GCC 13 headers
# trigger EDG preprocessor failures inside cuda_runtime.h.
CCBIN="/usr/bin/g++"
NVCC_FLAGS=(-O2)
if [[ -x "$CCBIN" ]]; then
    NVCC_FLAGS=(-ccbin "$CCBIN" -O2)
    log_ok "Host compiler: $CCBIN ($("$CCBIN" --version | head -1))"
else
    log_warn "Host compiler not found at $CCBIN — using default host compiler (may fail)"
fi

# =============================================================================
log_section "Writing CUDA Source"
# =============================================================================

cat > "$SOURCE" << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // Print device info
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("[GPU] %s  |  SM count: %d  |  Compute: %d.%d\n",
           prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = (float)(N - i); }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device, run kernel, copy back
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify: every element should equal N (i.e. i + (N-i) = N)
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != (float)N) errors++;
    }

    if (errors == 0)
        printf("[RESULT] All %d elements correct (each = %.0f)\n", N, (float)N);
    else
        printf("[RESULT] %d / %d elements WRONG\n", errors, N);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return errors > 0 ? 1 : 0;
}
EOF

log_ok "Source written: $SOURCE"

# =============================================================================
log_section "Compiling"
# =============================================================================

log_info "Running: nvcc ${NVCC_FLAGS[*]} $SOURCE -o $BINARY"

if nvcc "${NVCC_FLAGS[@]}" "$SOURCE" -o "$BINARY" 2>&1; then
    log_ok "Compilation succeeded"
else
    log_fail "Compilation failed"
    exit 1
fi

# =============================================================================
log_section "Running"
# =============================================================================

log_info "Running: $BINARY"
echo ""

OUTPUT=$("$BINARY" 2>&1)
STATUS=$?

echo "$OUTPUT" | sed 's/^/      /'
echo ""

if [[ $STATUS -eq 0 ]] && echo "$OUTPUT" | grep -q "All.*correct"; then
    log_ok "Program exited successfully"
    log_ok "Results verified — vector addition correct"
else
    log_fail "Program failed or produced incorrect results (exit code: $STATUS)"
    rm -rf "$WORKDIR"
    exit 1
fi

# =============================================================================
log_section "Summary"
# =============================================================================

log_ok "nvcc:        $(nvcc --version | grep 'release' | awk '{print $NF}' | tr -d ',')"
GPU_LINE=$(echo "$OUTPUT" | grep '\[GPU\]' | sed 's/\[GPU\] //')
log_ok "GPU:         $GPU_LINE"
log_ok "Kernel:      vector_add  (N=1024, blocks=4, threads=256)"
log_ok "Verified:    1024 / 1024 elements correct"
echo ""
echo "  All checks passed — CUDA environment is ready."
echo ""

rm -rf "$WORKDIR"
