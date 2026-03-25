# Vector Add Benchmark

This example compares a simple vector addition workload on the CPU and GPU.

It is intended as a minimal CUDA benchmark for:

- validating that `nvcc` builds correctly in the current environment
- measuring CPU execution time for a baseline implementation
- measuring GPU kernel time and end-to-end GPU time
- checking correctness between CPU and GPU results

## Files

- `vector_add.cu` — benchmark source code
- `Makefile` — build and run targets

## Build

Run this on a compute node with the CARC CUDA environment loaded:

```bash
cd carcusc
./setup/start_cuda.sh
cd cuda/vector_add
make
```

The build uses:

- `nvcc`
- `/usr/bin/g++` as the host compiler
- `-O2 -lineinfo`

## Run

```bash
make run
```

## Output

The program prints:

- number of elements processed
- active GPU model and compute capability
- CPU time in milliseconds
- GPU kernel time in milliseconds
- GPU total time including copies
- simple speedup estimates
- validation status

Typical usage:

```bash
cd carcusc
cd cuda/vector_add
make clean
make
make run
```

## Clean

```bash
make clean
```

This removes the generated `vector_add` binary.