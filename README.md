# carcusc

CUDA setup, profiling, and benchmark helpers for USC CARC cluster workflows.

This repository is prepared for **EE508 - Hardware Foundations for Machine Learning** and is intended to make it straightforward to:

- request GPU nodes on CARC
- load a working CUDA toolchain
- validate CUDA execution end to end
- run basic profiling experiments with Nsight Systems and Nsight Compute
- compare simple CPU and GPU implementations of CUDA examples

## Overview

The repository is organized around three practical tasks:

- `setup/` for cluster allocation and environment setup
- `profiling/` for Nsight-based profiling workflows
- `cuda/` for small standalone CUDA programs and benchmarks

The goal is not to be a generic CUDA framework. It is a lightweight, course-oriented working environment for cluster-based CUDA experiments.

## Repository Layout

- `setup/` — Slurm allocation helpers, module setup, and CUDA smoke tests
- `profiling/` — CUDA profiling demo plus `nsys` and `ncu` runners
- `cuda/` — standalone CUDA examples with simple build targets

## Quick Start

🚀 From the login node:

First connect to USC CARC:

```bash
ssh <username>@discovery.usc.edu
```

You need to be on the USC VPN before connecting.
For CARC documentation and cluster information, see: https://www.carc.usc.edu/

```bash
cd /home1/saifhash/carcusc
./setup/start_cuda.sh
```

This will:

1. request a GPU allocation
2. open an interactive shell on the compute node
3. load the CUDA module environment automatically

Then verify the environment:

```bash
nvidia-smi
bash ./setup/test_cuda.sh
```

## Running Profilers

### Nsight Systems

📈 Run:

```bash
bash ./profiling/run_profile.sh
```

This path is already verified working in the current environment.

### Nsight Compute

🔬 For `ncu`, use the EBS-enabled node path:

```bash
./setup/start_cuda.sh debug-ebs
bash ./profiling/run_ncu.sh
```

If you need to wait for the EBS node to become available:

```bash
bash ./profiling/run_ncu.sh --wait
```

Additional profiling details are in `profiling/README.md`.

## CUDA Examples

The `cuda/` folder contains small, self-contained CUDA programs intended for learning, benchmarking, and experimentation.

Current examples:

- `cuda/threads_intro/` — one block, multiple threads
- `cuda/blocks_intro/` — multiple blocks and global indexing
- `cuda/grid_stride_intro/` — grid-stride loop pattern
- `cuda/vector_add/` — CPU vs GPU timing for vector addition
- `cuda/reduction/` — shared-memory reduction with block partial sums

### Vector Add Benchmark

🧮 The repository includes a simple CPU-vs-GPU benchmark at:

- `cuda/vector_add/`

Build and run it on a compute node with modules loaded:

```bash
cd cuda/vector_add
make
make run
```

The program reports:

- CPU execution time
- GPU kernel execution time
- GPU total time including host-device transfers
- correctness validation
- simple speedup estimates

Example workflow:

```bash
cd /home1/saifhash/carcusc
./setup/start_cuda.sh
cd cuda/threads_intro
make
make run
```

Additional example-specific details are in:

- `cuda/README.md`
- `cuda/vector_add/README.md`
- `cuda/reduction/README.md`

## Common Launcher Modes

- `./setup/start_cuda.sh` — default debug GPU allocation
- `./setup/start_cuda.sh debug-ebs` — target `e23-02` for `ncu` / EBS
- `./setup/start_cuda.sh debug-a40` — request an A40 on debug
- `./setup/start_cuda.sh p100` — request a P100 on the GPU partition
- `./setup/start_cuda.sh debug-cpu` — CPU-only debug allocation

## Notes

💡 Useful reminders:

- `nvidia-smi` works only on a compute node with an active allocation.
- `nvcc` is configured to use `/usr/bin/g++` to avoid the GCC 13 header mismatch seen with the cluster toolchain.
- `run_profile.sh` is verified working.
- `run_ncu.sh` depends on EBS / GPU performance counter availability.

## Documentation

- `setup/README.md` — startup workflow and allocation modes
- `profiling/README.md` — profiling workflow and EBS node guidance
- `cuda/vector_add/README.md` — build and run instructions for the vector-add benchmark

## License

📄 This project is licensed under the MIT License. See `LICENSE` for details.
