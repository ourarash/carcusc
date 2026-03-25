# Blocks Intro

This example shows how CUDA organizes work into **multiple blocks**, each containing multiple threads.

## What it teaches

- `blockIdx.x` identifies which block a thread belongs to
- `threadIdx.x` identifies a thread within its block
- `global_index = blockIdx.x * blockDim.x + threadIdx.x` maps a thread to a unique 1D position

## Build and Run

```bash
cd carcusc
cd cuda/blocks_intro
make
make run
```

## Expected Output

The program launches **3 blocks** with **4 threads per block**.
You should see global indices from `0` to `11`.

Example:

```text
Launching 3 blocks with 4 threads each
block=0 thread=0 global=0
block=0 thread=1 global=1
...
block=2 thread=3 global=11
```

The line order may vary, but each `(block, thread)` pair should appear exactly once.
