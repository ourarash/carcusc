# Grid-Stride Intro

This example shows how a **grid** of blocks can cover more data than there are threads by using a **grid-stride loop**.

## What it teaches

- `gridDim.x` is the number of blocks in the grid
- `blockDim.x` is the number of threads in each block
- `stride = gridDim.x * blockDim.x` tells each thread how far to jump to process more elements
- grid-stride loops are a standard CUDA pattern for scalable kernels

## Build and Run

```bash
cd carcusc
cd cuda/grid_stride_intro
make
make run
```

## Expected Output

The program launches **2 blocks** with **4 threads per block**, so the stride is `8`.
It fills an array of length `20`, showing which global thread handled each output position.

You should see entries where the same thread writes multiple positions separated by 8.
For example, a thread with global index `3` may write indices `3`, `11`, and `19`.
