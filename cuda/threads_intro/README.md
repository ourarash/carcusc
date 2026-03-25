# Threads Intro

This example introduces the most basic CUDA execution concept: **threads inside a single block**.

## What it teaches

- a kernel launch creates many threads
- `threadIdx.x` is the thread's index inside its block
- `blockIdx.x` stays `0` when only one block is launched

## Build and Run

```bash
cd /home1/saifhash/carcusc/cuda/threads_intro
make
make run
```

## Expected Output

The program launches **1 block with 8 threads** and prints one line per thread.
You should see output like:

```text
Launching 1 block with 8 threads
Hello from block 0, thread 0
Hello from block 0, thread 1
...
Hello from block 0, thread 7
```

The order of printed lines may vary slightly because threads execute in parallel.
