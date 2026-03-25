# Reduction Example

This example introduces a basic **parallel reduction** in CUDA using shared memory.

## What it teaches

- how blocks cooperate to reduce many values into one partial sum
- how `threadIdx.x` maps to positions in shared memory
- why `__syncthreads()` is needed during block-wide reductions
- how a GPU reduction often produces **partial sums per block** that can be combined on the CPU

## Build and Run

```bash
cd /home1/saifhash/carcusc/cuda/reduction
make
make run
```

## Expected Output

The program:

- computes a sum on the CPU
- computes block-level partial sums on the GPU
- copies the partial sums back to the host
- combines them on the CPU
- compares the two results

You should see output like:

```text
Reduction example
  Elements           : 1048576
  Blocks             : ...
  Threads per block  : 256
  CPU sum            : ...
  GPU sum            : ...
  Validation         : PASS
```

## Teaching Notes

This is intentionally a simple reduction, not a fully optimized one.
It is meant to teach the core ideas before introducing more advanced optimizations such as:

- loop unrolling
- fewer synchronizations
- warp-level primitives
- multi-stage GPU-only reductions
