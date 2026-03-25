# CUDA Examples

This folder contains small, focused CUDA programs for students learning GPU programming.

## Examples

- `threads_intro/` — introduces thread indexing within a single block
- `blocks_intro/` — shows how blocks and global indices work
- `grid_stride_intro/` — demonstrates grid-stride loops for scalable kernels
- `vector_add/` — compares CPU and GPU vector addition timing
- `reduction/` — introduces shared-memory reduction and block partial sums

## Recommended Order

1. `threads_intro`
2. `blocks_intro`
3. `grid_stride_intro`
4. `vector_add`
5. `reduction`

## General Usage

Start on a compute node with the repository setup flow:

```bash
cd /home1/saifhash/carcusc
./setup/start_cuda.sh
```

Then enter any example directory:

```bash
cd cuda/threads_intro
make
make run
```
