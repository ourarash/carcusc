# Setup

Use this folder to request an allocation, load a working CUDA environment, and run a smoke test on CARC.

## Fastest Way to Start

From the login node, run:

```bash
cd carcusc
./setup/start_cuda.sh
```

This assumes the repository is cloned into your home directory.

That command will:
1. Request a GPU allocation
2. Move you to a compute node
3. Load the CUDA-ready module environment automatically
4. Leave you in an interactive shell

## Default Mode

`./setup/start_cuda.sh` defaults to `debug-gpu`.

You can also choose a mode explicitly:

```bash
./setup/start_cuda.sh debug-gpu
./setup/start_cuda.sh debug-a40
./setup/start_cuda.sh debug-ebs
./setup/start_cuda.sh p100
./setup/start_cuda.sh l40s
./setup/start_cuda.sh a40
./setup/start_cuda.sh debug-cpu
```

`debug-ebs` targets `e23-02`, where EBS (event based sampling) is enabled for profiling.

## After the Launcher Starts

Verify you are on a compute node:

```bash
hostname
echo $SLURM_JOB_ID
nvidia-smi
```

Run the CUDA smoke test:

```bash
bash ./setup/test_cuda.sh
```

Run the profiling demos:

```bash
bash ./profiling/run_profile.sh
bash ./profiling/run_ncu.sh
bash ./profiling/run_ncu.sh --wait
```

Profiling outputs are written under `./profiling/results/`.
`run_profile.sh` using Nsight Systems is verified working here.
`run_ncu.sh` may fail on this cluster if GPU performance counters are restricted.

If you want to run `ncu`, do not use the default launcher mode. Start with the EBS-enabled node instead:

```bash
./setup/start_cuda.sh debug-ebs
```

Then, once you are on the compute node, run:

```bash
hostname
bash ./profiling/run_ncu.sh
```

You should expect `hostname` to start with `e23-02`.
That is the debug-partition node currently configured for EBS (event based sampling).

Short version:

```bash
cd carcusc
./setup/start_cuda.sh debug-ebs
bash ./profiling/run_ncu.sh
```

`run_ncu.sh` now checks whether `e23-02` is currently in use.
If it is busy, the script prints:
- current jobs on `e23-02`
- node details from `scontrol show node e23-02`
- a resource summary from `noderes`

By default it exits after showing that information.
Use `--wait` if you want it to keep polling until `e23-02` becomes free.

## Compile Your Own CUDA Program

```bash
nvcc my_program.cu -o my_program
./my_program
```

## Manual Workflow

If you want to do the steps yourself:

```bash
cd carcusc
./setup/gpu_alloc.sh debug-gpu
source ./setup/module.sh
bash ./setup/test_cuda.sh
```

## What Each File Does

- `start_cuda.sh` — one-command launcher
- `gpu_alloc.sh` — allocate a node and open a shell
- `module.sh` — load the CUDA/NVHPC/Python environment
- `test_cuda.sh` — compile and run a CUDA smoke test
- `setup.sh` — checklist/reference notes

## Profiling Folder

- `profiling/profile_demo.cu` — CUDA kernel used for profiling
- `profiling/run_profile.sh` — Nsight Systems (`nsys`) profiler runner
- `profiling/run_ncu.sh` — Nsight Compute (`ncu`) profiler runner
- `profiling/results/` — generated profiler reports

## When Finished

Exit the compute-node shell to release the allocation:

```bash
exit
```

If a job is queued or stuck, inspect and cancel it:

```bash
squeue --me
scancel <jobid>
```
