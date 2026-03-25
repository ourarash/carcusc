# Profiling

This folder contains a small CUDA workload and helper scripts for checking whether profiling works on the cluster.

## Files

- `profile_demo.cu` — simple SAXPY-style CUDA kernel with repeated launches
- `run_profile.sh` — profiles with Nsight Systems (`nsys`)
- `run_ncu.sh` — profiles with Nsight Compute (`ncu`)
- `results/` — generated profiler outputs

For a simpler non-profiling CUDA timing example, see `../cuda/vector_add/`.

## Recommended Node For Nsight Compute

Current EBS-enabled target node for `ncu`:

- `e23-02`

This is the node currently configured for event-based sampling in the debug partition. Availability is dynamic, so check it before trying to profile.

Check who is using the node and for how long:

```bash
squeue -w e23-02 -o "%.18i %.8u %.9P %.2t %.10M %.10l %.20S %.20e %R"
scontrol show node e23-02
/apps/utilities/noderes -c -g | grep -E '^e23-02\b|^NODE|e23-02'
```

## Usage

Start a shell on a compute node:

```bash
cd /home1/saifhash/carcusc
./setup/start_cuda.sh
```

For the EBS node specifically:

```bash
./setup/start_cuda.sh debug-ebs
```

### Nsight Systems

```bash
bash ./profiling/run_profile.sh
```

Expected output:
- compile succeeds
- `.nsys-rep` report is written under `profiling/results/`
- `.sqlite` export is generated

### Nsight Compute

```bash
bash ./profiling/run_ncu.sh
```

If you are not on `e23-02`, the script will:
- tell you which host you are on
- check whether `e23-02` is busy
- print current jobs on `e23-02`
- print node details and resource summary

To wait until `e23-02` becomes free:

```bash
bash ./profiling/run_ncu.sh --wait
```

Then start a fresh shell on that node:

```bash
./setup/start_cuda.sh debug-ebs
```

## Artifacts

Generated files are written under `profiling/results/`, for example:

- `profile_demo.nsys-rep`
- `profile_demo.sqlite`
- `profile_demo_ncu.ncu-rep`
- `profile_demo_ncu.txt`

## Current Status

- `nsys` is working in this environment.
- `ncu` requires the EBS-capable node and may still be restricted by cluster permissions depending on the node configuration at runtime.
