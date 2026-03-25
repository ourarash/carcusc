# =============================================================================
# setup.sh — Workflow to get a GPU node and run a CUDA program
#
# Run each step manually in order.
# This file is a checklist, not an executable setup script.
# `nvidia-smi` only works after step 1 succeeds and your shell is on a compute
# node with an active Slurm allocation.
# =============================================================================

# 1. Get a GPU node
./setup/gpu_alloc.sh debug-gpu        # opens a shell on the compute node
#    Verify you are no longer on the login node:
#      hostname
#      echo $SLURM_JOB_ID
#    Expect a host like b11-09... and a non-empty SLURM_JOB_ID.

# 2. Load modules (inside that allocation shell)
source ./setup/module.sh

# 3. Compile
# nvcc my_program.cu -o my_program

# 4. Run
# ./my_program

# =============================================================================
# Architecture notes
# =============================================================================
#
# debug-gpu gives you any available GPU. If you need a specific architecture
# (e.g., A40 for sm_86), use debug-a40 instead so nvcc compiles for the right
# target, e.g.:
#
#   nvcc -arch=sm_86 my_program.cu -o my_program   # A40
#   nvcc -arch=sm_60 my_program.cu -o my_program   # P100
