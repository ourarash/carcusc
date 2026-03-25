#!/usr/bin/env bash
# =============================================================================
# Slurm Interactive Allocation Cheat Sheet
# Copy ONE command at a time and run it directly in your terminal.
# Do NOT run this script top-to-bottom — salloc is interactive/blocking.
# =============================================================================

ACCOUNT="${CARC_ACCOUNT:-${USER}_1190}"

# --- GPU partition: P100 (8 CPUs, 32 GB) -------------------------------------
salloc --account="$ACCOUNT" \
       --time=01:00:00 \
       --cpus-per-task=8 \
       --mem=32G \
       --partition=gpu \
       --ntasks=1 \
       --gres=gpu:p100:1

# --- GPU partition: specific node b10-11, no GPU pinned ----------------------
#   (Add --gres=gpu:1 if you need a GPU here)
salloc --account="$ACCOUNT" \
       --time=01:00:00 \
       --cpus-per-task=8 \
       --mem=32G \
       --partition=gpu \
       --ntasks=1 \
       --nodelist=b10-11

# --- Debug partition: CPU only -----------------------------------------------
salloc --account="$ACCOUNT" \
       --time=00:30:00 \
       --cpus-per-task=4 \
       --mem=8G \
       --partition=debug \
       --ntasks=1

# --- Debug partition: 1 GPU (any type) ----------------------------------------
salloc --account="$ACCOUNT" \
       --time=00:30:00 \
       --cpus-per-task=4 \
       --mem=8G \
       --partition=debug \
       --ntasks=1 \
       --gres=gpu:1

# --- GPU partition: L40S (node b17-15) ----------------------------------------
salloc --account="$ACCOUNT" \
       --time=00:30:00 \
       --cpus-per-task=4 \
       --mem=8G \
       --partition=gpu \
       --ntasks=1 \
       --nodelist=b17-15 \
       --gres=gpu:l40s:1

# --- Debug partition: A40 on specific node b11-09 -----------------------------
salloc --account="$ACCOUNT" \
       --time=01:00:00 \
       --cpus-per-task=4 \
       --mem=8G \
       --partition=debug \
       --ntasks=1 \
       --nodelist=b11-09 \
       --gres=gpu:a40:1

# --- Debug partition: A40, any node -------------------------------------------
salloc --account="$ACCOUNT" \
       --time=01:00:00 \
       --cpus-per-task=4 \
       --mem=8G \
       --partition=debug \
       --ntasks=1 \
       --gres=gpu:a40:1

# =============================================================================
# Cluster inspection commands
# =============================================================================

# Show nodes, GPU types, and CPU state for the debug partition
sinfo -p debug -o "%N %G %C"

# List available nodes and GPUs (free)
/apps/utilities/noderes -f -g

# Detailed node/GPU availability
/apps/utilities/noderes -c -g

# Check which GPUs are visible on the allocated node
nvidia-smi
