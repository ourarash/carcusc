#!/usr/bin/env bash
# =============================================================================
# start_cuda.sh — One-command launcher for a ready-to-use CUDA shell
#
# Usage:
#   ./setup/start_cuda.sh [mode]
#
# Examples:
#   ./setup/start_cuda.sh
#   ./setup/start_cuda.sh debug-a40
#   ./setup/start_cuda.sh debug-ebs
#   ./setup/start_cuda.sh p100
#
# Default mode:
#   debug-gpu
#
# What it does:
#   1. Requests a Slurm allocation
#   2. Starts an interactive shell on the compute node
#   3. Sources the repo's setup/module.sh automatically
#   4. Prints quick verification commands
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export REPO_ROOT

ACCOUNT="${CARC_ACCOUNT:-${USER}_1190}"
MODE="${1:-debug-gpu}"

usage() {
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
}

alloc_args() {
    case "$MODE" in
        debug-cpu)
            cat <<EOF
--account=$ACCOUNT
--time=00:30:00
--cpus-per-task=4
--mem=8G
--partition=debug
--ntasks=1
EOF
            ;;
        debug-gpu)
            cat <<EOF
--account=$ACCOUNT
--time=00:30:00
--cpus-per-task=4
--mem=8G
--partition=debug
--ntasks=1
--gres=gpu:1
EOF
            ;;
        debug-a40)
            cat <<EOF
--account=$ACCOUNT
--time=01:00:00
--cpus-per-task=4
--mem=8G
--partition=debug
--ntasks=1
--gres=gpu:a40:1
EOF
            ;;
        debug-ebs)
            cat <<EOF
--account=$ACCOUNT
--time=01:00:00
--cpus-per-task=4
--mem=8G
--partition=debug
--ntasks=1
--nodelist=e23-02
--gres=gpu:1
EOF
            ;;
        p100)
            cat <<EOF
--account=$ACCOUNT
--time=01:00:00
--cpus-per-task=8
--mem=32G
--partition=gpu
--ntasks=1
--gres=gpu:p100:1
EOF
            ;;
        l40s)
            cat <<EOF
--account=$ACCOUNT
--time=00:30:00
--cpus-per-task=4
--mem=8G
--partition=gpu
--ntasks=1
--nodelist=b17-15
--gres=gpu:l40s:1
EOF
            ;;
        a40)
            cat <<EOF
--account=$ACCOUNT
--time=01:00:00
--cpus-per-task=4
--mem=8G
--partition=gpu
--ntasks=1
--gres=gpu:a40:1
EOF
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "Error: unknown mode '$MODE'."
            usage
            ;;
    esac
}

mapfile -t ARGS < <(alloc_args)

READY_CMD=$(cat <<'EOF'
cd "$REPO_ROOT"
source "$REPO_ROOT/setup/module.sh"

echo
echo "Ready on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
if [[ "$(hostname)" == e23-02* ]]; then
    echo "EBS-capable node detected: Nsight Compute sampling should be available here."
fi
echo
echo "Suggested checks:"
echo "  nvidia-smi"
echo "  bash $REPO_ROOT/setup/test_cuda.sh"
echo "  bash $REPO_ROOT/profiling/run_profile.sh"
echo "  bash $REPO_ROOT/profiling/run_ncu.sh"
echo
echo "Exit this shell to release the allocation."
echo
exec bash -i
EOF
)

exec salloc "${ARGS[@]}" srun --pty bash -lc "$READY_CMD"
