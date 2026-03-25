#!/usr/bin/env bash
# =============================================================================
# run_ncu.sh — Compile and profile the demo CUDA program with Nsight Compute
#
# Usage:
#   bash /path/to/carcusc/profiling/run_ncu.sh
#   bash /path/to/carcusc/profiling/run_ncu.sh --wait
#
# Run this inside an active GPU allocation.
# Results are written to profiling/results/ under the repo root.
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ROOT_DIR="$REPO_ROOT/profiling"
RESULTS_DIR="$ROOT_DIR/results"
SOURCE_FILE="$ROOT_DIR/profile_demo.cu"
BINARY_FILE="$ROOT_DIR/profile_demo"
REPORT_BASE="$RESULTS_DIR/profile_demo_ncu"
CCBIN="/usr/bin/g++"
TARGET_NODE="e23-02"
WAIT_FOR_NODE=0
PASS="✓"
FAIL="✗"
WARN="!"

log_section() { echo; echo "=========================================="; echo "  $1"; echo "=========================================="; }
log_ok() { echo "  $PASS  $1"; }
log_fail() { echo "  $FAIL  $1"; }
log_warn() { echo "  $WARN  $1"; }
log_info() { echo "      $1"; }

usage() {
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait)
            WAIT_FOR_NODE=1
            shift
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

show_target_node_usage() {
    echo
    log_info "Current jobs on $TARGET_NODE:"
    squeue -w "$TARGET_NODE" -o '%.18i %.8u %.9P %.2t %.10M %.10l %.20S %.20e %R' || true
    echo
    log_info "Node details for $TARGET_NODE:"
    scontrol show node "$TARGET_NODE" || true
    echo
    log_info "Resource summary for $TARGET_NODE:"
    /apps/utilities/noderes -c -g | grep -E "^$TARGET_NODE\\b|^NODE|$TARGET_NODE" || true
}

target_node_busy() {
    squeue -h -w "$TARGET_NODE" -t R -o '%i' | grep -q .
}

wait_for_target_node() {
    if ! target_node_busy; then
        log_ok "$TARGET_NODE is currently free"
        return 0
    fi

    log_warn "$TARGET_NODE is currently in use"
    show_target_node_usage

    if [[ $WAIT_FOR_NODE -eq 0 ]]; then
        echo
        echo "  Re-run with --wait to wait for $TARGET_NODE to become free."
        echo "  Example: bash $REPO_ROOT/profiling/run_ncu.sh --wait"
        exit 2
    fi

    echo
    log_info "Waiting for $TARGET_NODE to become free... checking every 30 seconds"
    while target_node_busy; do
        sleep 30
        echo
        log_info "Still waiting on $TARGET_NODE"
        squeue -w "$TARGET_NODE" -o '%.18i %.8u %.9P %.2t %.10M %.10l %.20S %.20e %R' || true
    done

    echo
    log_ok "$TARGET_NODE is now free"
}

log_section "Environment"
CURRENT_HOST=$(hostname -s 2>/dev/null || hostname)

if [[ "$CURRENT_HOST" != "$TARGET_NODE" ]]; then
    log_warn "Nsight Compute should be run on $TARGET_NODE where EBS is enabled"
    log_info "Current host: $CURRENT_HOST"
    wait_for_target_node
    echo
    echo "  You are not on $TARGET_NODE. Start a fresh shell there with:"
    echo "    $REPO_ROOT/setup/start_cuda.sh debug-ebs"
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    log_fail "No active Slurm allocation detected"
    echo "  Start one first with: $REPO_ROOT/setup/start_cuda.sh"
    exit 1
fi

source "$REPO_ROOT/setup/module.sh" >/tmp/profile_ncu_module.log 2>&1
cat /tmp/profile_ncu_module.log

if ! command -v nvcc >/dev/null 2>&1; then
    log_fail "nvcc not found after loading modules"
    exit 1
fi
if ! command -v ncu >/dev/null 2>&1; then
    log_fail "ncu not found after loading modules"
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_warn "nvidia-smi not found in PATH, but continuing"
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    log_ok "GPU visible: ${GPU_NAME:-unknown}"
fi

mkdir -p "$RESULTS_DIR"

log_section "Compile"
log_info "Running: nvcc -ccbin $CCBIN -O2 -lineinfo $SOURCE_FILE -o $BINARY_FILE"
if nvcc -ccbin "$CCBIN" -O2 -lineinfo "$SOURCE_FILE" -o "$BINARY_FILE"; then
    log_ok "Compilation succeeded"
else
    log_fail "Compilation failed"
    exit 1
fi

log_section "Profile"
rm -f "$REPORT_BASE" "$REPORT_BASE".ncu-rep "$REPORT_BASE".csv
log_info "Running: ncu --set full --kernel-name saxpy_kernel --target-processes all --force-overwrite --export $REPORT_BASE $BINARY_FILE"
NCU_LOG=$(mktemp /tmp/ncu_run_XXXXXX.log)
if ncu --set full --kernel-name saxpy_kernel --target-processes all --force-overwrite --export "$REPORT_BASE" "$BINARY_FILE" 2>&1 | tee "$NCU_LOG"; then
    log_ok "Nsight Compute run succeeded"
else
    if grep -q "ERR_NVGPUCTRPERM" "$NCU_LOG"; then
        log_fail "Nsight Compute cannot access GPU performance counters on this cluster"
        echo "  The CUDA program ran, but kernel metrics collection is blocked by system permissions."
        echo "  Ask the cluster admins whether Nsight Compute counters can be enabled for user jobs."
    else
        log_fail "Nsight Compute run failed"
    fi
    rm -f "$NCU_LOG"
    exit 1
fi
rm -f "$NCU_LOG"

log_section "Artifacts"
if [[ -f "$REPORT_BASE.ncu-rep" ]]; then
    log_ok "Report file: $REPORT_BASE.ncu-rep"
else
    log_fail "Expected .ncu-rep file was not created"
    exit 1
fi

log_info "Generating text summary"
if ncu --import "$REPORT_BASE.ncu-rep" --page details > "$REPORT_BASE.txt"; then
    log_ok "Text summary: $REPORT_BASE.txt"
else
    log_warn "Could not generate text summary from report"
fi

log_info "Open the .ncu-rep file in Nsight Compute for full kernel metrics."
