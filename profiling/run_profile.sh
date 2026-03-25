#!/usr/bin/env bash
# =============================================================================
# run_profile.sh — Compile and profile the demo CUDA program with Nsight Systems
#
# Usage:
#   bash /path/to/carcusc/profiling/run_profile.sh
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
REPORT_BASE="$RESULTS_DIR/profile_demo"
CCBIN="/usr/bin/g++"
PASS="✓"
FAIL="✗"
WARN="!"

log_section() { echo; echo "=========================================="; echo "  $1"; echo "=========================================="; }
log_ok() { echo "  $PASS  $1"; }
log_fail() { echo "  $FAIL  $1"; }
log_warn() { echo "  $WARN  $1"; }
log_info() { echo "      $1"; }

log_section "Environment"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    log_fail "No active Slurm allocation detected"
    echo "  Start one first with: $REPO_ROOT/setup/start_cuda.sh"
    exit 1
fi

source "$REPO_ROOT/setup/module.sh" >/tmp/profile_module.log 2>&1
cat /tmp/profile_module.log

if ! command -v nvcc >/dev/null 2>&1; then
    log_fail "nvcc not found after loading modules"
    exit 1
fi
if ! command -v nsys >/dev/null 2>&1; then
    log_fail "nsys not found after loading modules"
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
rm -f "$REPORT_BASE".nsys-rep "$REPORT_BASE".sqlite
log_info "Running: nsys profile --stats=true --force-overwrite=true -o $REPORT_BASE $BINARY_FILE"
if nsys profile --stats=true --force-overwrite=true -o "$REPORT_BASE" "$BINARY_FILE"; then
    log_ok "Profiling run succeeded"
else
    log_fail "Profiling run failed"
    exit 1
fi

log_section "Artifacts"
if [[ -f "$REPORT_BASE.nsys-rep" ]]; then
    log_ok "Report file: $REPORT_BASE.nsys-rep"
else
    log_fail "Expected report file was not created"
    exit 1
fi
if [[ -f "$REPORT_BASE.sqlite" ]]; then
    log_ok "SQLite export: $REPORT_BASE.sqlite"
fi

log_info "You can inspect the summary above or open the .nsys-rep file in Nsight Systems."
