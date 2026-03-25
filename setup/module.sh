# =============================================================================
# module.sh — Load environment modules for CUDA development
#
# Usage (must be SOURCED, not executed):
#   source ~/setup/module.sh
#   . ~/setup/module.sh
#
# Note: Run this after getting an allocation (gpu_alloc.sh), so that
#       GPU-related modules (cuda, nvhpc) are available on the compute node.
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Guard: warn if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Warning: this script must be sourced, not executed."
    echo "  Use:  source ${BASH_SOURCE[0]}"
    exit 1
fi

echo "==> Loading modules..."

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Warning: no active Slurm allocation detected."
    echo "  You are likely on a login node, so nvidia-smi will not work here."
    echo "  First run: $REPO_ROOT/setup/gpu_alloc.sh debug-gpu"
fi

module purge

# Clear any stale C/C++ include paths that may have been set by a previously
# loaded gcc/13.x module (e.g. from ~/.bashrc). nvhpc's EDG preprocessor was
# built with gcc-8.5.0 and cannot parse gcc-13 <type_traits> __has_builtin()
# patterns, causing compilation errors.
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH INCLUDE

# NVIDIA HPC SDK — bundles its own CUDA (12.4) and replaces gcc automatically.
# Do NOT load cuda/12.6.3 alongside nvhpc: nvhpc overrides gcc which cuda depends
# on, making cuda/12.6.3 inactive. Use nvhpc's bundled nvcc instead.
module load nvhpc/24.5

# Python
module load python/3.10.16

# The python module on this cluster re-exports compiler include paths under
# gcc/13.3.0. Those paths break nvcc preprocessing, so clear them again after
# all module loads are complete.
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH INCLUDE

# GUI libraries (libxkbcommon, qt, libx11, mesa) require extra prerequisites
# on this cluster and are only needed for Nsight GUI tools. Omitted here to
# keep the environment clean. Load them separately if needed:
#   module spider libxkbcommon/1.5.0

echo "==> Modules loaded:"
module list 2>&1 | grep -v "^$"

echo ""
echo "==> Versions:"
nvcc  --version 2>/dev/null | grep "release" || echo "  nvcc: not found (are you on a compute node?)"
python --version 2>/dev/null                 || echo "  python: not found"

# =============================================================================
# Useful cluster commands (for reference — not run automatically)
# =============================================================================
#
# Check free GPU nodes:
#   /apps/utilities/noderes -f -g
#
# Check all GPU nodes (detailed):
#   /apps/utilities/noderes -c -g
#
# View your jobs in the queue:
#   squeue --me
#
# View node/GPU info for a partition:
#   sinfo -p debug -o "%N %G %C"
#
# Check GPUs on current node:
#   nvidia-smi
