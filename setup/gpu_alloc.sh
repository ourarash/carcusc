#!/usr/bin/env bash
# =============================================================================
# gpu_alloc.sh — Request an interactive GPU allocation by mode
#
# Usage:
#   ./gpu_alloc.sh <mode>
#
# Modes:
#   debug-cpu    Debug partition, CPU only
#                  ./gpu_alloc.sh debug-cpu
#
#   debug-gpu    Debug partition, 1 GPU (any type)
#                  ./gpu_alloc.sh debug-gpu
#
#   debug-a40    Debug partition, A40 GPU
#                  ./gpu_alloc.sh debug-a40
#
#   debug-ebs    Debug partition, GPU on e23-02 (EBS enabled for ncu)
#                  ./gpu_alloc.sh debug-ebs
#
#   p100         GPU partition, P100 (8 CPUs, 32 GB)
#                  ./gpu_alloc.sh p100
#
#   l40s         GPU partition, L40S (node b17-15)
#                  ./gpu_alloc.sh l40s
#
#   a40          GPU partition, A40 GPU
#                  ./gpu_alloc.sh a40
# =============================================================================

set -euo pipefail

ACCOUNT="${CARC_ACCOUNT:-${USER}_1190}"

launch_shell() {
  exec salloc "$@" srun --pty bash -l
}

usage() {
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
}

[[ $# -eq 1 ]] || { echo "Error: exactly one mode argument required."; usage; }

MODE="$1"

case "$MODE" in

  debug-cpu)
    launch_shell \
      --account="$ACCOUNT" \
      --time=00:30:00 \
      --cpus-per-task=4 \
      --mem=8G \
      --partition=debug \
      --ntasks=1
    ;;

  debug-gpu)
    launch_shell \
      --account="$ACCOUNT" \
      --time=00:30:00 \
      --cpus-per-task=4 \
      --mem=8G \
      --partition=debug \
      --ntasks=1 \
      --gres=gpu:1
    ;;

  debug-a40)
    launch_shell \
      --account="$ACCOUNT" \
      --time=01:00:00 \
      --cpus-per-task=4 \
      --mem=8G \
      --partition=debug \
      --ntasks=1 \
      --gres=gpu:a40:1
    ;;

  debug-ebs)
    launch_shell \
      --account="$ACCOUNT" \
      --time=01:00:00 \
      --cpus-per-task=4 \
      --mem=8G \
      --partition=debug \
      --ntasks=1 \
      --nodelist=e23-02 \
      --gres=gpu:1
    ;;

  p100)
    launch_shell \
      --account="$ACCOUNT" \
      --time=01:00:00 \
      --cpus-per-task=8 \
      --mem=32G \
      --partition=gpu \
      --ntasks=1 \
      --gres=gpu:p100:1
    ;;

  l40s)
    launch_shell \
      --account="$ACCOUNT" \
      --time=00:30:00 \
      --cpus-per-task=4 \
      --mem=8G \
      --partition=gpu \
      --ntasks=1 \
      --nodelist=b17-15 \
      --gres=gpu:l40s:1
    ;;

  a40)
    launch_shell \
      --account="$ACCOUNT" \
      --time=01:00:00 \
      --cpus-per-task=4 \
      --mem=8G \
      --partition=gpu \
      --ntasks=1 \
      --gres=gpu:a40:1
    ;;

  *)
    echo "Error: unknown mode '$MODE'."
    usage
    ;;

esac
