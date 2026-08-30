#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/summaries
{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo UNKNOWN)"
  echo "compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo UNKNOWN)"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo UNKNOWN)"
  echo "nvcc=$(nvcc --version 2>/dev/null | tail -1 || echo UNKNOWN)"
  echo "g++=$(g++ --version 2>/dev/null | head -1 || echo UNKNOWN)"
  echo "cmake=$(cmake --version 2>/dev/null | head -1 || echo UNKNOWN)"
  echo "ncu=$(ncu --version 2>/dev/null | tail -1 || echo unavailable)"
  echo "nsys=$(nsys --version 2>/dev/null || echo unavailable)"
  echo "wsl_device=$(test -e /dev/dxg && echo present || echo absent)"
} | tee results/summaries/environment.txt
