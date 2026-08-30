#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/summaries
OUT="results/summaries/environment_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "os=$(grep '^PRETTY_NAME=' /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '\"' || echo UNKNOWN)"
  echo "wsl_status=$(test -e /dev/dxg && echo WSL2_GPU_DEVICE_PRESENT || (grep -qi microsoft /proc/version 2>/dev/null && echo WSL_DETECTED_NO_DXG || echo NOT_WSL))"
  echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo UNKNOWN)"
  echo "gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null || echo UNKNOWN)"
  echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo UNKNOWN)"
  echo "nvcc=$(nvcc --version 2>/dev/null | tail -1 || echo UNKNOWN)"
  echo "g++=$(g++ --version 2>/dev/null | head -1 || echo UNKNOWN)"
  echo "cmake=$(cmake --version 2>/dev/null | head -1 || echo UNKNOWN)"
  echo "ncu=$(ncu --version 2>/dev/null | tail -1 || echo unavailable)"
  echo "nsys=$(nsys --version 2>/dev/null | head -1 || echo unavailable)"
} | tee "$OUT"
printf 'Environment snapshot: %s\n' "$OUT"
