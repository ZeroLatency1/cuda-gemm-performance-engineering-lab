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
  ncu_bin=""
  if command -v ncu >/dev/null 2>&1; then
    ncu_bin="$(command -v ncu)"
  else
    for candidate in \
      "/usr/local/NVIDIA-Nsight-Compute/target/linux-desktop-glibc_2_11_3-x64/ncu" \
      "/usr/local/NVIDIA-Nsight-Compute-2026.2/target/linux-desktop-glibc_2_11_3-x64/ncu" \
      "/opt/nvidia/nsight-compute/2024.3.2/target/linux-desktop-glibc_2_11_3-x64/ncu"; do
      if [[ -x "$candidate" ]]; then ncu_bin="$candidate"; break; fi
    done
  fi
  if [[ -n "$ncu_bin" ]]; then
    echo "ncu_bin=$ncu_bin"
    echo "ncu=$($ncu_bin --version 2>/dev/null | tail -1 || echo unavailable)"
  else
    echo "ncu_bin=UNAVAILABLE"
    echo "ncu=unavailable"
  fi
  echo "nsys=$(nsys --version 2>/dev/null | head -1 || echo unavailable)"
} | tee "$OUT"
printf 'Environment snapshot: %s\n' "$OUT"
