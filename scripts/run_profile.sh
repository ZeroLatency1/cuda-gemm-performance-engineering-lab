#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p profiles
if ! command -v ncu >/dev/null 2>&1; then
  echo 'Nsight Compute (ncu) not found; profiling unavailable.' >&2
  exit 2
fi
set +e
ncu --set basic --launch-count 1 --export profiles/register_2048 --force \
  ./build/cuda_gemm_benchmark --M 2048 --N 2048 --K 2048 --kernel register --warmup 10 --iterations 50 --no-verify --experiment-name profile_register_2048
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  printf 'Nsight Compute could not complete. Exit code: %s\n' "$rc" > profiles/PROFILING_LIMITATION.txt
  exit "$rc"
fi
