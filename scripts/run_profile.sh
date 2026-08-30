#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p profiles profiles/limitations
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="profiles/register_${STAMP}"
LIMIT="profiles/limitations/PROFILING_${STAMP}.txt"

if ! command -v ncu >/dev/null 2>&1; then
  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool=nsight_compute"
    echo "status=UNAVAILABLE"
    echo "command=ncu --set basic --launch-count 1 ./build/cuda_gemm_benchmark ..."
    echo "error=ncu not found in PATH"
  } | tee "$LIMIT"
  python3 scripts/update_project_state.py --profiling LIMITED
  exit 0
fi

if [[ ! -x build/cuda_gemm_benchmark ]]; then
  "$PWD/scripts/build.sh"
fi

set +e
ncu --set basic --launch-count 1 --export "$OUT" \
  ./build/cuda_gemm_benchmark --M 2048 --N 2048 --K 2048 --kernel register \
  --warmup 10 --iterations 50 --no-verify --experiment-name "profile_register_${STAMP}"
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tool=nsight_compute"
    echo "status=LIMITED_OR_FAILED"
    echo "command=ncu --set basic --launch-count 1 --export $OUT ./build/cuda_gemm_benchmark --M 2048 --N 2048 --K 2048 --kernel register --warmup 10 --iterations 50 --no-verify"
    echo "exit_code=$rc"
  } | tee "$LIMIT"
  python3 scripts/update_project_state.py --profiling LIMITED
  exit 0
fi
python3 scripts/update_project_state.py --profiling PASS
printf 'Nsight Compute profiling completed: %s\n' "$OUT"
