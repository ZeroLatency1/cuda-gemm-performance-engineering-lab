#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"$PWD/scripts/build.sh"

for s in 1024 2048 4096; do
  "$PWD/build/cuda_gemm_benchmark" --M "$s" --N "$s" --K "$s" --kernel all --warmup 10 --iterations 50 --no-verify --experiment-name "square_${s}"
done

for dims in \
  "4096 1024 2048 rectangular_4096x1024x2048" \
  "1024 4096 2048 rectangular_1024x4096x2048" \
  "2048 4096 1024 rectangular_2048x4096x1024" \
  "127 131 125 odd_127x131x125"; do
  read -r M N K NAME <<< "$dims"
  "$PWD/build/cuda_gemm_benchmark" --M "$M" --N "$N" --K "$K" --kernel all --warmup 10 --iterations 50 --no-verify --experiment-name "$NAME"
done

"$PWD/build/cuda_gemm_benchmark" --M 2048 --N 2048 --K 2048 --kernel tensorcore --dtype fp16 --warmup 10 --iterations 50 --no-verify --experiment-name "tensorcore_2048"
printf 'Benchmark suite complete. Results were appended; existing experiment history was preserved.\n'
