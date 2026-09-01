#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"$PWD/scripts/build.sh"

BIN="$PWD/build/cuda_gemm_benchmark"
"$BIN" --M 128 --N 128 --K 128 --kernel all --warmup 3 --iterations 10 --verify --experiment-name correctness_baseline_128
"$BIN" --M 127 --N 131 --K 125 --kernel all --warmup 3 --iterations 10 --verify --experiment-name correctness_odd_127x131x125
"$BIN" --M 129 --N 257 --K 193 --kernel all --warmup 2 --iterations 6 --verify --experiment-name correctness_rectangular_129x257x193
"$BIN" --M 128 --N 128 --K 128 --kernel tensorcore --dtype fp16 --warmup 3 --iterations 10 --verify --experiment-name correctness_tensorcore_128
"$BIN" --M 128 --N 128 --K 128 --kernel cublas --dtype fp16 --warmup 3 --iterations 10 --verify --experiment-name correctness_cublas_fp16_128
"$BIN" --M 128 --N 131 --K 125 --kernel cpu --warmup 1 --iterations 3 --verify --experiment-name correctness_cpu_128x131x125

python3 scripts/update_project_state.py --correctness PASS
printf 'Correctness PASS: all runnable requested kernels returned verified results.\n'
