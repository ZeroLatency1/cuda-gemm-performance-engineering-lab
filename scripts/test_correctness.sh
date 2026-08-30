#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"$PWD/scripts/build.sh"

"$PWD/build/cuda_gemm_benchmark" --M 128 --N 128 --K 128 --kernel all --warmup 2 --iterations 5 --verify --experiment-name correctness_128
"$PWD/build/cuda_gemm_benchmark" --M 127 --N 131 --K 125 --kernel all --warmup 2 --iterations 5 --verify --experiment-name correctness_odd_127x131x125
"$PWD/build/cuda_gemm_benchmark" --M 1021 --N 1023 --K 1025 --kernel all --warmup 1 --iterations 2 --verify --experiment-name correctness_1021x1023x1025
"$PWD/build/cuda_gemm_benchmark" --M 128 --N 128 --K 128 --kernel tensorcore --dtype fp16 --warmup 2 --iterations 5 --verify --experiment-name correctness_tensorcore
"$PWD/build/cuda_gemm_benchmark" --M 128 --N 131 --K 125 --kernel cpu --warmup 1 --iterations 2 --verify --experiment-name correctness_cpu

printf 'Correctness suite passed.\n'
