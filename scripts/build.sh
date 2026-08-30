#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"$PWD/scripts/collect_environment.sh"
command -v nvcc >/dev/null 2>&1 || { echo 'ERROR: nvcc is required; CUDA was not found.' >&2; exit 2; }
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
python3 scripts/update_project_state.py --build PASS
printf 'Build PASS: target=SM 8.9, build=build/cuda_gemm_benchmark\n'
