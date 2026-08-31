#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"$PWD/scripts/build.sh"
BIN="$PWD/build/cuda_gemm_benchmark"

# Performance runs intentionally disable CPU verification; the result is recorded as NOT_VERIFIED.
for s in 128 1024 2048 4096; do
  "$BIN" --M "$s" --N "$s" --K "$s" --kernel all --warmup 10 --iterations 50 --no-verify --experiment-name "square_${s}"
done

while read -r M N K NAME; do
  "$BIN" --M "$M" --N "$N" --K "$K" --kernel all --warmup 10 --iterations 50 --no-verify --experiment-name "$NAME"
done <<'EOF'
4096 1024 2048 rectangular_4096x1024x2048
1024 4096 2048 rectangular_1024x4096x2048
2048 4096 1024 rectangular_2048x4096x1024
127 131 125 odd_127x131x125
EOF

"$BIN" --M 2048 --N 2048 --K 2048 --kernel tensorcore --dtype fp16 --warmup 10 --iterations 50 --no-verify --experiment-name tensorcore_fp16_2048
"$BIN" --M 2048 --N 2048 --K 2048 --kernel cublas --dtype fp16 --warmup 10 --iterations 50 --no-verify --experiment-name cublas_fp16_2048

python3 scripts/validate_results.py
python3 scripts/update_project_state.py --benchmark PASS
python3 scripts/generate_plots.py
printf 'Benchmark suite PASS. Every result was appended; historical evidence was preserved.\n'
