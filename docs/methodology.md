# Benchmark Methodology

Kernel timing uses CUDA events. GPU kernel timing is separated from host-device transfer timing.

Reported statistics:

- median
- p95
- minimum

For GEMM:

`FLOPs = 2 * M * N * K`

GFLOPS uses the reported median kernel latency.

Correctness compares against the CPU reference using max absolute error and a relative error denominator floor to avoid instability near zero.

Unverified performance runs are recorded as `NOT_VERIFIED`, not `FAIL`.

## Result preservation

The benchmark logger is append-only. Each successful kernel benchmark receives a unique experiment ID and is written to both `results/experiments.csv` and `results/experiments.jsonl`. A per-experiment JSON record is also written under `results/raw/`. The result directory is resolved from the executable location under the normal CMake `<project>/build` layout, so launching the executable from either the project root or `build/` does not create a second results tree. `KUCH_RESULTS_DIR` can be used to override the output directory explicitly.
