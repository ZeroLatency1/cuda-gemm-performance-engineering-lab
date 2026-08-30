# Benchmark Methodology

## Timing

GPU kernel timing uses CUDA events. Warmups run before timed iterations, and each timed iteration records one CUDA-event interval around the kernel launch. Reported kernel statistics are median, p95, and minimum latency.

Transfer measurements are independent from kernel timing:

- **H2D:** host-to-device input copies only
- **D2H:** device-to-host output copy only
- **End-to-end:** H2D + kernel + D2H in one measured interval

Transfer timings are never substituted for kernel latency in GFLOPS.

## Throughput

For GEMM:

`FLOPs = 2 × M × N × K`

`GFLOPS = FLOPs / (median_kernel_time_ms × 1e6)`

The same median latency reported in the result record is used in the throughput calculation.

## Correctness

The CPU reference uses the project’s row-major GEMM implementation. The error metric is:

`abs_error = |got - reference|`

`relative_error = abs_error / max(|reference|, 1e-3)`

A result passes when every element satisfies `abs_error <= 1e-3 OR relative_error <= 1e-2`, and no NaN/Inf is observed. The denominator floor prevents near-zero reference values from dominating the metric.

For FP16 Tensor Core/cuBLAS tests, inputs are quantized to FP16 first and the CPU reference accumulates the quantized operands in FP32.

## Workload control

Comparisons use the same input seed, dimensions, warmups, iteration count, dtype, and execution environment unless an experiment explicitly records a changed condition. Important experiments carry `parent_experiment_id`, `baseline_experiment_id`, and `optimization_description` metadata.

## Result preservation

Every attempted configuration gets a permanent `EXP-XXXXXX` identifier. Records are appended to both `results/experiments.csv` and `results/experiments.jsonl`; a unique raw JSON artifact is also stored. The CSV schema is checked before append so incompatible history is not silently mixed.

`NOT_VERIFIED` means performance was measured with verification disabled. It is not the same as `FAIL`.
