# Final Report

## Status

This report is a template until the repository is executed on the target RTX 4060 Laptop GPU. No numerical performance claim is included without a corresponding stored experiment record.

## 1. What was implemented?

CPU reference; naive CUDA; coalesced CUDA; shared-memory 16×16 and 32×32 tiling; register-tiled 32×32 and 64×64 output tiles; float4 vectorization; warp-shuffle experimental kernel; WMMA Tensor Core FP16-input/FP32-accumulate path; cuBLAS FP32 and FP16 baselines; append-only experiment recording; correctness and profiling automation.

## 2. Baseline

Populate from the first official baseline experiment recorded after a clean target build.

## 3. Optimization experiments

Populate from `docs/optimization_log.md` and the experiment IDs in `results/`.

## 4. Winners and regressions

Accept only measured improvements. Record rejected variants and the evidence for rejection.

## 5. cuBLAS comparison

Use matched dimensions/dtype and the stored median kernel timing. Do not compare incompatible timing definitions.

## 6. Tensor Core comparison

Compare the WMMA implementation against the FP16 cuBLAS Tensor Core-enabled path for the same supported workload. Profiler-based Tensor Core utilization statements require Nsight Compute evidence.

## 7. Correctness

Report workload coverage, tolerances, max absolute error, max relative error, and any unsupported cases from the experiment history.

## 8. Profiling

Report actual Nsight Compute/System results, exact commands, and any WSL2/driver limitation artifacts under `profiles/limitations/`.

## 9. Remaining work

The target-side execution, controlled optimization experiments, profiler observations, measured plots, and final conclusions must be filled from actual runs. Anything not observed remains UNKNOWN rather than being inferred as fact.
