# Release Readiness

The release is considered complete when the implementation, benchmark evidence, profiler artifacts, and documentation describe the same measured state.

## Verification

- [x] Clean build succeeds for CUDA 12.0.140 / SM 8.9.
- [x] Correctness suite passes on the target GPU.
- [x] Benchmark baselines and optimization runs are recorded in append-only history.
- [x] `NOT_VERIFIED` is distinct from numerical `FAIL`.
- [x] CUDA events are used for GPU kernel timing.
- [x] GFLOPS is derived from the reported median kernel latency.
- [x] Tensor Core correctness is checked for supported dimensions.
- [x] cuBLAS baselines use persistent handles.
- [x] Nsight Compute artifacts are retained for selected kernels and treated as diagnostic timing evidence only.
- [x] Optimization experiments retain their lineage and measured outcomes.
- [x] Reports and plots are generated from stored measurements.
- [x] Documentation reflects the finalized implementation and dataset.

## Release snapshot

The current release contains 589 recorded experiments, 589 raw experiment artifacts, and 15 profiling-index records. The latest recorded experiment is `EXP-000589`.
