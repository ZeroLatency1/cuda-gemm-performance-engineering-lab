# CUDA GEMM Performance Engineering Lab

A reproducible CUDA GEMM portfolio focused on evidence-backed optimization on the **NVIDIA GeForce RTX 4060 Laptop GPU / SM 8.9** using the requested **Ubuntu 24.04 under WSL2, CUDA 12.0, GCC 13.x** target environment.

## What is implemented

CPU reference, naive CUDA, coalesced CUDA, shared-memory tiling (16/32), register tiling (32/64 output tiles), float4 vectorization, warp-shuffle experimental GEMM, WMMA Tensor Core FP16-input/FP32-accumulate GEMM, and cuBLAS FP32/FP16 baselines are included.

The Tensor Core kernel uses a one-warp-per-16x16-output-tile WMMA mapping and explicitly requires dimensions divisible by 16. The FP16 cuBLAS path uses `cublasGemmEx` with Tensor Core math enabled for an apples-to-apples library comparison.

## Evidence policy

The project never treats source existence as proof of completion. The lifecycle is:

`IMPLEMENTED → BUILT → CORRECTNESS VERIFIED → BENCHMARKED → PROFILED WHERE POSSIBLE → DOCUMENTED`

No benchmark or profiler numbers are fabricated. Unknown values remain unknown until observed on the target machine.

## Build

```bash
./scripts/build.sh
```

The CMake build targets **SM 8.9**. The build script captures an environment snapshot before compiling and updates `project_state.json` only after a successful build.

## Correctness

```bash
./scripts/test_correctness.sh
```

The suite covers square, odd, rectangular, Tensor Core-supported, and FP16 cuBLAS cases. Verification uses a CPU reference and records both maximum absolute error and maximum relative error with a documented denominator floor.

## Benchmarking

A single run can be controlled with:

```bash
./build/cuda_gemm_benchmark \
  --M 1024 --N 1024 --K 1024 \
  --kernel register --dtype fp32 \
  --warmup 10 --iterations 50 --verify \
  --experiment-name register_1024
```

Useful flags include `--M`, `--N`, `--K`, `--kernel`, `--dtype`, `--warmup`, `--iterations`, `--verify`, `--no-verify`, `--seed`, `--experiment-name`, `--output`, `--parent-experiment-id`, `--baseline-experiment-id`, and `--optimization-description`.

GPU kernel timing uses CUDA events. Kernel-only timing is kept separate from H2D, D2H, and end-to-end timing. GFLOPS is always computed from the same **median kernel latency** that is reported.

Run the controlled matrix with:

```bash
./scripts/run_benchmarks.sh
```

## Result preservation

The official experiment history lives in:

- `results/experiments.csv`
- `results/experiments.jsonl`
- `results/raw/EXP-XXXXXX.json`

Every attempted configuration receives a unique experiment ID. Successful measured runs are `PASS`; unsupported configurations are `UNSUPPORTED`; execution failures are `ERROR`; performance-only runs explicitly use `verification_status=NOT_VERIFIED`. Existing history is never truncated or rewritten during normal benchmark operation. The CSV header is schema-checked before append.

Historical performance from previous projects is not imported into the official dataset. The official record starts with the first clean run of this repository on the target hardware.

## Profiling

```bash
./scripts/run_profile.sh
```

Nsight Compute output is written under a unique timestamped path in `profiles/`. If WSL2/driver/tool permissions prevent attachment, the script records the exact limitation instead of inventing metrics or failing as though profiling had succeeded.

## Static repository checks

```bash
./scripts/test_static.sh
```

These checks validate the experiment schema, CSV/JSONL alignment, SM 8.9 targeting, duplicate-result-type protection, Python syntax, and shell syntax. They do not substitute for target GPU correctness or performance execution.

## Project state

`project_state.json` tracks implementation/build/correctness/benchmark/profiling/documentation status, latest experiment ID, current best verified kernel, and known limitations. `results/summaries/` stores timestamped environment snapshots.

## Repository map

- `src/` — CUDA, cuBLAS, CPU, benchmark and correctness implementations
- `include/` — public interfaces
- `scripts/` — build/test/benchmark/profile/environment/plot automation
- `results/` — append-only experiment history and generated plots
- `profiles/` — profiler artifacts and documented limitations
- `docs/` — methodology, optimization log, profiling notes and final report

## Current completion state

This archive contains the engineering implementation and verification infrastructure, but **target RTX 4060 execution is not claimed here** because this development environment has no `nvcc` and no NVIDIA GPU. The repository therefore starts with target-side status pending and contains no fabricated performance numbers.
