# CUDA GEMM Performance Lab

A CUDA performance-engineering study of dense matrix multiplication on an NVIDIA GeForce RTX 4060 Laptop GPU (SM 8.9). The repository contains multiple GEMM implementations, reproducible benchmark history, correctness checks, and Nsight Compute profiles used to connect measured throughput with GPU execution behavior.

[![CUDA](https://img.shields.io/badge/CUDA-12.0-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Compute Capability](https://img.shields.io/badge/SM-8.9-76B900)](#hardware)
[![Experiments](https://img.shields.io/badge/experiments-589-76B900)](#experiment-records)
[![Status](https://img.shields.io/badge/release-v1.0--final-blue)](#release)

## At a glance

| Item | Result |
|---|---|
| Target GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Architecture | Ada Lovelace, SM 8.9 |
| CUDA | 12.0.140 |
| Compiler | GCC 13.3.0 |
| OS | Ubuntu 24.04 under WSL2 |
| Recorded experiments | **589** |
| Raw experiment artifacts | **589** |
| Profiling index records | **15** |
| Nsight Compute reports in full release | **16** |
| Best custom FP32 result | **2.81 TFLOPS** at 1024³ |
| Best custom FP32 speedup | **3.91×** vs. naive at 1024³ |
| cuBLAS FP32 reference | **8.46 TFLOPS** at 1024³ |
| cuBLAS FP16 reference | **28.84 TFLOPS** at 2048³ |

## Performance snapshot

The standardized 1024³ FP32 workload provides the clearest apples-to-apples comparison among the custom kernels.

| Kernel | Median GFLOPS | vs. naive |
|---|---:|---:|
| Naive | 716.92 | 1.00× |
| Coalesced | 967.83 | **1.35×** |
| Shared16 | 857.56 | **1.20×** |
| Shared32 | 836.07 | **1.17×** |
| Register | 1,614.97 | **2.25×** |
| Register64 | **2,806.70** | **3.91×** |
| Vectorized | 2,671.83 | **3.73×** |
| Warp | 103.46 | **0.14×** |
| cuBLAS | **8,460.93** | **11.80×** |

The measurements show that more sophisticated code is not automatically faster. Register blocking produced the largest repeatable custom-kernel gains, while the warp-oriented implementation regressed substantially on the same workload.

## What is implemented

- CPU reference GEMM for numerical validation
- Naive CUDA GEMM
- Coalesced global-memory access
- Shared-memory tiling with 16×16 and 32×32 tiles
- Register tiling with 32×32 and 64×64 output tiles
- `float4` vectorized memory access
- Warp-oriented experimental GEMM
- WMMA FP16-input / FP32-accumulate Tensor Core GEMM
- cuBLAS FP32 and FP16 reference paths
- Append-only experiment history and immutable raw results
- Automated correctness, benchmark, profiling, environment, and static validation scripts

## Why the register64 kernel wins

The 64×64 register-tiled kernel trades resident warps for greater per-thread reuse.

| Metric | Register32 | Register64 |
|---|---:|---:|
| Registers / thread | 39 | 56 |
| Theoretical occupancy | 100% | 66.67% |
| Achieved occupancy (representative profile) | 98.74% | 64.75% |
| 1024³ median | 1.615 TFLOPS | **2.807 TFLOPS** |

The result is a concrete example of why occupancy is a diagnostic constraint rather than a standalone optimization target: the lower-occupancy kernel is substantially faster because additional register reuse reduces repeated work and improves arithmetic efficiency for the tested GEMM.

## Tensor Core validation

The custom WMMA path was checked with Nsight Compute rather than inferred from source code alone. A focused profile recorded:

- **4,194,304 HMMA Tensor Core instructions**
- **67,108,864 HMMA-pipe active cycles**
- FP16→FP32 tensor-path activity

At 2048³ FP16, the custom WMMA kernel reached a median of **5.33 TFLOPS**, while cuBLAS reached **28.84 TFLOPS**. The gap is retained as an engineering result: the implementation exercises the intended Tensor Core hardware but leaves substantial performance on the table relative to a production vendor library.

## Correctness

Numerical validation uses a CPU reference with an absolute/relative error check and explicit NaN/Inf rejection. Performance-only runs are marked `NOT_VERIFIED` and are not treated as correctness failures.

The history includes square, rectangular, odd-sized, FP32, and FP16 workloads. Unsupported configurations are recorded explicitly rather than silently omitted.

## Benchmark methodology

The standard performance suite uses **10 warmup iterations and 50 timed iterations**. GPU kernel timing uses CUDA events, and GFLOPS is calculated from the same median kernel latency reported in the experiment record:

```text
FLOPs   = 2 × M × N × K
GFLOPS  = FLOPs / (median_kernel_time_ms × 1e6)
```

H2D, D2H, and end-to-end measurements are recorded independently. End-to-end timing is a separate interval encompassing transfer and execution, so it is not expected to equal the arithmetic sum of the independently sampled component medians.

Nsight Compute timing is diagnostic only because instrumentation perturbs execution.

## Experiment records

Every execution receives a unique experiment ID. The authoritative history is stored in:

```text
results/experiments.csv
results/experiments.jsonl
results/raw/EXP-XXXXXX.json
```

The current release contains **589 experiments and 589 raw artifacts**, with `EXP-000589` as the latest experiment. The raw-artifact set is complete from `EXP-000001` through `EXP-000589`.

For example:

```bash
python3 scripts/validate_results.py
```

returns:

```text
Result validation PASS: 589 experiments, 589 raw artifacts
```

## Profiling

Selected kernels have full Nsight Compute reports under `profiles/` alongside machine-readable metadata and the profiling index in `results/profiling.jsonl`.

The complete release retains the actual `.ncu-rep` files so profile conclusions can be independently inspected in Nsight Compute.

Useful command:

```bash
./scripts/run_profile.sh \
  --kernel register64 \
  --M 1024 --N 1024 --K 1024 \
  --dtype fp32
```

## Reproduce the benchmark suite

```bash
./scripts/build.sh
./scripts/test_correctness.sh
./scripts/run_benchmarks.sh
./scripts/test_static.sh
python3 scripts/validate_results.py
```

The benchmark suite records new experiment IDs; it does not overwrite historical raw artifacts.

## Repository layout

```text
src/        CUDA, cuBLAS, CPU, benchmark and correctness implementations
include/    public interfaces
scripts/    build, test, benchmark, profiling and analysis automation
results/    experiment history, raw artifacts and summaries
profiles/   Nsight Compute artifacts and profile metadata
docs/       methodology, architecture, optimization analysis and final report
```

## Engineering conclusions

<details>
<summary><strong>Memory access</strong></summary>

Coalesced access improves the 1024³ FP32 result by about 35%, but the same strategy does not remain uniformly dominant on larger shapes. Global-memory efficiency is therefore useful but not sufficient to predict the best GEMM implementation.
</details>

<details>
<summary><strong>Shared memory</strong></summary>

Shared-memory tiling improves the 1024³ baseline by roughly 17–20% in this implementation, while larger workloads show smaller or negative relative gains. The result is consistent with a workload-dependent balance among memory reuse, instruction overhead, and other execution resources.
</details>

<details>
<summary><strong>Register reuse</strong></summary>

Register blocking is the strongest custom FP32 strategy in the tested large workloads. The 64×64 variant reaches about 3.9× the naive throughput at 1024³ despite lower theoretical occupancy.
</details>

<details>
<summary><strong>Vectorization</strong></summary>

Vectorized memory access is another strong custom strategy, reaching about 3.7× the naive throughput at 1024³. Its relative advantage changes with matrix shape.
</details>

<details>
<summary><strong>Negative result</strong></summary>

The warp-oriented variant is retained as evidence of experimental discipline: on the 1024³ workload it reaches only about 14% of naive throughput. The implementation was measured, diagnosed, and rejected rather than presented as an optimization success.
</details>

## Release

The frozen release snapshot contains the source code, benchmark history, raw experiment artifacts, profiling metadata, and full Nsight Compute reports. Documentation describes the measured state of the release rather than a future work plan.

Latest experiment: **EXP-000589**

Best custom FP32 kernel in the tested release: **register64**
