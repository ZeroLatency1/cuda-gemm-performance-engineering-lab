# Final Report — CUDA GEMM Performance Engineering Lab

## Executive summary

This project is an evidence-driven CUDA GEMM performance laboratory targeting an **NVIDIA GeForce RTX 4060 Laptop GPU (SM 8.9)** under Ubuntu 24.04 / WSL2 with CUDA 12.0 and GCC 13.3.

The finalized archive contains **589 recorded experiments**, **589 immutable raw experiment artifacts**, and **15 profiling-index records**. The benchmark history evaluates multiple GEMM strategies across square, rectangular, and odd dimensions, FP32 and FP16 precision, and both custom kernels and cuBLAS.

The strongest custom FP32 result is the **64x64 register-tiled (`register64`) kernel**, reaching **2.81 TFLOPS median at 1024^3**, or approximately **3.91x the naive kernel**. The same strategy remains the strongest custom FP32 kernel at 2048^3 and 4096^3.

The vendor cuBLAS reference remains substantially faster, reaching **8.46 TFLOPS median at 1024^3 FP32** and **28.84 TFLOPS median at 2048^3 FP16**.

The custom WMMA Tensor Core path is also hardware-validated: a focused Nsight Compute run recorded **4,194,304 HMMA instructions** and **67,108,864 HMMA-pipe active cycles**, establishing that the implementation genuinely executes the Tensor Core path.

## 1. What was implemented

The implementation includes:

- CPU reference GEMM
- naive CUDA GEMM
- coalesced global-memory GEMM
- shared-memory 16x16 and 32x32 tiling
- register-tiled 32x32 and 64x64 output tiles
- float4 vectorized memory access
- warp-oriented experimental GEMM
- WMMA Tensor Core FP16-input / FP32-accumulate GEMM
- cuBLAS FP32 and FP16 reference paths
- append-only experiment recording
- correctness and benchmark automation
- Nsight Compute profiling and profile lineage metadata

## 2. Experimental methodology

GPU kernel timing uses CUDA events with standardized benchmark runs using **10 warmups and 50 timed iterations**. The reported throughput is based on the recorded median kernel latency:

`FLOPs = 2 x M x N x K`

`GFLOPS = FLOPs / (median_kernel_time_ms x 1e6)`

Transfer timings are recorded independently as H2D, D2H, and end-to-end measurements. End-to-end is a separate timed interval encompassing transfer and execution phases; it is not expected to equal the arithmetic sum of the independently sampled component medians.

Performance-only runs use `NOT_VERIFIED`; this is explicitly different from numerical failure. Correctness runs use the project's declared rule of **absolute error <= 1e-3 OR relative error <= 1e-2**, with a 1e-3 denominator floor and NaN/Inf rejection.

Nsight Compute instrumentation materially perturbs execution and therefore its timings are diagnostic only. The project records this explicitly with `performance_claim_eligible=false`.

## 3. Optimization results

### 1024^3 FP32

| Strategy | Median GFLOPS | vs naive |
|---|---:|---:|
| naive | 716.92 | 1.00x |
| coalesced | 967.83 | 1.35x |
| shared16 | 857.56 | 1.20x |
| shared32 | 836.07 | 1.17x |
| register | 1614.97 | 2.25x |
| register64 | **2806.70** | **3.91x** |
| vectorized | 2671.83 | 3.73x |
| warp | 103.46 | 0.14x |
| cuBLAS | **8460.93** | **11.80x** |

The strongest custom result is therefore register64. The warp-level experiment is a useful negative result: it was measured and rejected rather than treated as an optimization success by assumption.

### Larger FP32 workloads

The same ranking is broadly preserved:

- **2048^3:** register64 2829.48 GFLOPS vs naive 927.54 GFLOPS (**3.05x**)
- **4096^3:** register64 2466.25 GFLOPS vs naive 773.83 GFLOPS (**3.19x**)

Vectorization remains a strong alternative, but the relative benefit varies with shape. Coalescing and shared-memory tiling are clearly workload-sensitive rather than universally dominant.

## 4. Why register64 wins

Nsight Compute provides direct evidence of the resource trade-off:

| Kernel | Registers/thread | Theoretical occupancy | Achieved occupancy |
|---|---:|---:|---:|
| register | 39 | 100% | 98.74% |
| register64 | 56 | 66.67% | 64.75% |

The 64x64 register tile therefore uses more register state and supports fewer resident warps, yet the clean benchmark is substantially faster. The defensible interpretation is that greater per-thread reuse and arithmetic intensity outweigh the lost occupancy for these GEMM workloads.

This is a stronger performance-engineering conclusion than simply “higher occupancy is better.”

## 5. Memory and cache behavior

The Nsight profiles show that high reported `Memory Throughput` should not be interpreted as equivalent to high external DRAM bandwidth utilization. For the representative custom kernels, DRAM throughput was only a few percent in the extracted profiles, while L1/L2 activity was substantial.

Accordingly, the report does **not** claim that the custom kernels are simply DRAM-bandwidth-bound. The measured data is more consistent with substantial on-chip reuse and execution-resource pressure.

## 6. Tensor Core implementation

The 2048^3 FP16 custom WMMA implementation achieves a median of **5.33 TFLOPS** across six standardized runs, compared with **28.84 TFLOPS** for cuBLAS FP16.

The custom Tensor Core path is nonetheless directly validated at the hardware level. The focused Nsight Compute profile contains:

- **4,194,304 HMMA tensor-pipe instructions**
- **67,108,864 HMMA-pipe active cycles**
- **17,179,869,184 FP16-to-FP32 tensor-path operations**

The correct conclusion is therefore: **the WMMA implementation genuinely executes Tensor Core HMMA operations but remains substantially below cuBLAS performance.**

## 7. Correctness

The archive contains **111 numerically verified PASS records**. Across those verified records, the maximum observed error is approximately:

- maximum absolute error: **1.72e-4**
- maximum relative error: **3.76%**

Because the project's acceptance criterion is an OR between absolute and relative error thresholds, the latter does not imply a failed result. The precise claim is that the verified runs passed the project's declared numerical tolerance.

The odd-shape suite also records unsupported vectorized cases explicitly when `N` is not divisible by four.

## 8. Repeatability

The standardized six-run groups provide useful variance estimates. Examples at 1024^3 FP32 include:

- shared32: **0.62% CV**
- shared16: **0.89% CV**
- register64: **1.61% CV**
- vectorized: **2.44% CV**
- register: **4.00% CV**
- cuBLAS: **4.79% CV**

Small 128^3 and odd-size workloads have much higher variance and should therefore be treated primarily as functionality/shape-coverage evidence.

## 9. Theory vs measured performance

NVIDIA lists the RTX 4060 Laptop GPU with **3,072 CUDA cores** and a **1,470–2,370 MHz** boost-clock range. The laptop GPU's actual operating point depends on the system's power and thermal configuration. citeturn643766search0

Using the top listed boost value as a simple scalar-FP32 reference gives approximately **14.56 TFLOPS**. The project's best verified FP32 result is **8.01 TFLOPS** at 1024^3 with cuBLAS, or roughly **55% of that nominal boost-clock reference**. This should be read as a contextual reference, not a claim that every workload can sustain the published maximum clock.

The best custom 1024^3 FP32 result, register64 at **2.81 TFLOPS**, is roughly **19% of that nominal scalar reference**. That gap is expected for a small educational custom GEMM compared with a highly optimized production library and is itself an important benchmark conclusion.

## 10. Complete data access

The project preserves the full history in machine- and spreadsheet-friendly forms:

- `results/experiments.csv`
- `results/experiments.jsonl`
- `results/raw/EXP-*.json`
- `results/profiling.jsonl`
- `profiles/EXP-XXXXXX/`

Example run lookup:

```bash
# Search all recorded experiments for one workload/kernel
python3 - <<'PY'
import csv
with open('results/experiments.csv', newline='') as f:
    for r in csv.DictReader(f):
        if r['kernel'] == 'register64' and r['M'] == '1024' and r['N'] == '1024' and r['K'] == '1024':
            print(r['experiment_id'], r['median_ms'], r['gflops'], r['verification_status'])
PY
```

## 11. Final engineering conclusions

1. **Register-level tiling produced the largest repeatable custom FP32 gains.**
2. **Register64 is the strongest custom kernel in the tested large FP32 workloads.**
3. **Occupancy must be treated as a constraint/diagnostic, not as the optimization objective by itself.**
4. **Coalescing and shared-memory tiling produce shape-dependent gains and regressions.**
5. **The warp-oriented variant was experimentally rejected because it regressed severely.**
6. **The custom WMMA path genuinely uses Tensor Core HMMA instructions but has substantial headroom relative to cuBLAS.**
7. **The complete 589-run append-only history makes the conclusions reproducible and auditable rather than dependent on one benchmark run.**

## 12. Release status

The original benchmark/result artifacts are intentionally preserved. This report update changes documentation only; the experiment history, raw result records, and profiling index are not rewritten as part of the documentation refresh. The finalized archive contains **589 experiments**, with **EXP-000589** as the latest recorded experiment.
