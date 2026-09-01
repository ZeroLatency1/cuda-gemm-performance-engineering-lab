# Benchmark Analysis

## Executive result

This archive contains the executed RTX 4060 Laptop GPU benchmark history for the CUDA GEMM Performance Engineering Lab. The current dataset contains **589 experiments** and **589 immutable raw artifacts**. Of these, **579 completed with `PASS`**, **10 were explicit `UNSUPPORTED` cases**, **111 were numerically verified**, **468 were performance-only `NOT_VERIFIED` runs**, and the 10 unsupported records are marked `NOT_APPLICABLE` for verification.

All performance claims below use the project's ordinary benchmark records, not Nsight-instrumented timing. The standardized performance suite uses **10 warmups and 50 timed iterations**. `GFLOPS` is computed from `2*M*N*K` divided by the recorded median kernel latency. H2D, D2H, and end-to-end timings are independently sampled measurements; end-to-end should not be expected to equal the arithmetic sum of the reported component medians.

## Standardized 1024^3 FP32 comparison

Six repeated runs are available for each kernel under the standardized workload. Median throughput is the primary comparison metric.

| Kernel | Median GFLOPS | Speedup vs naive |
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

This is the clearest controlled optimization progression in the dataset. The strongest custom implementation is `register64`, while cuBLAS remains the vendor-optimized reference and is substantially faster.

## Shape sensitivity

Optimization effects are workload-dependent rather than universally monotonic.

### 2048^3 FP32

| Kernel | Median GFLOPS | vs naive |
|---|---:|---:|
| naive | 927.54 | 1.00x |
| coalesced | 912.31 | 0.98x |
| shared16 | 842.14 | 0.91x |
| shared32 | 824.43 | 0.89x |
| register | 1585.52 | 1.71x |
| register64 | **2829.48** | **3.05x** |
| vectorized | 2204.54 | 2.38x |
| warp | 70.62 | 0.08x |
| cuBLAS | **7682.04** | **8.28x** |

### 4096^3 FP32

| Kernel | Median GFLOPS | vs naive |
|---|---:|---:|
| naive | 773.83 | 1.00x |
| coalesced | 762.00 | 0.98x |
| shared16 | 758.89 | 0.98x |
| shared32 | 776.44 | 1.00x |
| register | 1401.98 | 1.81x |
| register64 | **2466.25** | **3.19x** |
| vectorized | 1961.77 | 2.54x |
| warp | 49.53 | 0.06x |
| cuBLAS | **6174.38** | **7.98x** |

The data therefore supports a stronger engineering conclusion than “every optimization helped”: **register tiling is consistently valuable; larger register tiling is the strongest custom optimization in the measured FP32 suite; shared-memory/coalescing wins are shape-sensitive; the warp-oriented experiment was rejected.**

## Rectangular workloads

The same pattern persists for non-square GEMMs, with workload shape affecting the relative value of memory and register optimizations. For example, the 1024x4096x2048 median results are approximately **835 GFLOPS naive, 1512 GFLOPS register, 2651 GFLOPS register64, 2139 GFLOPS vectorized, and 7165 GFLOPS cuBLAS**. The 2048x4096x1024 and 4096x1024x2048 matrices show the same broad ranking while exposing modest changes in absolute throughput.

These records matter because the project is not optimized against only one square matrix.

## Small and odd workloads

The 128^3 and 127x131x125 groups have substantially higher run-to-run variance. Fixed launch and scheduling overhead becomes a large fraction of the total work for these small matrices, so they should be used primarily as **functionality/shape-coverage evidence**, not as the strongest evidence for peak throughput.

The odd-shape workload also demonstrates explicit kernel applicability boundaries. The float4 vectorized implementation is recorded as `UNSUPPORTED` when `N` is not divisible by 4 rather than silently producing an invalid result.

## FP16 and Tensor Core results

At 2048^3 FP16, six standardized runs give:

| Kernel | Median GFLOPS | Mean GFLOPS | CV |
|---|---:|---:|---:|
| custom WMMA Tensor Core | 5332.02 | 5598.57 | 9.74% |
| cuBLAS | **28840.64** | 28780.51 | **1.34%** |

The custom WMMA implementation is therefore a real Tensor Core path, but it remains substantially behind the optimized cuBLAS implementation. This should be presented as a measured gap and optimization opportunity, not as a failure of the Tensor Core concept itself.

## Repeatability

For the major 1024^3 and larger FP32 workloads, many standardized groups have coefficients of variation in the low single digits. Examples include 1024^3 `shared32` at **0.62%**, `shared16` at **0.89%**, `register64` at **1.61%**, and `vectorized` at **2.44%**. Higher-variance cases exist, particularly for very small matrices and some cuBLAS/FP16 groups, so final conclusions should use medians and report variance where relevant.

## Correctness interpretation

The project's correctness rule is **absolute error <= 1e-3 OR relative error <= 1e-2**, with a `1e-3` denominator floor and NaN/Inf rejection. Across the recorded history, **111 records passed this declared rule**. The maximum recorded verified error is approximately **1.72e-4 absolute** and **3.76% relative**. Because the acceptance rule is an OR, a small absolute error can legitimately pass even when the relative error is above 1%.

Therefore the accurate statement is: **the verified runs passed the project's declared numerical tolerance**. It is not accurate to claim that every verified run stayed below 1% relative error.

## Theory vs measured performance

The target is the RTX 4060 Laptop GPU, for which NVIDIA lists **3,072 CUDA cores** and a **1,470–2,370 MHz** boost-clock range; the exact laptop power/clock configuration varies by system. citeturn643766search0

A simple scalar-FP32 upper-bound reference using 3,072 CUDA cores and the top listed 2.37 GHz boost value is approximately **14.56 TFLOPS**. The best verified FP32 benchmark in the project is the cuBLAS 1024^3 result at about **8.01 TFLOPS** in experiment `EXP-000361`, which is about **55% of that nominal boost-clock reference**. This comparison is contextual rather than a claim of sustained hardware peak because actual observed clocks are workload- and power-dependent.

For custom FP32 kernels, the best standardized 1024^3 result is register64 at **2.81 TFLOPS**, about **19% of the nominal 14.56 TFLOPS scalar reference**. The correct engineering interpretation is that the project makes substantial progress through tiling and reuse but remains below an optimized vendor GEMM implementation.

## Easy access to all runs

The complete run-by-run evidence remains in:

- `results/experiments.csv` — spreadsheet-friendly history
- `results/experiments.jsonl` — machine-readable append-only history
- `results/raw/EXP-XXXXXX.json` — immutable per-experiment records
- `results/profiling.jsonl` — profile index
- `profiles/EXP-XXXXXX/` — profile metadata and, when retained, corresponding Nsight report artifacts

Useful queries:

```bash
# All runs for one workload/kernel
python3 - <<'PY'
import csv
with open('results/experiments.csv', newline='') as f:
    for r in csv.DictReader(f):
        if (r['dtype'], r['M'], r['N'], r['K'], r['kernel']) == ('fp32','1024','1024','1024','register64'):
            print(r['experiment_id'], r['median_ms'], r['gflops'], r['verification_status'])
PY

# All records from the raw artifact store
ls results/raw/EXP-*.json

# Profiling index
cat results/profiling.jsonl
```

## Bottom line

The measurements support four high-confidence conclusions:

1. **Register tiling is the strongest custom FP32 optimization in this implementation family.**
2. **Register64 is faster despite lower occupancy because additional per-thread state/reuse outweighs the occupancy reduction for the tested large GEMMs.**
3. **Memory/coalescing/shared-memory optimizations are workload-sensitive and should not be described as universally beneficial.**
4. **cuBLAS remains the strongest measured reference, while the custom WMMA implementation demonstrates genuine Tensor Core execution but retains substantial headroom for further optimization.**
