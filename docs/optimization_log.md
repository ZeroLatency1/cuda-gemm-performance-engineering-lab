# Optimization Log

This log is derived from the recorded optimization-lineage experiments. The authoritative numerical history remains in `results/experiments.csv`, `results/experiments.jsonl`, and `results/raw/`.

| Experiment | Hypothesis / change | Parent | Baseline | Median ms | GFLOPS | Verification | Decision |
|---|---|---|---|---:|---:|---|---|
| EXP-000003 | Change global-memory access pattern from naive indexing to coalesced loads/stores | EXP-000002 | EXP-000002 | — | 716.26 | PASS | **Context-dependent**; coalescing helps some shapes but is not universally faster |
| EXP-000004 | Introduce 16x16 shared-memory tiling to reuse A/B data | EXP-000002 | EXP-000002 | — | 834.69 | PASS | **Accepted as a valid optimization path; shape-sensitive** |
| EXP-000006 | Repeatability measurement of corrected naive 1024^3 FP32 baseline | EXP-000005 | EXP-000005 | — | 980.72 | PASS | Baseline repeatability |
| EXP-000007 | Repeatability measurement of corrected naive 1024^3 FP32 baseline | EXP-000005 | EXP-000005 | — | 981.02 | PASS | Baseline repeatability |
| EXP-000008 | Controlled 16x16 shared-memory tiling follow-up | EXP-000007 | EXP-000007 | — | 867.51 | PASS | **Useful but not dominant** |
| EXP-000009 | Increase shared-memory tile to 32x32 | EXP-000007 | EXP-000007 | — | 839.58 | PASS | **Rejected relative to shared16 for this controlled case** |
| EXP-000010 | Introduce register tiling for greater output reuse per thread | EXP-000007 | EXP-000007 | — | 1589.96 | PASS | **Accepted** |
| EXP-000011 | Increase register-tile footprint to evaluate register pressure vs reuse | EXP-000010 | EXP-000007 | — | 2058.05 | PASS | **Accepted; strongest custom direction** |
| EXP-000012 | Evaluate vectorized global-memory access after register tiling | EXP-000011 | EXP-000007 | — | 1983.30 | PASS | **Accepted as a strong alternative; below register64 in the controlled run** |
| EXP-000013 | Evaluate warp-oriented GEMM execution | EXP-000011 | EXP-000007 | — | 107.44 | PASS | **Rejected; severe regression** |
| EXP-000043 | Final repeatability measurement of best custom FP32 kernel | EXP-000011 | EXP-000007 | — | 2068.20 | PASS | Repeatability |
| EXP-000044 | Final repeatability measurement of best custom FP32 kernel | EXP-000011 | EXP-000007 | — | 2071.26 | PASS | Repeatability |
| EXP-000045 | Final repeatability measurement of best custom FP32 kernel | EXP-000011 | EXP-000007 | — | 2062.10 | PASS | Repeatability |
| EXP-000277 | Metadata persistence test for profiling lineage | EXP-000274 | — | 1456.17* | 0.02* | NOT_VERIFIED | Infrastructure validation |
| EXP-000278 | Baseline forwarding test | EXP-000274 | EXP-000274 | 1456.17* | 0.02* | NOT_VERIFIED | Infrastructure validation |

`*` Nsight-instrumented runs are diagnostic artifacts and are **not** eligible for performance comparison. The profile index explicitly marks them `performance_claim_eligible=false`.

## Controlled optimization results from the repeated benchmark matrix

The later repeated matrix provides the strongest apples-to-apples evidence. At 1024^3 FP32, the median improvements over the naive kernel are:

- Coalesced: **+35.0%**
- Shared16: **+19.6%**
- Shared32: **+16.6%**
- Register: **+125.3%**
- Register64: **+291.5% / 3.91x**
- Vectorized: **+272.7% / 3.73x**
- Warp: **-85.6%**
- cuBLAS: **+1080% / 11.80x**

These values are calculated from the standardized six-run median groups, not from profiler timing.

## Microarchitectural interpretation

Nsight Compute supports the register-reuse conclusion:

| Profiled kernel | Registers/thread | Theoretical occupancy | Achieved occupancy |
|---|---:|---:|---:|
| register | 39 | 100% | 98.74% |
| register64 | 56 | 66.67% | 64.75% |
| vectorized | 40 | 100% | 92.54% |
| cuBLAS FP32 | 122 | 33.33% | 31.79% |
| WMMA FP16 | 40 | 100% | 97.75% |

The key conclusion is not “maximize occupancy.” The measured register64 result shows that additional register-resident reuse can be more valuable than maximum occupancy for this GEMM workload.

The WMMA profile also contains direct Tensor Core evidence: `smsp__inst_executed_pipe_tensor_op_hmma_v2.sum = 4194304` and `smsp__pipe_tensor_op_hmma_cycles_active_v2.sum = 67108864`. The custom WMMA path therefore genuinely exercises the HMMA Tensor Core execution path.

## Recording rule

New optimization claims should continue to follow the existing policy: record the experiment ID, controlled workload, baseline, hypothesis, measured result, correctness state, and decision. Nsight timing remains diagnostic only.


## Audited optimization outcomes (final dataset)

The completed RTX 4060 dataset replaces the earlier planning-stage expectation with measured results. All figures below use the standardized repeated benchmark groups (10 warmups, 50 timed iterations) and median kernel throughput.

### 1024^3 FP32

| Stage | Median GFLOPS | Relative to naive | Decision |
|---|---:|---:|---|
| Naive | 716.92 | 1.00x | Baseline |
| Coalesced | 967.83 | +35.0% | Accepted as a shape/workload improvement |
| Shared16 | 857.56 | +19.6% | Useful, but not dominant |
| Shared32 | 836.07 | +16.6% | Useful, but not dominant |
| Register | 1614.97 | +125.3% | Strong improvement |
| Register64 | **2806.70** | **+291.5% / 3.91x** | **Best custom FP32 variant** |
| Vectorized | 2671.83 | +272.7% / 3.73x | Strong alternative |
| Warp | 103.46 | -85.6% | Rejected |
| cuBLAS | **8460.93** | **11.80x** | Vendor reference |

### Scale dependence

The same custom strategy does not behave identically at every shape. Register64 remains the strongest custom FP32 kernel at 2048^3 and 4096^3, while coalescing and shared-memory variants become comparatively workload-sensitive. The warp-oriented experiment regresses severely at large sizes and is retained as a documented negative result.

### Profiling evidence

Nsight Compute measured **39 registers/thread, 100% theoretical occupancy** for the 32x32 register kernel and **56 registers/thread, 66.67% theoretical occupancy** for register64. The faster register64 result therefore demonstrates the benefit of increased per-thread reuse despite lower occupancy.

For the custom WMMA FP16 path, the final focused profile recorded **4,194,304 HMMA instructions**, **67,108,864 HMMA-pipe active cycles**, and **17,179,869,184 FP16-to-FP32 tensor-path operations**, directly confirming Tensor Core execution.

## Final status

The optimization investigation is complete for the current release snapshot. No additional kernel changes are required to justify the documented conclusions. Future optimization work should begin from the frozen release rather than rewriting historical results.
