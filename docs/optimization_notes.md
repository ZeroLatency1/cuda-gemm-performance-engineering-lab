# Optimization Notes

The implementation evolves from a scalar CPU reference into progressively more specialized GPU kernels. Each stage isolates a specific execution or data-movement strategy.

## Optimization progression

1. **CPU reference:** serial reference implementation used for numerical validation.
2. **Naive CUDA:** one output element per thread with direct global-memory access.
3. **Coalesced access:** reorganized global-memory indexing so adjacent threads access adjacent elements where possible.
4. **Shared-memory tiling:** staged input tiles through on-chip shared memory to increase data reuse.
5. **Register tiling:** assigned each thread multiple output elements so partial results and reused operands remain in registers.
6. **Vectorization:** used `float4` memory operations where alignment and shape permitted.
7. **Warp-oriented execution:** explored warp-level decomposition and shuffle-based coordination.
8. **WMMA Tensor Cores:** used the FP16 matrix-multiply path with FP32 accumulation.
9. **cuBLAS:** retained as the optimized vendor-library reference.

## Measured 1024^3 FP32 results

| Strategy | Median GFLOPS | Relative to naive | Observation |
|---|---:|---:|---|
| Naive | 716.92 | 1.00x | Baseline |
| Coalesced | 967.83 | 1.35x | +35.0% |
| Shared16 | 857.56 | 1.20x | +19.6% |
| Shared32 | 836.07 | 1.17x | +16.6% |
| Register | 1614.97 | 2.25x | +125.3% |
| Register64 | 2806.70 | 3.91x | +291.5% |
| Vectorized | 2671.83 | 3.73x | +272.7% |
| Warp | 103.46 | 0.14x | -85.6% |
| cuBLAS | 8460.93 | 11.80x | Vendor reference |

These figures are workload-specific measurements. The relative ranking changes with matrix shape and precision.
