# Optimization Notes

## Progression of Optimizations

1. **CPU Baseline:** Standard memory access. Cache misses severely impact performance.
2. **Naive CUDA:** Offloaded to GPU, but memory bandwidth becomes the bottleneck.
3. **Coalescing:** Ensuring adjacent threads in a warp access adjacent memory addresses in global memory.
4. **Shared Memory Tiling:** Loading blocks of matrices into shared memory to reuse data and reduce global memory reads (Arithmetic Intensity increases).
5. **Register Tiling:** Increasing the work per thread to reuse data directly from registers, bypassing the latency of shared memory.
6. **Tensor Cores:** Using specialized hardware to perform 4x4 or 16x16 matrix multiplications in hardware in a single instruction.

*(Note: Actual measured speedups and tradeoffs for each stage are pending execution).*
