# Functionality Matrix

The fresh build preserves the functional surface of the original KUCH repository while replacing the benchmark/result layer with append-only experiment tracking.

| Original capability | Fresh project |
|---|---|
| CPU GEMM | Included |
| Alternate CPU loop order | Included |
| Naive CUDA GEMM | Included |
| Coalesced CUDA GEMM | Included |
| Shared-memory 16x16 | Included |
| Shared-memory 32x32 | Included |
| Register-tiled CUDA | Included |
| Vectorized CUDA | Included |
| Warp/shuffle experiment | Included |
| Tensor Core / WMMA FP16 input, FP32 accumulate | Included |
| cuBLAS SGEMM | Included, persistent handle |
| CUDA-event kernel timing | Included |
| Warmup / repeated measurements | Included |
| Median / p95 / minimum | Included |
| GFLOPS from median latency | Included |
| Correctness verification | Included |
| Odd/non-square dimensions | Included |
| Append-only CSV history | Included |
| Append-only JSONL history | Included |
| Unique experiment IDs | Included |
| Results path independent of launch directory | Included |
| Per-experiment raw JSON artifacts | Included |
| Git commit captured per experiment | Included |
| GPU/CUDA metadata captured per experiment | Included |
| WSL/profiler limitation recording | Included |
| Original React/Vite frontend files | Preserved |
| Original architecture/memory/optimization/profiling documentation | Preserved |
