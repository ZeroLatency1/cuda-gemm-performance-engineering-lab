# Functionality Matrix

| Capability | Implementation | Target-side verification required |
|---|---|---|
| CPU reference GEMM | `src/cpu_gemm.cpp` | Yes |
| Naive CUDA | `src/naive_cuda.cu` | Yes |
| Coalesced CUDA | `src/coalesced_cuda.cu` | Yes |
| Shared-memory 16/32 | `src/shared_mem_cuda.cu` | Yes |
| Register tiled 32/64 | `src/register_tiled_cuda.cu` | Yes |
| Float4 vectorized | `src/vectorized_cuda.cu` | Yes |
| Warp shuffle experiment | `src/warp_experiments.cu` | Yes |
| WMMA Tensor Core FP16/FP32 | `src/tensor_core_cuda.cu` | Yes, SM 8.9 |
| cuBLAS FP32 | `src/cublas_gemm.cu` | Yes |
| cuBLAS FP16 Tensor Core math | `src/cublas_gemm.cu` | Yes |
| CUDA-event kernel timing | `src/benchmark.cu` | Yes |
| Separate H2D/D2H/end-to-end timing | `src/benchmark.cu` | Yes |
| Append-only CSV/JSONL | `src/benchmark.cu` | Yes |
| Unique raw experiment artifacts | `src/benchmark.cu` | Yes |
| Explicit UNSUPPORTED vs ERROR | `src/benchmark.cu` | Yes |
| Project state automation | `scripts/update_project_state.py` | Yes |
| Plot generation from stored results | `scripts/generate_plots.py` | Yes |
| Profiling limitation capture | `scripts/run_profile.sh` | Yes |
