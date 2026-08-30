# CUDA GEMM Performance Engineering Portfolio

## Project Overview

This repository is a serious CUDA GEMM performance-engineering portfolio project. It demonstrates the progression of matrix multiplication optimizations from a CPU baseline up through Tensor Cores and cuBLAS.

The kernels focus on:
- Global memory coalescing
- Shared memory tiling
- Register tiling
- Tensor Cores (WMMA)
- Roofline-style performance reasoning

## Target Hardware & Environment

The expected target environment for this project is:
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- **Architecture:** Ada Lovelace (Compute Capability 8.9)
- **CUDA Toolkit:** 12.0.140
- **Host Compiler:** GCC/G++ 13.3.0
- **OS:** Ubuntu 24.04 under WSL2 on Windows

### Current Status

**Note:** The autonomous agent built this repository structure and the implementations. However, the agent is executing inside a serverless container environment (Node.js/React environment on Cloud Run) that lacks an NVIDIA GPU and the CUDA toolkit (`nvcc`). 

Therefore, **compilation, testing, verification, benchmarking, and profiling are currently BLOCKED.**
All performance results and profiling sections are marked as **PENDING** until the code is executed on the intended local hardware.

## Kernel Progression

1. **CPU Baseline (`src/cpu_gemm.cpp`)**
   - Standard i-j-k loop order.
   - Cache-friendly i-k-j loop order for reference.

2. **Naive CUDA (`src/naive_cuda.cu`)**
   - One thread per output element.
   - Direct global memory access.

3. **Coalesced CUDA (`src/coalesced_cuda.cu`)**
   - Ensures memory accesses are coalesced within warps.

4. **Shared-Memory Tiled CUDA (`src/shared_mem_cuda.cu`)**
   - Uses shared memory to cache blocks of A and B.
   - Supports 16x16 and 32x32 tiles.

5. **Register-Tiled CUDA (`src/register_tiled_cuda.cu`)**
   - Reduces shared memory accesses by computing a 2x2 tile of the output per thread in registers.

6. **Tensor Core CUDA (`src/tensor_core_cuda.cu`)**
   - Uses WMMA API for hardware acceleration on Ada Lovelace.

7. **cuBLAS Reference (`src/cublas_gemm.cu`)**
   - Baseline for peak achievable performance.

## Build Instructions (Local Hardware)

If you have cloned this repository to an environment with the CUDA Toolkit and an NVIDIA GPU:

```bash
mkdir build && cd build
cmake ..
make
./cuda_gemm_benchmark
```

## Performance Results

### PENDING MEASUREMENT
*Due to the lack of an NVIDIA GPU in the execution environment, all performance metrics (latency, GFLOPS, speedup) are pending measurement on the target hardware. No results are fabricated.*

## Profiling Methodology

*Nsight Systems and Nsight Compute usage are pending execution on the target hardware. No profiler outputs have been generated.*

## Limitations

- **Environment Restriction:** The code cannot be compiled or verified by the agent due to missing `nvcc` and GPU.
- **Verification Pending:** Numerical correctness verification must be run locally.
- **Tensor Core Path:** The WMMA implementation requires testing on Compute Capability 8.9 to verify alignment and precision behaviors.


## Canonical result-preservation policy

This repository preserves benchmark evidence rather than replacing it. Every benchmark result is appended to `results/experiments.csv` and `results/experiments.jsonl` under a unique experiment ID. Performance-only runs are explicitly recorded as `NOT_VERIFIED`.

The original kernel families are retained, including CPU, naive, coalesced, shared-memory, register-tiled, vectorized, warp, Tensor Core, and cuBLAS implementations. An additional `register64` experimental variant is included for controlled optimization work.

Official performance history begins with the first clean baseline run of this fresh project. Historical numbers from earlier attempts are not mixed into the official dataset.
