#pragma once

#include <vector>
#include <chrono>
#include <cuda_fp16.h>

// CPU Implementations
void cpu_gemm(int M, int N, int K, const float* A, const float* B, float* C);
void cpu_gemm_loop_order(int M, int N, int K, const float* A, const float* B, float* C);

// CUDA Implementations
void naive_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C);
void coalesced_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C);
void shared_mem_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C, int tile_size);
void register_tiled_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C);
void register_tiled_cuda_gemm_64(int M, int N, int K, const float* d_A, const float* d_B, float* d_C);
void vectorized_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C);
void warp_shuffle_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C);
void tensor_core_cuda_gemm(int M, int N, int K, const __half* d_A, const __half* d_B, float* d_C);
void cublas_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C);
