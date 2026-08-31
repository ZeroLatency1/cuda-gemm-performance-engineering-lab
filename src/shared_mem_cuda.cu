#include "gemm.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
template <int TILE>
__global__ void shared_mem_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) {
        const int a_col = ph * TILE + tx;
        const int b_row = ph * TILE + ty;
        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
}

void shared_mem_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C, int tile_size) {
    if (tile_size == 16) {
        constexpr int T = 16;
        shared_mem_gemm_kernel<T><<<dim3((N + T - 1) / T, (M + T - 1) / T), dim3(T, T)>>>(M, N, K, d_A, d_B, d_C);
    } else if (tile_size == 32) {
        constexpr int T = 32;
        shared_mem_gemm_kernel<T><<<dim3((N + T - 1) / T, (M + T - 1) / T), dim3(T, T)>>>(M, N, K, d_A, d_B, d_C);
    } else {
        throw std::runtime_error("unsupported shared-memory tile size: " + std::to_string(tile_size));
    }
}
