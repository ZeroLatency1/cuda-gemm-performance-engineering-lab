#include "gemm.h"
#include <cuda_runtime.h>

namespace {
__global__ void coalesced_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    // Threads traverse output columns consecutively, so B and C accesses are warp-coalesced.
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
}

void coalesced_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    constexpr int BX = 32, BY = 8;
    const dim3 block(BX, BY);
    const dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
    coalesced_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}
