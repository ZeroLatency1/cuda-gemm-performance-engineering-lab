#include "gemm.h"
#include "utils.h"
#include <cuda_runtime.h>

namespace {
__global__ void naive_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
}

void naive_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    constexpr int BX = 16, BY = 16;
    const dim3 block(BX, BY);
    const dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
    naive_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}
