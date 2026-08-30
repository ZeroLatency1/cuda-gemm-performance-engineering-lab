#include "gemm.h"
#include <cuda_runtime.h>

__global__ void naive_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void naive_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}
