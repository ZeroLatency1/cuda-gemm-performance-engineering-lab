#include "gemm.h"
#include <cuda_runtime.h>
#include <iostream>

// Simple float4 vectorized GEMM
// Assumes N is a multiple of 4.
__global__ void vectorized_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (row < M && col < N) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k < K; ++k) {
            float a = A[row * K + k];
            float4 b = *reinterpret_cast<const float4*>(&B[k * N + col]);
            sum.x += a * b.x;
            sum.y += a * b.y;
            sum.z += a * b.z;
            sum.w += a * b.w;
        }
        *reinterpret_cast<float4*>(&C[row * N + col]) = sum;
    }
}

void vectorized_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    if (N % 4 != 0) {
        std::cerr << "Vectorized GEMM requires N to be a multiple of 4.\n";
        return;
    }
    dim3 block(32, 8);
    dim3 grid((N/4 + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    vectorized_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}
