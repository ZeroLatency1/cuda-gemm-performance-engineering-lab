#include "gemm.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
__global__ void vectorized_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (row >= M || col >= N) return;

    float4 sum = make_float4(0, 0, 0, 0);
    for (int k = 0; k < K; ++k) {
        const float a = A[row * K + k];
        const float4 b = *reinterpret_cast<const float4*>(B + static_cast<size_t>(k) * N + col);
        sum.x += a * b.x;
        sum.y += a * b.y;
        sum.z += a * b.z;
        sum.w += a * b.w;
    }
    *reinterpret_cast<float4*>(C + static_cast<size_t>(row) * N + col) = sum;
}
}

void vectorized_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    if (N % 4 != 0) throw std::runtime_error("vectorized kernel requires N divisible by 4");
    constexpr int BX = 32, BY = 8;
    vectorized_gemm_kernel<<<dim3((N / 4 + BX - 1) / BX, (M + BY - 1) / BY), dim3(BX, BY)>>>(M, N, K, d_A, d_B, d_C);
}
