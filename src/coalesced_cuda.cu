#include "gemm.h"
#include <cuda_runtime.h>

__global__ void coalesced_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    // Explicit mapping to ensure consecutive threads read consecutive memory addresses
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A is accessed by row (not coalesced, same for all threads in warp)
            // B is accessed by col (coalesced, different for each thread in warp)
            sum += A[y * K + k] * B[k * N + x];
        }
        C[y * N + x] = sum;
    }
}

void coalesced_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    coalesced_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}
