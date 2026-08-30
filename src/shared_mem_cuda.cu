#include "gemm.h"
#include <cuda_runtime.h>

template <int TILE_SIZE>
__global__ void shared_mem_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        if (row < M && ph * TILE_SIZE + tx < K)
            sA[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;

        if (col < N && ph * TILE_SIZE + ty < K)
            sB[ty][tx] = B[(ph * TILE_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void shared_mem_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C, int tile_size) {
    if (tile_size == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        shared_mem_gemm_kernel<16><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    } else if (tile_size == 32) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        shared_mem_gemm_kernel<32><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    }
}
