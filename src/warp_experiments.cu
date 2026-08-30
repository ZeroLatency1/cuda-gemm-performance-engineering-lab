#include "gemm.h"
#include <cuda_runtime.h>
#include <iostream>

// Warp-synchronous shuffle reduction experiment
// 32 threads collaborate to compute a single output element's dot product.
__global__ void warp_shuffle_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int global_warp_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y / 32) + 
                         (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    
    int row = global_warp_id / N;
    int col = global_warp_id % N;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = lane_id; k < K; k += 32) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Warp shuffle reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) {
            C[row * N + col] = sum;
        }
    }
}

void warp_shuffle_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    int total_elements = M * N;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int total_blocks = (total_elements + warps_per_block - 1) / warps_per_block;
    
    warp_shuffle_gemm_kernel<<<total_blocks, threads_per_block>>>(M, N, K, d_A, d_B, d_C);
}
