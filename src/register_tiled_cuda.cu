#include "gemm.h"
#include <cuda_runtime.h>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void register_tiled_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * BM + ty * TM;
    int col = bx * BN + tx * TN;

    float rC[TM][TN] = {0.0f};

    // Number of threads in a block
    int num_threads = blockDim.x * blockDim.y;
    int tid = ty * blockDim.x + tx;

    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) {
        // Load A to shared memory cooperatively
        for (int i = tid; i < BM * BK; i += num_threads) {
            int r = i / BK;
            int c = i % BK;
            int g_r = by * BM + r;
            int g_c = ph * BK + c;
            sA[r][c] = (g_r < M && g_c < K) ? A[g_r * K + g_c] : 0.0f;
        }
        
        // Load B to shared memory cooperatively
        for (int i = tid; i < BK * BN; i += num_threads) {
            int r = i / BN;
            int c = i % BN;
            int g_r = ph * BK + r;
            int g_c = bx * BN + c;
            sB[r][c] = (g_r < K && g_c < N) ? B[g_r * N + g_c] : 0.0f;
        }

        __syncthreads();

        // Compute outer product
        for (int k = 0; k < BK; ++k) {
            float rA[TM];
            float rB[TN];
            
            for (int i = 0; i < TM; ++i) rA[i] = sA[ty * TM + i][k];
            for (int j = 0; j < TN; ++j) rB[j] = sB[k][tx * TN + j];

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + col + j] = rC[i][j];
            }
        }
    }
}

void register_tiled_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    // Basic configuration BM=32, BN=32, BK=8, TM=2, TN=2
    // Block size is 16x16
    const int BM = 32, BN = 32, BK = 8;
    const int TM = 2, TN = 2;
    dim3 block(BN/TN, BM/TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    register_tiled_gemm_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}

// Experimental 64x64 output tile. This is deliberately separate from the original
// register-tiled kernel so optimization experiments remain comparable.
void register_tiled_cuda_gemm_64(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    constexpr int BM = 64, BN = 64, BK = 8;
    constexpr int TM = 4, TN = 4;
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    register_tiled_gemm_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}

