#include "gemm.h"
#include <cuda_runtime.h>

namespace {
template <int BM, int BN, int BK, int TM, int TN>
__global__ void register_tiled_gemm_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int threads = blockDim.x * blockDim.y;
    const int base_row = blockIdx.y * BM + ty * TM;
    const int base_col = blockIdx.x * BN + tx * TN;
    float rC[TM][TN]{};

    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) {
        for (int idx = tid; idx < BM * BK; idx += threads) {
            const int r = idx / BK, c = idx % BK;
            const int gr = blockIdx.y * BM + r, gc = ph * BK + c;
            sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += threads) {
            const int r = idx / BN, c = idx % BN;
            const int gr = ph * BK + r, gc = blockIdx.x * BN + c;
            sB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) rC[i][j] += rA[i] * rB[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            if (base_row + i < M && base_col + j < N)
                C[(base_row + i) * N + base_col + j] = rC[i][j];
}
}

void register_tiled_cuda_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    constexpr int BM = 32, BN = 32, BK = 8, TM = 2, TN = 2;
    register_tiled_gemm_kernel<BM, BN, BK, TM, TN><<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), dim3(BN / TN, BM / TM)>>>(M, N, K, d_A, d_B, d_C);
}

void register_tiled_cuda_gemm_64(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    register_tiled_gemm_kernel<BM, BN, BK, TM, TN><<<dim3((N + BN - 1) / BN, (M + BM - 1) / BM), dim3(BN / TN, BM / TM)>>>(M, N, K, d_A, d_B, d_C);
}
