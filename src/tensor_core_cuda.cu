#include "gemm.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Basic WMMA configuration
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_gemm_kernel(int M, int N, int K, const __half* A, const __half* B, float* C) {
    // Determine the position of the warp
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = blockIdx.x * blockDim.x + threadIdx.x;

    // A simple grid stride to allow smaller grids if necessary
    int lda = K;
    int ldb = N;
    int ldc = N;
    
    // Check bounds. Assuming M, N, K are multiples of 16 for simplicity in this experiment.
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * lda + i, lda);
        wmma::load_matrix_sync(b_frag, B + i * ldb + warpN * WMMA_N, ldb);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * WMMA_M * ldc + warpN * WMMA_N, c_frag, ldc, wmma::mem_row_major);
}

void tensor_core_cuda_gemm(int M, int N, int K, const __half* d_A, const __half* d_B, float* d_C) {
    dim3 block(32, 4); // 4 warps
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + 4 * WMMA_M - 1) / (4 * WMMA_M)); // Grid covers N and M
    wmma_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}
