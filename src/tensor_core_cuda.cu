#include "gemm.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdexcept>

using namespace nvcuda;

namespace {
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
constexpr int WARPS_PER_BLOCK = 4;

__global__ void wmma_gemm_kernel(int M, int N, int K, const __half* A, const __half* B, float* C) {
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int warp = threadIdx.y;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp;
    const int row = tile_m * WM;
    const int col = tile_n * WN;
    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WK) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(row) * K + k, K);
        wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(k) * N + col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + static_cast<size_t>(row) * N + col, c_frag, N, wmma::mem_row_major);
}
}

void tensor_core_cuda_gemm(int M, int N, int K, const __half* d_A, const __half* d_B, float* d_C) {
    if ((M % WM) || (N % WN) || (K % WK)) throw std::runtime_error("WMMA kernel requires M, N, and K divisible by 16");
    const dim3 block(32, WARPS_PER_BLOCK);
    const dim3 grid((N / WN + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, M / WM);
    wmma_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
}
