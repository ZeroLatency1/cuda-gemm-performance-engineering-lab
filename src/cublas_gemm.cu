#include "gemm.h"
#include <cublas_v2.h>
#include <iostream>

void cublas_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed\n";
        return;
    }
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS uses column-major by default, but we have row-major matrices.
    // C^T = B^T * A^T
    // So we can compute using CUBLAS_OP_N and swap A and B.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    cublasDestroy(handle);
}
