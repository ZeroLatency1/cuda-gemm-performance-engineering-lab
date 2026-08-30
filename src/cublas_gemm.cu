#include "gemm.h"
#include "utils.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <iostream>
#include <stdexcept>

namespace {
class CublasContext {
public:
    explicit CublasContext(bool tensor_op = false) {
        const cublasStatus_t rc = cublasCreate(&handle_);
        if (rc != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed with status " + std::to_string(static_cast<int>(rc)));
        if (tensor_op) {
            const cublasStatus_t mode_rc = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
            if (mode_rc != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSetMathMode(TENSOR_OP) failed with status " + std::to_string(static_cast<int>(mode_rc)));
        }
    }
    ~CublasContext() {
        if (handle_) cublasDestroy(handle_);
    }
    cublasHandle_t get() const { return handle_; }

    CublasContext(const CublasContext&) = delete;
    CublasContext& operator=(const CublasContext&) = delete;
private:
    cublasHandle_t handle_{};
};

CublasContext& default_context() {
    static CublasContext ctx(false);
    return ctx;
}

CublasContext& tensor_context() {
    static CublasContext ctx(true);
    return ctx;
}
}

void cublas_gemm(int M, int N, int K, const float* d_A, const float* d_B, float* d_C) {
    cublasHandle_t h = default_context().get();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t rc = cublasSgemm(
        h, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N);
    if (rc != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSgemm failed with status " + std::to_string(static_cast<int>(rc)));
}

void cublas_gemm_half(int M, int N, int K, const __half* d_A, const __half* d_B, float* d_C) {
    cublasHandle_t h = tensor_context().get();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t rc = cublasGemmEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (rc != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx FP16 failed with status " + std::to_string(static_cast<int>(rc)));
}
