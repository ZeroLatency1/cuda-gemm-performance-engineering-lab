#include "gemm.h"
#include <iostream>

// Standard i-j-k loop order
void cpu_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// i-k-j loop order for better cache locality (B matrix accessed linearly)
void cpu_gemm_loop_order(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}
