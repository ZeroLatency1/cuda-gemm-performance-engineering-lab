#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \""                        \
                      << cudaGetErrorString(err) << "\"" << std::endl;   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

struct BenchmarkResult {
    double median_latency_ms;
    double p95_latency_ms;
    double min_latency_ms;
    double gflops;
    bool verified;
    double max_abs_err;
    double max_rel_err;
};

void randomize_matrix(float* mat, int size, int seed);
void randomize_matrix_half(void* mat, int size, int seed); // using void* for __half
bool verify_matrices(const float* ref, const float* test, int size, double abs_tol, double rel_tol, double& out_max_abs, double& out_max_rel);

