#include "utils.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>

void randomize_matrix(float* mat, int size, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        mat[i] = dis(gen);
    }
}

void randomize_matrix_half(void* mat, int size, int seed) {
    __half* h_mat = (__half*)mat;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size; ++i) {
        h_mat[i] = __float2half(dis(gen));
    }
}

bool verify_matrices(const float* ref, const float* test, int size, double abs_tol, double rel_tol, double& out_max_abs, double& out_max_rel) {
    out_max_abs = 0.0;
    out_max_rel = 0.0;
    bool passed = true;

    for (int i = 0; i < size; ++i) {
        double r = ref[i];
        double t = test[i];
        
        if (std::isnan(t) || std::isinf(t)) {
            return false;
        }

        double abs_err = std::abs(r - t);
        double rel_err = abs_err / (std::abs(r) + 1e-5);

        out_max_abs = std::max(out_max_abs, abs_err);
        out_max_rel = std::max(out_max_rel, rel_err);

        if (abs_err > abs_tol && rel_err > rel_tol) {
            passed = false;
        }
    }
    return passed;
}
