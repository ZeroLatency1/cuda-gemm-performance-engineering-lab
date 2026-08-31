#pragma once

#include <vector>

struct CorrectnessMetrics {
    bool pass = false;
    double max_abs = 0.0;
    double max_rel = 0.0;
    bool saw_nan = false;
    bool saw_inf = false;
};

CorrectnessMetrics compare_outputs(const std::vector<float>& ref,
                                   const std::vector<float>& got);
