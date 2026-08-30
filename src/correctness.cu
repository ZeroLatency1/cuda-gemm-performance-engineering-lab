#include "correctness.h"
#include <algorithm>
#include <cmath>
#include <limits>

CorrectnessMetrics compare_outputs(const std::vector<float>& ref,
                                   const std::vector<float>& got) {
    CorrectnessMetrics m;
    if (ref.size() != got.size() || ref.empty()) return m;

    // Use a floor so relative error does not become meaningless near zero.
    constexpr double REL_FLOOR = 1e-3;
    constexpr double ABS_TOL = 1e-3;
    constexpr double REL_TOL = 1e-2;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double value = static_cast<double>(got[i]);
        if (std::isnan(value)) { m.saw_nan = true; return m; }
        if (std::isinf(value)) { m.saw_inf = true; return m; }
        const double abs_err = std::abs(value - static_cast<double>(ref[i]));
        const double denom = std::max(std::abs(static_cast<double>(ref[i])), REL_FLOOR);
        const double rel_err = abs_err / denom;
        m.max_abs = std::max(m.max_abs, abs_err);
        m.max_rel = std::max(m.max_rel, rel_err);
    }

    m.pass = (m.max_abs <= ABS_TOL) || (m.max_rel <= REL_TOL);
    return m;
}
