#include "correctness.h"

#include <algorithm>
#include <cmath>

CorrectnessMetrics compare_outputs(const std::vector<float>& ref,
                                   const std::vector<float>& got) {
    CorrectnessMetrics m;
    if (ref.size() != got.size() || ref.empty()) return m;

    // Relative error is max(abs(got-ref) / max(abs(ref), floor)).
    // The floor keeps tiny/near-zero reference values from dominating the metric.
    constexpr double REL_FLOOR = 1e-3;
    constexpr double ABS_TOL = 1e-3;
    constexpr double REL_TOL = 1e-2;

    bool all_ok = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = static_cast<double>(ref[i]);
        const double t = static_cast<double>(got[i]);
        if (std::isnan(t)) { m.saw_nan = true; return m; }
        if (std::isinf(t)) { m.saw_inf = true; return m; }

        const double abs_err = std::abs(t - r);
        const double rel_err = abs_err / std::max(std::abs(r), REL_FLOOR);
        m.max_abs = std::max(m.max_abs, abs_err);
        m.max_rel = std::max(m.max_rel, rel_err);
        if (abs_err > ABS_TOL && rel_err > REL_TOL) all_ok = false;
    }
    m.pass = all_ok;
    return m;
}
