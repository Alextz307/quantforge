#include "quant/strategies/state_machines.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace quant::strategies {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[nodiscard]] inline bool any_nan(double a, double b, double c, double d) noexcept {
    return std::isnan(a) || std::isnan(b) || std::isnan(c) || std::isnan(d);
}

}  // namespace

std::vector<double> run_mean_reversion_state_machine(
    std::span<const double> close,
    std::span<const double> mid,
    std::span<const double> upper,
    std::span<const double> lower,
    std::span<const double> trend_ma)
{
    const auto n = close.size();
    if (mid.size() != n || upper.size() != n || lower.size() != n ||
        trend_ma.size() != n) {
        throw std::invalid_argument(
            "run_mean_reversion_state_machine: input spans must all have the same length");
    }

    std::vector<double> out(n, kNaN);
    double position = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        if (any_nan(mid[t], upper[t], lower[t], trend_ma[t])) {
            continue;
        }
        const bool is_bull = close[t] > trend_ma[t];
        if (position == 0.0) {
            if (is_bull && close[t] < lower[t]) {
                position = 1.0;
            } else if (!is_bull && close[t] > upper[t]) {
                position = -1.0;
            }
        } else if (position == 1.0) {
            if (close[t] >= mid[t]) {
                position = 0.0;
            }
        } else {  // position == -1.0
            if (close[t] <= mid[t]) {
                position = 0.0;
            }
        }
        out[t] = position;
    }
    return out;
}

std::vector<double> run_pairs_state_machine(
    std::span<const double> zscore,
    double entry_zscore,
    double exit_zscore,
    double stop_loss_zscore)
{
    const auto n = zscore.size();
    std::vector<double> out(n, kNaN);
    double position = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
        const double z = zscore[t];
        if (std::isnan(z)) {
            continue;
        }
        const double abs_z = std::abs(z);
        if (abs_z >= stop_loss_zscore) {
            position = 0.0;
        } else if (position == 0.0) {
            if (z >= entry_zscore) {
                position = -1.0;
            } else if (z <= -entry_zscore) {
                position = 1.0;
            }
        } else if (abs_z <= exit_zscore) {
            position = 0.0;
        }
        out[t] = position;
    }
    return out;
}

}  // namespace quant::strategies
