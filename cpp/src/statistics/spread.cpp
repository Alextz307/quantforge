#include "quant/statistics/spread.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "quant/core/validation.hpp"

namespace quant::statistics {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kZScoreDDoF = 1;

}  // namespace

std::vector<double> SpreadCalculator::compute_spread(
    std::span<const double> a,
    std::span<const double> b,
    double hedge_ratio)
{
    std::vector<double> out(a.size());
    compute_spread(a, b, hedge_ratio, out);
    return out;
}

void SpreadCalculator::compute_spread(
    std::span<const double> a,
    std::span<const double> b,
    double hedge_ratio,
    std::span<double> out)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "SpreadCalculator::compute_spread: a and b must have the same length");
    }
    detail::check_out_size(a.size(), out.size(), "SpreadCalculator::compute_spread");
    const auto n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] - hedge_ratio * b[i];
    }
}

std::vector<double> SpreadCalculator::compute_zscore(
    std::span<const double> spread,
    int window)
{
    std::vector<double> out(spread.size());
    compute_zscore(spread, window, out);
    return out;
}

// Keep the Welford recurrence below in sync with detail::rolling_mean_std.
// NaN semantics diverge from pandas — see the header docstring.
void SpreadCalculator::compute_zscore(
    std::span<const double> spread,
    int window,
    std::span<double> out)
{
    if (window < 2) {
        throw std::invalid_argument(
            "SpreadCalculator::compute_zscore: window must be >= 2");
    }
    detail::check_out_size(spread.size(), out.size(), "SpreadCalculator::compute_zscore");
    const auto n = spread.size();
    std::fill(out.begin(), out.end(), kNaN);
    if (static_cast<int>(n) < window) {
        return;
    }

    const double denom = static_cast<double>(window - kZScoreDDoF);

    double sum = 0.0;
    double w_mean = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < window; ++i) {
        sum += spread[i];
        const double delta = spread[i] - w_mean;
        w_mean += delta / (i + 1);
        m2 += delta * (spread[i] - w_mean);
    }
    {
        const double mean = sum / window;
        const double var = m2 / denom;
        const double sd = std::sqrt(std::max(0.0, var));
        if (sd > 0.0) {
            out[window - 1] = (spread[window - 1] - mean) / sd;
        }
    }

    const auto nn = static_cast<int>(n);
    for (int i = window; i < nn; ++i) {
        const double old_val = spread[i - window];
        const double new_val = spread[i];
        sum += new_val - old_val;
        const double old_wmean = w_mean;
        w_mean += (new_val - old_val) / window;
        m2 += (new_val - old_val) * (new_val - w_mean + old_val - old_wmean);
        const double mean = sum / window;
        const double var = m2 / denom;
        const double sd = std::sqrt(std::max(0.0, var));
        if (sd > 0.0) {
            out[i] = (new_val - mean) / sd;
        }
    }
}

}  // namespace quant::statistics
