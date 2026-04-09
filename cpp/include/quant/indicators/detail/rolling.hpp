#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace quant::detail {

/// Sliding-window rolling mean.
/// First (window - 1) values are NaN.
[[nodiscard]] inline std::vector<double> rolling_mean(
    std::span<const double> data,
    int window)
{
    const auto n = static_cast<int>(data.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(data.size(), nan);

    if (n < window || window < 1) {
        return result;
    }

    // Compute initial window sum
    double sum = 0.0;
    for (int i = 0; i < window; ++i) {
        sum += data[i];
    }
    result[window - 1] = sum / window;

    // Slide the window
    for (int i = window; i < n; ++i) {
        sum += data[i] - data[i - window];
        result[i] = sum / window;
    }

    return result;
}

/// Sliding-window rolling standard deviation using Welford's online algorithm.
/// Numerically stable — avoids catastrophic cancellation from sum-of-squares formulas.
/// @param ddof Degrees of freedom (1 = sample std, 0 = population std). Default 1 to match pandas.
/// First (window - 1) values are NaN.
[[nodiscard]] inline std::vector<double> rolling_std(
    std::span<const double> data,
    int window,
    int ddof = 1)
{
    const auto n = static_cast<int>(data.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(data.size(), nan);

    if (n < window || window < 1 || window <= ddof) {
        return result;
    }

    const double denom = static_cast<double>(window - ddof);

    // Phase 1: Compute mean and variance for the first window using Welford's method
    double mean = 0.0;
    double m2 = 0.0;  // sum of squared deviations from the running mean
    for (int i = 0; i < window; ++i) {
        double delta = data[i] - mean;
        mean += delta / (i + 1);
        double delta2 = data[i] - mean;
        m2 += delta * delta2;
    }
    result[window - 1] = std::sqrt(std::max(0.0, m2 / denom));

    // Phase 2: Slide the window using the update formula
    // When removing old_val and adding new_val:
    //   new_mean = old_mean + (new_val - old_val) / window
    //   m2 += (new_val - old_val) * (new_val - new_mean + old_val - old_mean)
    for (int i = window; i < n; ++i) {
        double old_val = data[i - window];
        double new_val = data[i];
        double old_mean = mean;
        mean += (new_val - old_val) / window;
        m2 += (new_val - old_val) * (new_val - mean + old_val - old_mean);
        result[i] = std::sqrt(std::max(0.0, m2 / denom));
    }

    return result;
}

}  // namespace quant::detail
