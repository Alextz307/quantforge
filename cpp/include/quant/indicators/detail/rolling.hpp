#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace quant::detail {

/// Sliding-window rolling mean into a caller-owned buffer. ``out.size()``
/// must equal ``data.size()``; first (window - 1) slots are NaN.
inline void rolling_mean(
    std::span<const double> data,
    int window,
    std::span<double> out) noexcept
{
    const auto n = static_cast<int>(data.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill(out.begin(), out.end(), nan);

    if (n < window || window < 1) {
        return;
    }

    double sum = 0.0;
    for (int i = 0; i < window; ++i) {
        sum += data[i];
    }
    out[window - 1] = sum / window;

    for (int i = window; i < n; ++i) {
        sum += data[i] - data[i - window];
        out[i] = sum / window;
    }
}

/// Sliding-window rolling mean.
/// First (window - 1) values are NaN.
[[nodiscard]] inline std::vector<double> rolling_mean(
    std::span<const double> data,
    int window)
{
    std::vector<double> result(data.size());
    rolling_mean(data, window, result);
    return result;
}

/// Rolling mean and sample std in one sliding pass. Caller-owned output
/// buffers must each have size data.size(); the first (window - 1) slots
/// are filled with NaN. Uses the sum-based mean recurrence of rolling_mean
/// and the Welford m2 recurrence of rolling_std so the two outputs are
/// bit-identical to those helpers in fp64.
inline void rolling_mean_std(
    std::span<const double> data,
    int window,
    std::span<double> mean_out,
    std::span<double> std_out,
    int ddof = 1) noexcept
{
    const auto n = static_cast<int>(data.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();

    if (n < window || window < 1 || window <= ddof) {
        std::fill(mean_out.begin(), mean_out.end(), nan);
        std::fill(std_out.begin(), std_out.end(), nan);
        return;
    }

    const auto warmup = static_cast<std::ptrdiff_t>(window - 1);
    std::fill(mean_out.begin(), mean_out.begin() + warmup, nan);
    std::fill(std_out.begin(), std_out.begin() + warmup, nan);

    const double denom = static_cast<double>(window - ddof);

    double sum = 0.0;
    double w_mean = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < window; ++i) {
        sum += data[i];
        double delta = data[i] - w_mean;
        w_mean += delta / (i + 1);
        m2 += delta * (data[i] - w_mean);
    }
    mean_out[window - 1] = sum / window;
    std_out[window - 1] = std::sqrt(std::max(0.0, m2 / denom));

    for (int i = window; i < n; ++i) {
        double old_val = data[i - window];
        double new_val = data[i];
        sum += new_val - old_val;
        double old_wmean = w_mean;
        w_mean += (new_val - old_val) / window;
        m2 += (new_val - old_val) * (new_val - w_mean + old_val - old_wmean);
        mean_out[i] = sum / window;
        std_out[i] = std::sqrt(std::max(0.0, m2 / denom));
    }
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

    // Welford's online algorithm — m2 is sum of squared deviations from the running mean.
    double mean = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < window; ++i) {
        double delta = data[i] - mean;
        mean += delta / (i + 1);
        double delta2 = data[i] - mean;
        m2 += delta * delta2;
    }
    result[window - 1] = std::sqrt(std::max(0.0, m2 / denom));

    // Sliding update: when removing old_val and adding new_val,
    //   new_mean = old_mean + (new_val - old_val) / window
    //   m2     += (new_val - old_val) * (new_val - new_mean + old_val - old_mean)
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
