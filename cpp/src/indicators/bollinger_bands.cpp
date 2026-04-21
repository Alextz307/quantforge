#include "quant/indicators/bollinger_bands.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "quant/indicators/detail/rolling.hpp"

namespace quant {

namespace {

// Strip `std::to_string(double)`'s trailing zeros for display names.
// Keeps one zero after the decimal (`2.000000` → `2.0`) so the value
// is still visibly a float; non-round values lose only the padding
// (`2.500000` → `2.5`).
std::string format_double(double x) {
    std::string s = std::to_string(x);
    const auto dot = s.find('.');
    if (dot == std::string::npos) return s;
    const auto last_nonzero = s.find_last_not_of('0');
    if (last_nonzero == dot) {
        s.erase(dot + 2);
    } else {
        s.erase(last_nonzero + 1);
    }
    return s;
}

}  // namespace

BollingerBands::BollingerBands(int period, double num_std)
    : period_(period)
    , num_std_(num_std)
{
    if (period < 1) {
        throw std::invalid_argument("BollingerBands period must be >= 1, got "
                                    + std::to_string(period));
    }
    if (num_std < 0.0) {
        throw std::invalid_argument("BollingerBands num_std must be >= 0");
    }
}

// Keep the Welford recurrence below in sync with detail::rolling_mean_std;
// inlined here so mid/upper/lower are emitted in the same sliding pass
// with no std-deviation temp.
BollingerResult BollingerBands::compute_all(std::span<const double> prices) const {
    const auto n = prices.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    BollingerResult result;
    result.mid.resize(n, nan);
    result.upper.resize(n, nan);
    result.lower.resize(n, nan);

    if (static_cast<int>(n) < period_) {
        return result;
    }

    const double denom = static_cast<double>(period_ - 1);
    double sum = 0.0;
    double w_mean = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < period_; ++i) {
        sum += prices[i];
        const double delta = prices[i] - w_mean;
        w_mean += delta / (i + 1);
        m2 += delta * (prices[i] - w_mean);
    }
    {
        const double mean = sum / period_;
        const double sd = std::sqrt(std::max(0.0, m2 / denom));
        result.mid[period_ - 1] = mean;
        result.upper[period_ - 1] = mean + num_std_ * sd;
        result.lower[period_ - 1] = mean - num_std_ * sd;
    }

    for (std::size_t i = static_cast<std::size_t>(period_); i < n; ++i) {
        const double old_val = prices[i - period_];
        const double new_val = prices[i];
        sum += new_val - old_val;
        const double old_wmean = w_mean;
        w_mean += (new_val - old_val) / period_;
        m2 += (new_val - old_val) * (new_val - w_mean + old_val - old_wmean);
        const double mean = sum / period_;
        const double sd = std::sqrt(std::max(0.0, m2 / denom));
        result.mid[i] = mean;
        result.upper[i] = mean + num_std_ * sd;
        result.lower[i] = mean - num_std_ * sd;
    }

    return result;
}

std::vector<double> BollingerBands::compute(std::span<const double> prices) const {
    return detail::rolling_mean(prices, period_);
}

int BollingerBands::warmup_period() const noexcept {
    return period_ - 1;
}

std::string BollingerBands::name() const {
    return "BB(" + std::to_string(period_) + "," + format_double(num_std_) + ")";
}

}  // namespace quant
