#include "quant/filters/garch_filter.hpp"

#include <algorithm>
#include <cstddef>

namespace quant::filters {

std::vector<double> garch_filter(
    std::span<const double> scaled_returns,
    const GarchParams& params)
{
    const auto n = scaled_returns.size();
    std::vector<double> sigma2(n);
    const auto p = params.alpha.size();
    const auto q = params.beta.size();

    for (std::size_t t = 0; t < n; ++t) {
        double var_t = params.omega;

        for (std::size_t i = 0; i < p; ++i) {
            double e2;
            if (t >= i + 1) {
                const double e = scaled_returns[t - i - 1] - params.mu;
                e2 = e * e;
            } else {
                e2 = params.backcast;
            }
            var_t += params.alpha[i] * e2;
        }

        for (std::size_t j = 0; j < q; ++j) {
            const double past_sigma2 =
                (t >= j + 1) ? sigma2[t - j - 1] : params.backcast;
            var_t += params.beta[j] * past_sigma2;
        }

        sigma2[t] = std::max(var_t, kVarianceFloor);
    }

    return sigma2;
}

}  // namespace quant::filters
