#pragma once

#include <cmath>
#include <cstdint>

namespace quant {

inline constexpr double kBpsPerUnit = 10000.0;

enum class SlippageModel : uint8_t {
    NoSlippage,
    Fixed,
    VolumeScaled,
};

struct SlippageConfig {
    SlippageModel model{SlippageModel::Fixed};
    double base_bps{1.0};
    double volume_impact_coeff{0.0};

    /// Compute the actual fill price for an order of `order_qty` shares on a
    /// bar with the given `theoretical_price` (typically the bar's open) and
    /// `bar_volume`. Buys (`order_qty > 0`) pay slippage upward; sells pay
    /// slippage downward. For `VolumeScaled`, extra bps proportional to
    /// |order_qty| / bar_volume (zero impact when bar_volume is non-positive).
    [[nodiscard]] double apply(
        double theoretical_price,
        double order_qty,
        double bar_volume
    ) const noexcept {
        const double sign = (order_qty >= 0.0) ? 1.0 : -1.0;
        double effective_bps = 0.0;
        switch (model) {
            case SlippageModel::NoSlippage:
                return theoretical_price;
            case SlippageModel::Fixed:
                effective_bps = base_bps;
                break;
            case SlippageModel::VolumeScaled: {
                const double impact = (bar_volume > 0.0)
                    ? volume_impact_coeff * std::abs(order_qty) / bar_volume
                    : 0.0;
                effective_bps = base_bps + impact;
                break;
            }
        }
        const double fraction = effective_bps / kBpsPerUnit;
        return theoretical_price * (1.0 + sign * fraction);
    }
};

}  // namespace quant
