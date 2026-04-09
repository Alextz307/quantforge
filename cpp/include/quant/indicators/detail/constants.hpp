#pragma once

#include <cmath>

#include "quant/core/types.hpp"

namespace quant::detail {

// sqrt(kTradingDaysPerYear) — not constexpr until C++26, so use inline const.
inline const double kSqrt252 =
    std::sqrt(static_cast<double>(kTradingDaysPerYear));

}  // namespace quant::detail
