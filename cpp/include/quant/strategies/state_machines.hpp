#pragma once

#include <span>
#include <vector>

namespace quant::strategies {

/// Run the mean-reversion Bollinger-band state machine used by
/// `AdaptiveBollingerStrategy`. All five spans must have the same length;
/// positions are {-1, 0, +1} with NaN for bars whose `mid`, `upper`, `lower`,
/// or `trend_ma` is NaN (the previous position is carried, unobserved).
///
/// Entry (flat): long when `close > trend_ma` && `close < lower`;
///                short when `close < trend_ma` && `close > upper`.
/// Exit (long):  `close >= mid` → flat.
/// Exit (short): `close <= mid` → flat.
[[nodiscard]] std::vector<double> run_mean_reversion_state_machine(
    std::span<const double> close,
    std::span<const double> mid,
    std::span<const double> upper,
    std::span<const double> lower,
    std::span<const double> trend_ma);

/// Run the pairs-trading z-score state machine used by
/// `PairsTradingStrategy`. NaN bars in `zscore` hold the previous position and
/// emit NaN. `|z| >= stop_loss_zscore` forces flat. From flat, `z >= entry`
/// opens short (-1) and `z <= -entry` opens long (+1). While in a position,
/// `|z| <= exit` closes it.
[[nodiscard]] std::vector<double> run_pairs_state_machine(
    std::span<const double> zscore,
    double entry_zscore,
    double exit_zscore,
    double stop_loss_zscore);

}  // namespace quant::strategies
