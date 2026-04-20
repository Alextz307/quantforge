#pragma once

#include <span>
#include <string>
#include <vector>

#include "quant/strategies/interface.hpp"

namespace quant::strategies {

/// Mean-reversion Bollinger-band strategy with GARCH-adaptive band widths.
///
/// Fuses ``rolling_mean`` (for the band midpoint and the longer-window
/// trend MA) plus ``run_mean_reversion_state_machine`` into a single C++
/// call. The GARCH fit (and its translation from annualized volatility to
/// a per-bar price sigma) stays in Python; this class consumes the
/// already-computed per-bar price sigma via ``cond_vol`` and only owns the
/// inference-path band construction + state machine.
class AdaptiveBollingerStrategy final : public IStrategy {
public:
    struct Config {
        int band_window = 20;
        double k = 2.0;
        int trend_window = 100;
    };

    explicit AdaptiveBollingerStrategy(Config config);

    /// Produce {-1, 0, +1} position signals. Leading ``max(band_window,
    /// trend_window) - 1`` bars are NaN. ``close`` and ``cond_vol`` must
    /// have the same length; ``cond_vol`` is the per-bar price sigma
    /// (``garch_annual_vol / sqrt(ann_factor) * close``).
    [[nodiscard]] std::vector<double> generate_signals(
        std::span<const double> close,
        std::span<const double> cond_vol) const;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int required_warmup() const noexcept override;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    Config config_;
};

}  // namespace quant::strategies
