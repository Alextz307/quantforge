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

    /// Caller-owned scratch buffers for the fused-generate path. Reusing the
    /// same ``Buffer`` across calls amortizes the four N-element allocations
    /// (mid + trend MA + upper/lower bands) that the composition would
    /// otherwise incur per call.
    struct Buffer {
        std::vector<double> mid;
        std::vector<double> trend_ma;
        std::vector<double> upper;
        std::vector<double> lower;
    };

    explicit AdaptiveBollingerStrategy(Config config);

    /// Produce {-1, 0, +1} position signals. Leading ``max(band_window,
    /// trend_window) - 1`` bars are NaN. ``close`` and ``cond_vol`` must
    /// have the same length; ``cond_vol`` is the per-bar price sigma
    /// (``garch_annual_vol / sqrt(ann_factor) * close``).
    [[nodiscard]] std::vector<double> generate_signals(
        std::span<const double> close,
        std::span<const double> cond_vol) const;

    /// Out-param overload: writes signals into ``out`` (same size as inputs).
    /// Internally allocates a fresh ``Buffer``; callers running in tight
    /// inner loops should pass a reused ``Buffer`` via the five-argument
    /// overload below.
    void generate_signals(
        std::span<const double> close,
        std::span<const double> cond_vol,
        std::span<double> out) const;

    /// Fully reusable overload — writes signals into ``out`` and reuses the
    /// four intermediate band/MA vectors held by ``scratch`` across calls.
    void generate_signals(
        std::span<const double> close,
        std::span<const double> cond_vol,
        std::span<double> out,
        Buffer& scratch) const;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int required_warmup() const noexcept override;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    Config config_;
};

}  // namespace quant::strategies
