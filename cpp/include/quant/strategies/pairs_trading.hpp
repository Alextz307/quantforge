#pragma once

#include <span>
#include <string>
#include <vector>

#include "quant/statistics/spread.hpp"
#include "quant/strategies/interface.hpp"

namespace quant::strategies {

/// Cointegration-based pairs-trading strategy.
///
/// Fuses ``SpreadCalculator::compute_spread`` + ``compute_zscore`` +
/// ``run_pairs_state_machine`` into a single C++ call. The fit-time work
/// (Engle-Granger cointegration test → ``CointegrationParams``) stays in
/// Python; this class owns only the inference-path rule logic.
class PairsTradingStrategy final : public IStrategy {
public:
    struct Config {
        double entry_zscore = 2.0;
        double exit_zscore = 0.5;
        double stop_loss_zscore = 4.0;
        int zscore_lookback = 60;
    };

    /// Caller-owned scratch buffers for the fused-generate path. Reusing the
    /// same ``Buffer`` across calls amortizes the two N-element allocations
    /// that the spread + rolling-z-score composition would otherwise incur
    /// per call — meaningful under HPO sweeps or walk-forward folds.
    struct Buffer {
        std::vector<double> spread;
        std::vector<double> zscore;
    };

    explicit PairsTradingStrategy(Config config);

    /// Produce {-1, 0, +1} leg-a positions. Leading ``zscore_lookback - 1``
    /// bars are NaN; bars where the rolling std is zero are also NaN.
    /// ``prices_a`` and ``prices_b`` must have the same length.
    [[nodiscard]] std::vector<double> generate_signals(
        std::span<const double> prices_a,
        std::span<const double> prices_b,
        const statistics::CointegrationParams& coint) const;

    /// Out-param overload: writes signals into ``out`` (same size as prices).
    /// Internally allocates a fresh ``Buffer`` per call; callers that run
    /// in tight inner loops should pass a reused ``Buffer`` via the
    /// five-argument overload below.
    void generate_signals(
        std::span<const double> prices_a,
        std::span<const double> prices_b,
        const statistics::CointegrationParams& coint,
        std::span<double> out) const;

    /// Fully reusable overload — writes signals into ``out`` and reuses the
    /// two intermediate vectors held by ``scratch`` across calls.
    void generate_signals(
        std::span<const double> prices_a,
        std::span<const double> prices_b,
        const statistics::CointegrationParams& coint,
        std::span<double> out,
        Buffer& scratch) const;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int required_warmup() const noexcept override;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    Config config_;
};

}  // namespace quant::strategies
