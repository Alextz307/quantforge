#pragma once

#include <span>
#include <vector>

namespace quant::statistics {

/// Frozen output of a cointegration test. ``hedge_ratio`` is the OLS slope
/// of ``a`` regressed on ``b``; ``spread_mean`` / ``spread_std`` are the
/// training-window spread moments (retained for provenance - the rolling
/// z-score recomputes them on the inference window).
struct CointegrationParams {
    double hedge_ratio;
    double spread_mean;
    double spread_std;
};

/// Spread + rolling z-score primitives for pairs-trading.
class SpreadCalculator {
public:
    /// ``a - hedge_ratio * b`` element-wise. Requires equal-length spans.
    [[nodiscard]] static std::vector<double> compute_spread(
        std::span<const double> a,
        std::span<const double> b,
        double hedge_ratio);

    /// Out-param overload: writes into caller-owned ``out`` (same size as ``a``).
    static void compute_spread(
        std::span<const double> a,
        std::span<const double> b,
        double hedge_ratio,
        std::span<double> out);

    /// Rolling z-score over ``spread`` with a trailing window of
    /// ``window`` bars. Leading ``window - 1`` outputs are NaN; bars
    /// whose rolling std is 0 also emit NaN (matches pandas'
    /// ``.rolling(w).std()`` behavior on constant windows).
    ///
    /// NaN semantics: the underlying Welford accumulator is poisoned by
    /// any NaN in ``spread``. Once a NaN enters the window, every
    /// subsequent output is NaN - unlike pandas' ``rolling(w).std()``
    /// which recovers when the NaN slides out. Callers whose spread may
    /// contain NaN must pre-clean it.
    [[nodiscard]] static std::vector<double> compute_zscore(
        std::span<const double> spread,
        int window);

    /// Out-param overload: writes into caller-owned ``out`` (same size as ``spread``).
    static void compute_zscore(
        std::span<const double> spread,
        int window,
        std::span<double> out);
};

}  // namespace quant::statistics
