#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "quant/core/types.hpp"
#include "quant/metrics/performance.hpp"

namespace {

// ───── Bench data constants ─────
constexpr int kSeed = 42;
constexpr double kInitialEquity = 10'000.0;
constexpr double kEquityFloor = 0.01;
constexpr double kReturnMean = 0.0003;   // ~7.6% annualized drift
constexpr double kReturnStd = 0.012;     // ~19% annualized vol

[[nodiscard]] std::vector<double> generate_equity_curve(size_t n) {
    std::mt19937 gen(kSeed);
    std::normal_distribution<double> dist(kReturnMean, kReturnStd);
    std::vector<double> equity(n);
    equity[0] = kInitialEquity;
    for (size_t i = 1; i < n; ++i) {
        equity[i] = std::max(kEquityFloor, equity[i - 1] * (1.0 + dist(gen)));
    }
    return equity;
}

// `compute` includes `equity_to_returns` inside the timed loop; the individual
// metric benches below exclude it (they pre-convert and reuse the returns
// vector). The breakdowns therefore won't sum to the `compute` time — that's
// by design, each bench measures its own primitive in isolation.
void BM_MetricsCalculator_Compute(benchmark::State& state) {
    const auto equity = generate_equity_curve(static_cast<size_t>(state.range(0)));
    const int af = quant::annualization_factor(quant::Interval::Daily);
    for (auto _ : state) {
        benchmark::DoNotOptimize(quant::MetricsCalculator::compute(equity, af));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MetricsCalculator_Compute)
    ->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_MetricsCalculator_MaxDrawdown(benchmark::State& state) {
    const auto equity = generate_equity_curve(static_cast<size_t>(state.range(0)));
    for (auto _ : state) {
        benchmark::DoNotOptimize(quant::MetricsCalculator::max_drawdown(equity));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MetricsCalculator_MaxDrawdown)
    ->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_MetricsCalculator_Sharpe(benchmark::State& state) {
    const auto equity = generate_equity_curve(static_cast<size_t>(state.range(0)));
    const auto returns = quant::MetricsCalculator::equity_to_returns(equity);
    const int af = quant::annualization_factor(quant::Interval::Daily);
    for (auto _ : state) {
        benchmark::DoNotOptimize(quant::MetricsCalculator::sharpe_ratio(returns, af));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MetricsCalculator_Sharpe)
    ->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_MetricsCalculator_Sortino(benchmark::State& state) {
    const auto equity = generate_equity_curve(static_cast<size_t>(state.range(0)));
    const auto returns = quant::MetricsCalculator::equity_to_returns(equity);
    const int af = quant::annualization_factor(quant::Interval::Daily);
    for (auto _ : state) {
        benchmark::DoNotOptimize(quant::MetricsCalculator::sortino_ratio(returns, af));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MetricsCalculator_Sortino)
    ->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
