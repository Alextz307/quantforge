#include <cstddef>

#include <benchmark/benchmark.h>

#include "detail/random.hpp"
#include "quant/filters/garch_filter.hpp"

namespace {

constexpr double kOmega = 0.05;
constexpr double kAlpha1 = 0.10;
constexpr double kBeta1 = 0.85;
constexpr double kBackcast = 1.0;
constexpr double kReturnStdDev = 1.0;

void BM_GarchFilter(benchmark::State& state) {
    const auto r = quant::bench::detail::filled_normal(
        static_cast<std::size_t>(state.range(0)), 0.0, kReturnStdDev);
    quant::filters::GarchParams params{kOmega, {kAlpha1}, {kBeta1}, 0.0, kBackcast};
    for (auto _ : state) {
        benchmark::DoNotOptimize(quant::filters::garch_filter(r, params));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_GarchFilter)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
