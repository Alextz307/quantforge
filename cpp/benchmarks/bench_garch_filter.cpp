#include <cstddef>

#include <benchmark/benchmark.h>

#include "detail/measure.hpp"
#include "detail/random.hpp"
#include "quant/filters/garch_filter.hpp"

namespace {

constexpr double kOmega = 0.05;
constexpr double kAlpha1 = 0.10;
constexpr double kBeta1 = 0.85;
constexpr double kBackcast = 1.0;
constexpr double kReturnStdDev = 1.0;

void BM_GarchFilter(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto r = quant::benchmark::detail::filled_normal(n, 0.0, kReturnStdDev);
    quant::filters::GarchParams params{kOmega, {kAlpha1}, {kBeta1}, 0.0, kBackcast};
    quant::benchmark::detail::measure(state, [&] {
        benchmark::DoNotOptimize(quant::filters::garch_filter(r, params));
    });
}
BENCHMARK(BM_GarchFilter)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
