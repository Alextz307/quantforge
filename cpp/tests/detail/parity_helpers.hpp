#pragma once

// Deterministic synthetic series + NaN-safe equality check for kernel
// parity tests. Sibling of cpp/benchmarks/detail/random.hpp; consolidate
// both if a single quant::testing/random.hpp is introduced.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "quant/core/types.hpp"

namespace quant::tests::detail {

inline std::vector<double> geometric_random_walk(
    std::size_t n, unsigned seed, double start, double sigma)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> dist(0.0, sigma);
    std::vector<double> v(n);
    if (n == 0) return v;
    v[0] = start;
    for (std::size_t i = 1; i < n; ++i) {
        v[i] = v[i - 1] * (1.0 + dist(gen));
    }
    return v;
}

inline std::vector<double> additive_random_walk(
    std::size_t n, unsigned seed, double start, double sigma)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> dist(0.0, sigma);
    std::vector<double> v(n);
    if (n == 0) return v;
    v[0] = start;
    for (std::size_t i = 1; i < n; ++i) {
        v[i] = v[i - 1] + dist(gen);
    }
    return v;
}

// Synthetic OHLC bars with hourly timestamps. Open is offset slightly below
// close so the backtest engine's fill-at-open / mark-at-close distinction
// is actually exercised; high/low bracket both values. Used by the
// buffer-reuse and slice-view C++ test suites.
inline std::vector<quant::Bar> make_synthetic_bars(
    std::size_t n, unsigned seed, double start, double sigma)
{
    const auto closes = geometric_random_walk(n, seed, start, sigma);
    std::vector<quant::Bar> bars;
    bars.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double c = closes[i];
        const double o = c * 0.999;
        bars.push_back(quant::Bar{
            static_cast<int64_t>(1'700'000'000 + 3600 * static_cast<int64_t>(i)),
            o, c * 1.01, o * 0.99, c, 1000.0});
    }
    return bars;
}

// EXPECT_EQ per element (not EXPECT_NEAR) because the kernels under test
// are designed to produce bit-identical fp64 output against their
// reference; any tolerance would hide a genuine regression.
inline void expect_array_eq(
    const std::vector<double>& a, const std::vector<double>& b)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            EXPECT_EQ(std::isnan(a[i]), std::isnan(b[i])) << "index " << i;
        } else {
            EXPECT_EQ(a[i], b[i]) << "index " << i;
        }
    }
}

}  // namespace quant::tests::detail
