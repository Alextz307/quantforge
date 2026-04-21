#pragma once

// Deterministic synthetic series + NaN-safe equality check for kernel
// parity tests. Sibling of cpp/benchmarks/detail/random.hpp; consolidate
// both if a single quant::testing/random.hpp is introduced.

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

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
