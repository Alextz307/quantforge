#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "detail/macd_reference.hpp"
#include "detail/parity_helpers.hpp"
#include "quant/indicators/macd.hpp"

namespace {

using quant::MACD;
using quant::tests::detail::expect_array_eq;
using quant::tests::detail::geometric_random_walk;
using quant::tests::detail::macd_compute_all_reference;
using quant::tests::detail::macd_compute_reference;

constexpr unsigned kSeed = 0xC0FFEEu;
constexpr double kStartPrice = 100.0;
constexpr double kReturnStdDev = 0.01;
constexpr std::size_t kSmallN = 1000;
constexpr std::size_t kLargeN = 10000;
constexpr int kFastPeriod = 12;
constexpr int kSlowPeriod = 26;
constexpr int kSignalPeriod = 9;

std::vector<double> prices(std::size_t n) {
    return geometric_random_walk(n, kSeed, kStartPrice, kReturnStdDev);
}

TEST(MACDFusedParity, ComputeAllMatchesReferenceSmall) {
    const auto p = prices(kSmallN);
    MACD macd(kFastPeriod, kSlowPeriod, kSignalPeriod);
    const auto fused = macd.compute_all(p);
    const auto ref = macd_compute_all_reference(
        p, kFastPeriod, kSlowPeriod, kSignalPeriod);
    expect_array_eq(fused.macd_line, ref.macd_line);
    expect_array_eq(fused.signal_line, ref.signal_line);
    expect_array_eq(fused.histogram, ref.histogram);
}

TEST(MACDFusedParity, ComputeAllMatchesReferenceLarge) {
    const auto p = prices(kLargeN);
    MACD macd(kFastPeriod, kSlowPeriod, kSignalPeriod);
    const auto fused = macd.compute_all(p);
    const auto ref = macd_compute_all_reference(
        p, kFastPeriod, kSlowPeriod, kSignalPeriod);
    expect_array_eq(fused.macd_line, ref.macd_line);
    expect_array_eq(fused.signal_line, ref.signal_line);
    expect_array_eq(fused.histogram, ref.histogram);
}

TEST(MACDFusedParity, ComputeMatchesReferenceLarge) {
    const auto p = prices(kLargeN);
    MACD macd(kFastPeriod, kSlowPeriod, kSignalPeriod);
    const auto fused = macd.compute(p);
    const auto ref = macd_compute_reference(p, kFastPeriod, kSlowPeriod);
    expect_array_eq(fused, ref);
}

TEST(MACDFusedParity, ComputeAllEqualsComputeOnMACDLine) {
    const auto p = prices(kLargeN);
    MACD macd(kFastPeriod, kSlowPeriod, kSignalPeriod);
    const auto full = macd.compute_all(p);
    const auto line_only = macd.compute(p);
    expect_array_eq(line_only, full.macd_line);
}

}  // namespace
