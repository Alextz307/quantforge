#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace quant::benchmark::detail {

inline constexpr std::uint_fast32_t kDefaultSeed = 42;

[[nodiscard]] inline std::mt19937 seeded_rng(std::uint_fast32_t seed = kDefaultSeed) {
    return std::mt19937(seed);
}

[[nodiscard]] inline std::vector<double> filled_normal(
    std::size_t n,
    double mean,
    double std,
    std::uint_fast32_t seed = kDefaultSeed) {
    auto gen = seeded_rng(seed);
    std::normal_distribution<double> dist(mean, std);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

[[nodiscard]] inline std::vector<double> additive_random_walk(
    std::size_t n,
    double start,
    double std,
    std::uint_fast32_t seed = kDefaultSeed) {
    auto gen = seeded_rng(seed);
    std::normal_distribution<double> dist(0.0, std);
    std::vector<double> v(n);
    double p = start;
    for (auto& x : v) {
        p += dist(gen);
        x = p;
    }
    return v;
}

}  // namespace quant::benchmark::detail
