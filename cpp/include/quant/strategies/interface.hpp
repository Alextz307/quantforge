#pragma once

#include <string>

namespace quant::strategies {

/// Abstract base for C++ strategy classes.
///
/// Strategies' signal-generation signatures differ too much for a single
/// virtual ``generate_signals`` (pairs takes two price series plus
/// cointegration params; mean-reversion takes close + adaptive band width;
/// future strategies may take other input packs). ``IStrategy`` is therefore
/// a metadata-only mixin — each concrete class exposes its own
/// ``generate_signals`` method with the appropriate inputs, and the base
/// only guarantees ``name()`` and ``required_warmup()``.
class IStrategy {
public:
    virtual ~IStrategy() = default;

    /// Human-readable strategy name (e.g., ``"PairsTrading"``).
    [[nodiscard]] virtual std::string name() const = 0;

    /// Window size that drives warmup — the first ``required_warmup() - 1``
    /// bars of any ``generate_signals`` output are NaN (the first valid
    /// output lives at index ``required_warmup() - 1``).
    [[nodiscard]] virtual int required_warmup() const noexcept = 0;
};

}  // namespace quant::strategies
