#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace quant::detail {

/// Shared precondition for out-param overloads across indicators, metrics,
/// spread primitives, strategies, and state machines — a single source of
/// truth so the error message stays consistent regardless of which kernel
/// caught the mismatch.
inline void check_out_size(
    std::size_t in_size,
    std::size_t out_size,
    const char* where)
{
    if (in_size != out_size) {
        throw std::invalid_argument(
            std::string(where) + ": output buffer size must equal input size");
    }
}

}  // namespace quant::detail
