// Minimal translation unit for the quant_core static library.
// Core types are header-only; this file ensures the library target has at least
// one source file, avoiding CMake warnings on some generators.

#include "quant/core/types.hpp"

namespace quant {

// Intentionally empty — all core types are defined in headers.

}  // namespace quant
