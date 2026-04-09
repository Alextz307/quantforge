# Quant Trading Framework

A high-performance, bifurcated C++/Python quantitative trading framework built for academic research. Designed with strict anti-leakage guarantees and a clear separation between computation (C++) and orchestration (Python).

## Architecture

```
Python (~40%)                         C++ (~60%)
Orchestration, I/O, ML bindings      Pure computation, hot loops
                                      
  Data loading (yfinance, CSV)          Indicators (RSI, MACD, Bollinger, GK, Parkinson)
  Config (YAML + Pydantic)              Backtesting engine (planned)
  ML models (PyTorch, XGBoost)          Performance metrics (planned)
  HPO (Optuna)                          Walk-forward splitting (planned)
  Visualization                         Signal generation (planned)
                                      
         ┌──── pybind11 bridge ────┐
         │  numpy ↔ std::span      │
         │  zero-copy transfers    │
         └─────────────────────────┘
```

**Design principles:**
- Anti-leakage by construction (temporal contracts, train/test tagging, embargo gaps)
- C++20 with `-Wall -Wextra -Wpedantic -Werror`
- Python with `mypy --strict`, no `Any` at internal boundaries
- Structure of Arrays for cache-friendly computation
- `std::span<const double>` for zero-copy array passing

## Project Structure

```
cpp/
  include/quant/core/       # Bar, TimeSeries, Interval, tagged series
  include/quant/indicators/  # IIndicator, IVolatilityEstimator, RSI, MACD, etc.
  src/                       # Implementation files
  tests/                     # Google Test suite
  benchmarks/                # Google Benchmark suite

src/
  core/        # Types, constants, contracts, registry, temporal validation
  data/        # Data sources, normalization, caching
  models/      # ML interfaces, TemporalDataset
  strategies/  # Strategy interfaces
  engine/      # Backtest engine interfaces

tests/
  unit/        # Python unit tests
```

## Prerequisites

- **C++**: CMake 3.20+, C++20 compiler (Clang 15+ or GCC 12+)
- **Python**: 3.12+, pip

## Setup

```bash
# Install Python package in dev mode
pip install -e ".[dev]"

# Build C++ (fetches GoogleTest, Google Benchmark, pybind11 via CMake)
cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Testing

```bash
# All tests (C++ + Python + mypy)
make test

# C++ only (100 tests)
cd cpp/build && ctest --output-on-failure

# Python only (106 tests)
python -m pytest tests/ -v

# Type checking
mypy --strict src/ tests/

# Benchmarks
cd cpp/build && ./quant_bench --benchmark_format=console
```

## Current Status

| Component | Status |
|-----------|--------|
| Python foundation (types, contracts, data layer, registry) | Complete |
| C++ core types (Bar, TimeSeries, Interval) | Complete |
| C++ indicators (RSI, MACD, Bollinger, Garman-Klass, Parkinson) | Complete |
| C++ backtesting engine | Planned |
| pybind11 bindings | Planned |
| Strategy signal generation | Planned |
| Benchmarking suite | Planned |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| C++ Engine | C++20, CMake, Google Test, Google Benchmark |
| Python | Pydantic v2, pandas, PyTorch, scikit-learn |
| Bridge | pybind11, scikit-build-core |
| Quality | mypy strict, ruff, ASan/UBSan, `-Werror` |

## License

This project is part of a university thesis and is not licensed for redistribution.
