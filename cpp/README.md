# `cpp/`

C++20 engine for the framework. Owns everything that runs inside the
backtest hot loop: indicator math, rolling-window primitives, the GARCH
inference filter, strategy state machines, the backtest engine itself,
and the performance-metrics calculator. A single pybind11 module
(`quant_engine`) surfaces it to Python and releases the GIL on every
compute call, so Python-side orchestration (Optuna trials, walk-forward
folds) can run the kernels in parallel.

## Public surface

The only intended Python entry point is `quant_engine` (built from
[`bindings/python_module.cpp`](bindings/python_module.cpp)). The
checked-in `.pyi` stubs and the Python-facing wrapper docs live under
[`src/quant_engine/README.md`](../src/quant_engine/README.md).

Direct C++ consumers (tests, future native callers) include
headers from [`include/quant/`](include/quant/). The include tree is
the canonical public surface and mirrors the implementation tree under
[`src/`](src/) one-for-one.

## Layout

| Path | Role |
| --- | --- |
| `CMakeLists.txt` | Top-level build script. C++20, `-O3 -march=native -flto` in Release; `-fsanitize=address,undefined` in Debug; `-Wall -Wextra -Werror`. PGO targets (`PGO_INSTRUMENT`, `PGO_OPTIMIZE`) reserved for Phase 6 work. |
| `include/quant/core/` | `types.hpp`, `constants.hpp`, `span` helpers, `TimeSeries` view (incl. `slice_view` zero-copy splitting). |
| `include/quant/indicators/` | `IIndicator` + `IVolatilityEstimator` interfaces; RSI, MACD, Bollinger, Garman-Klass, Parkinson; `detail/` helpers (Welford rolling mean/std, `annualize_rolling_variance`, `validate_ohlc_lengths`). |
| `include/quant/filters/` | `garch_filter` (recursive sigma^2 inference; params frozen post-fit on the Python side). |
| `include/quant/statistics/` | `SpreadCalculator` (`spread.hpp`): spread + rolling z-score, consumed by `PairsTradingStrategy`. |
| `include/quant/strategies/` | `IStrategy` ABC + `state_machines.{mean_reversion,pairs}`, full `PairsTradingStrategy` + `AdaptiveBollingerStrategy` C++ classes. |
| `include/quant/engine/` | `BacktestEngine` (order state machine | slippage | fills | equity curve). |
| `include/quant/metrics/` | `MetricsCalculator` (Sharpe / Sortino / Calmar / max-DD / win rate, Welford-fused). |
| `src/<subsystem>/` | Implementation `.cpp` for each `include/quant/<subsystem>/` header. One-to-one. |
| `bindings/python_module.cpp` | The single pybind11 module - re-exports every header above. Releases the GIL on every compute call, accepts numpy arrays via `py::array_t<double, py::array::c_style>` mapped to `std::span<const double>`. |
| `tests/` | GoogleTest suite. `CMakeLists.txt` registers binaries via `gtest_discover_tests()` - `enable_testing()` is required at top-level for `ctest` to see them. Per-test fixtures under `tests/fixtures/`; shared parity helpers under `tests/detail/`. |
| `build/` | Gitignored. CMake build tree; FetchContent caches GoogleTest here. |

## Conventions

- **Two indicator interfaces.** `IIndicator` consumes one `span<const double>` (RSI, MACD, Bollinger). `IVolatilityEstimator` consumes four OHLC spans (Garman-Klass, Parkinson). Don't force them together.
- **Multi-output indicators.** `compute()` returns the primary output; `compute_all()` returns a result struct (`MACDResult`, `BollingerResult`). `compute()` is a dedicated fast path, not a wrapper around `compute_all()`, which would re-allocate the discarded fields.
- **Numerical stability.** Rolling std uses Welford. Naive sum-of-squares is forbidden because it loses precision to catastrophic cancellation. Shared in `indicators/detail/`.
- **No virtual dispatch in the bar-iteration loop.** State machines and the backtest engine use CRTP / static polymorphism. Virtuals are fine outside the hot loop.
- **No `malloc` / `new` in the hot loop.** Pre-allocate vectors at construction; reuse buffers across bars.
- **Constants live in [`include/quant/core/types.hpp`](include/quant/core/types.hpp).** `kTradingDaysPerYear`, `kUSMinutesPerDay`, and the rest, referenced by both C++ code and the Python `src/core/constants.py` shim.

## Build and test

```bash
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Debug   # or Release
cmake --build cpp/build -j
cd cpp/build && ctest --output-on-failure             # GoogleTest suite
```

The Python `make test-cpp` target wraps the same commands. The pybind11
extension is built via `pip install -e .` (scikit-build-core invokes
CMake under the hood with the same flags) so Python tests can
`import quant_engine`.

## Cross-links

- [`src/quant_engine/`](../src/quant_engine/) - Python-side wrapper module + `.pyi` stubs.
- [`src/engine/`](../src/engine/) - `CppBacktestEngine` adapter that calls into `BacktestEngine` from Python.
