# `src/engine/`

Python adapter layer for the C++ `quant_engine` extension plus the
walk-forward orchestrator that wires strategies, validators, engines,
and metrics into per-fold results.

## Public surface

| Symbol | Role |
| --- | --- |
| `IBacktestEngine` | ABC: `run`, `run_scenarios`, `run_pairs`. Returns the C++ `BacktestResult` (equity curve, total return, trade count); statistical metrics are computed separately by `MetricsCalculator`. |
| `CppBacktestEngine` | Pandas → numpy → `quant_engine.BacktestEngine` adapter. Validates index + OHLCV columns at the boundary; defers all hot-loop work to C++. |
| `evaluate_walk_forward(strategy, bars, validator, engine, slippage, ...)` | Per-fold pipeline: optional feature-pipeline fit → `strategy.train` → deep-metadata leakage check → `generate_signals` → engine dispatch → `MetricsCalculator`. Returns `list[FoldResult]`. |
| `FoldResult` | Frozen bundle: fold index, train/test bounds, raw `BacktestResult`, computed `PerformanceMetrics`. |
| `SlippageScenario` (StrEnum) + `COST_SCENARIOS` (dict) | Named cost tiers — `ZERO` 0/0, `LOW` 1/1, `NORMAL` (default) 2/2, `HIGH` 5/5, as (slippage bp / commission bp). `CostScenario` bundles a `SlippageConfig` with `commission_bps`; `commission_fraction_for` converts to the engine's `transaction_fee_rate`. `SLIPPAGE_SCENARIOS` is the derived slippage-only view. |

## Layout

| File | Role |
| --- | --- |
| `interface.py` | `IBacktestEngine` ABC. |
| `cpp_engine.py` | Numpy marshalling + `_validate_bars_columns` / `_bars_to_ohlcv_arrays` helpers; `run` / `run_scenarios` / `run_pairs`. |
| `walk_forward.py` | `evaluate_walk_forward` + `validate_deep_metadata` (composite leaf-aware) + `split_pairs_frame` (wide → two single-leg frames via `PAIRS_LEG_SUFFIXES`). |
| `scenarios.py` | The four named cost tiers (slippage + commission); single source of truth shared by the backtest and the deployment scorecard. |

## Single-leg vs pairs dispatch

`evaluate_walk_forward` branches on `strategy.is_pairs_strategy`:

- **Single-leg** — calls `engine.run(fold.test, signals, slippage)`.
- **Pairs** — splits the wide-format frame on `_a` / `_b` suffixes via
  `split_pairs_frame`, then calls
  `engine.run_pairs(bars_a, bars_b, signals, strategy.hedge_ratio, slippage)`.
  Leg B's leverage is `-hedge_ratio * signals[t]`; cash is shared,
  fills happen per leg.

The wide-format suffix renames are pre-built once at module import as
`_LEG_A_RENAME` / `_LEG_B_RENAME` constants.

## Anti-leakage tripwire

The deep-metadata check inside `evaluate_walk_forward` is the one place
where strategy and leaf training metadata are validated against each
fold's test window. Two invariants:

- **Always:** `train_end < test_start` for every tracked component
  (the strategy and every wrapped leaf via `get_all_training_metadata`).
- **Pretrained-only:** `train_end < train_start` — a frozen-injected
  leaf must not have seen even the fold's train window. Fresh leaves
  legitimately re-fit on the fold train every fold and skip this check.

A `LeakageError` from either invariant names the offending component
(`StrategyClass.origin`).

## Snippet

```python
from src.core.config import load_experiment_config
from src.core.registry import data_source_registry, strategy_registry
from src.core.temporal import WalkForwardValidator
from src.engine.cpp_engine import CppBacktestEngine
from src.engine.scenarios import SLIPPAGE_SCENARIOS, SlippageScenario
from src.engine.walk_forward import evaluate_walk_forward

cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
bars = data_source_registry.create_from_config(cfg.data.source).fetch(
    cfg.data.tickers[0], cfg.data.start, cfg.data.end, cfg.data.interval
)
strategy = strategy_registry.create_from_config(cfg.strategy)
folds = evaluate_walk_forward(
    strategy=strategy,
    bars=bars,
    validator=WalkForwardValidator(n_splits=4, test_size=63, gap=5, expanding=True),
    engine=CppBacktestEngine(),
    slippage=SLIPPAGE_SCENARIOS[SlippageScenario.NORMAL],
    interval=cfg.data.interval,
)
```

In practice, build via `src/orchestration/builder.py::build_experiment`
— direct calls into `evaluate_walk_forward` are mostly for tests that
inject mocks per component.

## Cross-links

- Pairs C++ implementation lives under `cpp/src/engine/backtest_engine.cpp`
  (`run_pairs` two-leg shared-cash state machine).
- Annualization factor (`Interval.annualization_factor`) lives in
  `src/core/types.py`; the same constants exist in `cpp/include/quant/core/types.hpp`.
- `LeakageError` is defined in `src/core/exceptions.py`; the deep-metadata
  collector lives on `IStrategy` in `src/strategies/interface.py`.
