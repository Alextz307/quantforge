# Quant Trading Framework

A thesis-grade, bifurcated C++/Python quantitative trading framework with strict anti-leakage guarantees, temporal contracts, and a clean separation between computation (C++) and orchestration (Python). Built around walk-forward validation, typed interfaces, and end-to-end hyperparameter tuning.

**Current state:** a Python orchestration layer built on top of a C++ core. Implemented and under test: the typed temporal contracts, data layer, ML leaf models (GARCH, ARMA, LSTM, XGBoost), hybrid residual models, five trading strategies, the feature pipeline, a C++ indicator suite (RSI, MACD, Bollinger, Garman-Klass, Parkinson), a GARCH inference filter, two strategy state machines and two full `IStrategy` C++ classes (pairs trading + adaptive bollinger) with a shared `SpreadCalculator` primitive, and the C++ backtest engine + performance metrics — all bridged through a `pybind11` module (`quant_engine`) with the GIL released on every compute call. Every model and strategy round-trips through directory-based `save()` / `load()` (JSON configs + metadata + native binary weights, zero pickle). `WalkForwardValidator` supports an optional `snap_to_day` mode that keeps every train/test boundary on a daily close, honouring the intraday day-boundary rule. An offline `make thesis-demo` target runs the full data → walk-forward → metrics → reporters → cross-strategy comparison → regime split flow on a committed SPY parquet so a fresh checkout can verify the pipeline end-to-end without network access. CI is green on Linux and macOS with **1248 Python tests** (+21 opt-in skips), **222 C++ tests**, `mypy --strict` clean on the full Python tree, and `ruff` clean across the whole repo.

## Architecture

```mermaid
graph TB
    subgraph py["Python orchestration"]
        direction TB
        data["Data layer<br/>yfinance · CSV · parquet · cache"]
        feat["FeatureEngineeringPipeline<br/>(RSI + MACD delegate to C++)"]
        leaf["Leaf models<br/>GARCH · ARMA · LSTM · XGBoost<br/>(GARCH inference uses C++ filter)"]
        hyb["Hybrid models<br/>GARCH+LSTM · ARMA+LSTM"]
        strat["Strategies (5)<br/>Bollinger · Pairs · Momentum · VolTgt · RetFcst<br/>(Bollinger + Pairs rule-path in C++)"]
        persist["save / load<br/>directory-based · JSON + .pt + .ubj"]
        wforch["WalkForwardValidator<br/>+ evaluate_walk_forward orchestrator<br/>+ snap_to_day (day-boundary splits)"]
        hpo["Optuna HPO<br/>suggest_params on every model + strategy"]
        data --> feat --> leaf --> hyb --> strat
        strat --> wforch
        strat --> persist
        leaf --> persist
        hyb --> persist
        strat -. tuned by .-> hpo
    end

    subgraph bridge["pybind11 bridge"]
        numpy["numpy ↔ std::span · zero-copy<br/>GIL released on every compute call"]
    end

    subgraph cpp["C++ engine"]
        direction TB
        indicators["Indicators<br/>RSI · MACD · Bollinger"]
        voles["Volatility estimators<br/>Garman-Klass · Parkinson"]
        filters["Filters<br/>garch_filter (recursive σ²)"]
        smach["Strategy state machines<br/>mean-reversion · pairs z-score"]
        spreadc["SpreadCalculator<br/>spread · rolling z-score"]
        cppstrat["IStrategy classes<br/>PairsTrading · AdaptiveBollinger"]
        backtest["Backtest engine<br/>order state machine · fills · slippage"]
        metrics["Metrics<br/>Sharpe · Sortino · drawdown · Calmar"]
        indicators --> backtest
        voles --> backtest
        spreadc --> cppstrat
        smach --> cppstrat
        backtest --> metrics
    end

    wforch --> numpy
    numpy --> backtest
    numpy --> indicators
    numpy --> voles
    numpy --> filters
    numpy --> smach
    numpy --> spreadc
    numpy --> cppstrat
    metrics --> numpy
    numpy --> hpo

    style py fill:#1a1a2e,color:#e0e0ff,stroke:#4a90d9
    style bridge fill:#f39c12,color:#1a1a1a,stroke:#e67e22
    style cpp fill:#0d1117,color:#c9d1d9,stroke:#58a6ff
```

Anything that runs inside the backtest hot loop (bar iteration, indicators, metrics) lives in C++ with `std::span`-based zero-copy interfaces. Anything that benefits from Python's ecosystem (pandas, PyTorch, XGBoost, Optuna) stays in Python. The bridge is crossed once per batch — numpy arrays go in as contiguous C-order buffers and results come back the same way.

## Design Principles

- **Anti-leakage by construction.** No `.bfill()`, no `.fillna(0)`. Fit-once guards on scalers (a second `fit()` raises `LeakageError`), frozen params after `fit()` on GARCH and ARMA, `TrainingMetadata` populated on every model and checked at runtime by the backtest engine via `validate_no_overlap()`, and an intraday day-boundary rule so that even on hourly bars the training cutoff is always a daily close (enforced by `WalkForwardValidator(snap_to_day=True)` for intraday folds).
- **Temporal contracts.** `TemporalSplit`, `TemporalTripleSplit`, and `WalkForwardValidator` enforce train-then-test ordering with embargo gaps. The holdout set is reserved for final thesis evaluation and is never touched during development or HPO.
- **Strict typing.** `mypy --strict` across `src/`, `tests/`, and `scripts/`. No `Any` at internal boundaries. `**kwargs: object` rather than `**kwargs: Any`. Public APIs use pure `Enum` types — no `Enum | str` weak unions. CI enforces this on every push.
- **Performance, measured.** C++ uses `std::span<const double>` interfaces, SoA layouts, and Welford's algorithm for rolling mean/std fused in one pass. Every hot-path indicator, metric, engine, and strategy exposes both an allocating convenience overload and a buffer-reuse (`out`-param / `Buffer&`) overload so HPO inner loops reuse scratch across scenarios. `TimeSeries::slice_view` returns a non-owning `std::span` for zero-copy walk-forward splitting. Pybind11 bindings emit zero-copy numpy views over C++-owned buffers via pybind11 `py::capsule` and `handle base` ownership — no `memcpy` at the Python↔C++ boundary. Release builds compile with `-O3 -march=native -flto`; rolling kernels are `noexcept` to unblock inlining + vectorization. C++ benchmarks emit wall time alongside `Cycles` / `CyclesPerItem` (and `Instructions` / `IPC` on PMU-capable hosts) so optimization is driven by measurement, not intuition.
- **Registry-driven composition.** Every model, data source, and strategy registers via a decorator, which will let a future config loader instantiate an entire pipeline from a YAML file.
- **Drift guards over review vigilance.** Two sources of truth that must stay aligned (pyproject deps ↔ CI pip install, composite dataclass fields ↔ leaf ctor signature, Python `Interval` constants ↔ C++ `kTradingDaysPerYear`) get an automated stdlib-only script in `scripts/` plus a pytest, wired into the CI lint job as an early step.

## Model Composition

Every strategy is a composition of typed, independently-tested building blocks. Leaf models are swappable; a C++ port of any leaf automatically benefits every composite that depends on it.

```mermaid
graph LR
    subgraph leaves["Leaf Models"]
        GARCH["GARCHPredictor"]
        ARMA["ARMAPredictor"]
        LSTM["LSTMPredictor"]
        XGB["DirectionalClassifier"]
    end
    subgraph blocks["Composite Blocks"]
        HV["HybridVolatilityModel<br/>GARCH + LSTM residual"]
        HR["HybridReturnModel<br/>ARMA + LSTM residual"]
        FP["FeatureEngineeringPipeline"]
        CT["CointegrationTester"]
    end
    subgraph strategies["Strategies (IStrategy)"]
        AB["AdaptiveBollinger<br/>(-1 / 0 / +1)"]
        PT["PairsTrading<br/>(-1 / 0 / +1)"]
        MG["MomentumGatekeeper<br/>(0 / 1)"]
        VT["VolatilityTargeting<br/>[0, max_lev]"]
        RF["ReturnForecast<br/>[-lev, +lev]"]
    end
    GARCH --> HV
    LSTM --> HV
    ARMA --> HR
    LSTM --> HR
    GARCH --> AB
    CT --> PT
    FP --> MG
    XGB --> MG
    HV --> VT
    HR --> RF
```

## Training and Backtest Flow

```mermaid
sequenceDiagram
    autonumber
    participant User as user script / tuner
    participant Split as TemporalTripleSplit
    participant WF as evaluate_walk_forward
    participant Strat as IStrategy
    participant Meta as TrainingMetadata
    participant Eng as BacktestEngine · C++
    participant Metr as MetricsCalculator · C++

    User->>Split: split(df, val_pct, holdout_pct, gap)
    Split-->>User: {train, validation, holdout}

    User->>WF: evaluate_walk_forward(strategy, df, validator, engine, slippage)
    loop per fold
        WF->>Strat: train(fold.train)
        Strat->>Strat: fit leaf models · fit scalers (fit-once guard)
        Strat->>Meta: TrainingMetadata.from_fit(...)
        Strat-->>WF: _fitted = True (transactional)

        WF->>Meta: training_metadata.validate_no_overlap(fold.test)
        Note over WF,Meta: raises LeakageError on overlap
        WF->>Strat: generate_signals(fold.test)
        Strat-->>WF: signals  (NaN during warmup)

        WF->>Eng: run(bars, signals, slippage)
        Eng->>Eng: shift signals t → t+1 · fill · accumulate equity
        Eng-->>WF: BacktestResult (equity_curve, total_return, trade_count)

        WF->>Metr: compute(equity_curve, annualization, rf)
        Metr-->>WF: PerformanceMetrics (Sharpe · Sortino · drawdown · win rate)
    end
    WF-->>User: list[FoldResult]
```

The holdout split is reserved for the final thesis evaluation — it is never touched during development or HPO. `TrainingMetadata.validate_no_overlap()` is a runtime tripwire that lives in the orchestrator (`evaluate_walk_forward`), not in the engine itself: `engine.run()` is a pure number cruncher and does not inspect training metadata. Direct callers of `engine.run()` are responsible for their own data hygiene — the orchestrator is the recommended entry point precisely because it wires the tripwire in for free.

## Orchestration flow

The orchestration layer turns a validated YAML config into a fully-wired
`Experiment`, drives the walk-forward, and routes results to the
matching reporter. Three CLI subcommands compose the full surface — one
config feeds `experiment run`, N configs feed `experiment compare`, and
a saved run + a regime detector feed `experiment regime`.

```mermaid
graph LR
    Config["config/*.yaml<br/>(ExperimentConfig)"]
    Builder["build_experiment<br/>resolves registries<br/>+ injects pretrained leaves"]
    Experiment["Experiment<br/>(data_source · strategy · validator · engine)"]
    WalkForward["evaluate_walk_forward<br/>+ deep-metadata tripwire"]
    RunCLI["scripts.experiment.run<br/>artefacts → runs/"]
    CompareCLI["scripts.experiment.compare<br/>N configs → comparisons/"]
    RegimeCLI["scripts.experiment.regime<br/>1 run + detector → regime_reports/"]
    StrategyReporter["StrategyReporter<br/>(equity · stability · LaTeX)"]
    ComparisonReporter["ComparisonReporter<br/>(ranking · pairwise CIs)"]
    RegimeReporter["RegimeReporter<br/>(heatmap · timeline)"]

    Config --> Builder --> Experiment --> WalkForward
    WalkForward --> RunCLI --> StrategyReporter
    Config -->|N×| CompareCLI --> ComparisonReporter
    RunCLI --> RegimeCLI --> RegimeReporter
```

`Experiment` is a frozen bundle — every component is resolved once via
the global registries (`data_source_registry`, `strategy_registry`,
`feature_registry`) so the same YAML configures both ad-hoc runs and
HPO trials. Pretrained-leaf artefacts (`experiment train-model`) are
loaded at build time and threaded into the strategy ctor; the deep
metadata tripwire then enforces strict no-overlap between every leaf's
training window and each fold's test window.

## What's Implemented

### C++ engine (`cpp/`)
- **Core types.** `Bar`, `BarSoA`, `Signal`, `BacktestResult`, `Interval` enum with annualization factors, tagged series for train/test provenance.
- **Indicator framework.** `IIndicator` for single-array inputs and `IVolatilityEstimator` for OHLC four-span inputs. Multi-output indicators expose both a fast-path `compute()` returning the primary output and a richer `compute_all()` returning a result struct.
- **Indicators.** RSI (Wilder smoothing), MACD (EMA fast/slow/signal + histogram), Bollinger Bands (SMA ± k·σ with Welford rolling std).
- **Volatility estimators.** Garman-Klass and Parkinson, sharing `detail/` helpers for annualized rolling variance and OHLC length validation.
- **Backtest engine.** Bar-iteration loop with t→t+1 fill convention, position carry-forward, commission on turnover notional, NaN-signal-as-flat semantics, and an `allow_short` toggle. Slippage is pluggable: `NoSlippage`, `Fixed` (bps), and `VolumeScaled` (bps + volume-impact coefficient).
- **Performance metrics.** `MetricsCalculator` computes Sharpe, Sortino (downside), max drawdown, Calmar, win rate, annualized return + volatility from an equity curve. Single-pass Welford for mean/std; degenerate inputs return 0 rather than NaN.
- **GARCH inference filter.** `quant::filters::garch_filter(scaled_returns, GarchParams)` runs the recursive σ² recurrence (`sigma²[t] = ω + Σ αᵢ·(r-μ)² + Σ βⱼ·σ²`) with backcast substitution and a variance floor. Called by `GARCHPredictor.predict()` — the `arch`-library fit loop stays in Python, only inference moves to C++.
- **Strategy state machines.** `run_mean_reversion_state_machine(close, mid, upper, lower, trend_ma)` and `run_pairs_state_machine(zscore, entry, exit, stop)` — bar-by-bar position carry with NaN skipping, returned as numpy arrays.
- **`SpreadCalculator` primitive.** `compute_spread(a, b, hedge_ratio)` and `compute_zscore(spread, window)` (Welford rolling, NaN on leading warmup and zero-variance windows). Consumed by `PairsTradingStrategy`.
- **Full `IStrategy` C++ classes.** `PairsTradingStrategy` fuses `SpreadCalculator` + pairs state machine behind a keyword-ctor `Config`; `AdaptiveBollingerStrategy` fuses rolling mid/trend + mean-reversion state machine. Both release the GIL on every `generate_signals`. Momentum, ReturnForecast, and VolatilityTargeting remain Python-native — their signal logic is dominated by ML inference, so C++ ports give no measurable speedup.
- **222 GoogleTest cases** covering correctness, slippage variants, fill convention, filter recurrence, state-machine transitions, spread + rolling z-score parity, C++ strategy classes, buffer-reuse overload parity, `slice_view` pointer-identity, fused-pass parity against pre-fusion references (MACD EMAs, Bollinger rolling mean+std, metrics Welford, spread Welford z-score), and numerical edge cases; builds on Linux and macOS through the CI matrix.

### Bridge + Python engine layer (`src/quant_engine/`, `src/engine/`)
- **pybind11 module `quant_engine`.** Exposes `BacktestEngine`, `MetricsCalculator`, `SlippageConfig`, `SlippageModel`, `BacktestResult`, `PerformanceMetrics`, the five indicators (`RSI`, `MACD` + `MACDResult`, `BollingerBands` + `BollingerResult`, `Parkinson`, `GarmanKlass`), the `GarchParams` struct + `garch_filter` free function, the two state machines (`run_mean_reversion_state_machine`, `run_pairs_state_machine`), `SpreadCalculator` + `CointegrationParams`, and the two full strategy classes (`PairsTradingStrategy` + its `Config`, `AdaptiveBollingerStrategy` + its `Config`). Every compute method declares `py::call_guard<py::gil_scoped_release>()` so Python-side parallelism (Optuna HPO, pytest-xdist) can actually scale. Stubs are checked in (`src/quant_engine/quant_engine.pyi`) so `mypy --strict` sees the binding.
- **`CppBacktestEngine` adapter.** Implements `IBacktestEngine`. Validates the pandas-shaped contract (DatetimeIndex, OHLCV columns present, signals index aligned with bars index) before dispatching to the binding. Supports `run_scenarios` for single-pass scenario sweeps.
- **`SLIPPAGE_SCENARIOS`.** Predefined `SlippageConfig` constants keyed by `SlippageScenario` (`ZERO` / `NORMAL` / `HIGH` / `EXTREME`).
- **`evaluate_walk_forward` orchestrator.** Loops over `WalkForwardValidator` folds, retrains the strategy per fold, runs `validate_no_overlap()` as a runtime tripwire, and returns a list of `FoldResult` carrying both the raw `BacktestResult` and the `PerformanceMetrics`.

### Python ML layer (`src/`)
- **Leaf models.** `GARCHPredictor` (AIC grid search, params frozen post-fit, inference loop delegates to C++ `garch_filter`), `ARMAPredictor` (`pmdarima.auto_arima`, order and coefficients frozen; on reload reconstructed as a statsmodels `ARIMA` with the fitted order so `pmdarima` is a fit-time tool only), `MarketLSTM` + `LSTMPredictor` (configurable loss, temporal 80/20 validation split, early stopping, device auto-select), `DirectionalClassifier` (XGBoost binary direction). Every leaf implements `save(path)` / `load(path)` round-trip (JSON config + metadata, native binary weights for torch / XGBoost).
- **Hybrid residual models.** `HybridVolatilityModel` (GARCH + LSTM residual correction → conditional variance) and `HybridReturnModel` (ARMA + LSTM residual correction → conditional mean). Strict black-box composition — the leaves' anti-leakage guarantees are preserved at the composite level for free. `save` / `load` recurse into each leaf under `<root>/{garch,arma,lstm}/` subdirectories plus a root `scaler.json`.
- **Feature pipeline.** `FeatureEngineeringPipeline` produces log returns, RSI, MACD (+ signal and histogram), rolling volatility, MA ratio, and short/long return features. RSI and MACD delegate to the `quant_engine` bindings (Wilder smoothing for RSI, single-pass EMA fast/slow/signal for MACD). Every period is a ctor parameter and appears in `suggest_params`.
- **Cointegration.** `CointegrationTester` implements the Engle-Granger two-step procedure with hedge ratio and spread statistics.
- **Strategies.** All implement `IStrategy` with `train()` + `generate_signals()` + `save()` + `load()` + `suggest_params()`:
  - `AdaptiveBollingerStrategy` — mean-reversion bands scaled by GARCH forecast volatility, gated by a trend filter; the whole rule path (rolling mid, trend MA, position carry) runs in C++ via `quant_engine.AdaptiveBollingerStrategy`.
  - `PairsTradingStrategy` — Engle-Granger cointegrated spread z-score with configurable entry, exit, and stop-loss thresholds; the whole rule path (spread, rolling z-score, state machine) runs in C++ via `quant_engine.PairsTradingStrategy` with cached `CointegrationParams`.
  - `MomentumGatekeeperStrategy` — XGBoost directional classifier on the feature pipeline output, gated by a trend filter.
  - `VolatilityTargetingStrategy` — hybrid volatility forecast driving continuous leverage, with bearish-regime attenuation. Realized-vol training target is the annualized Garman-Klass OHLC estimator computed via `quant_engine.GarmanKlass`.
  - `ReturnForecastStrategy` — hybrid return forecast driving a bounded continuous position.
  Strategy persistence delegates to the owned leaves (e.g. `<root>/classifier/` for Momentum, `<root>/hybrid_vol/` for VolatilityTargeting) with a root `config.json` + `metadata.json`.

### Infrastructure
- **Device selection** (`src/core/device.py`). Auto-picks CUDA > MPS > CPU for PyTorch and CUDA > CPU for XGBoost (MPS is explicitly rejected). Every model accepts `device: Device | None`. On `load()`, device is re-resolved against the current environment — `.pt` weights are loaded with `map_location` set to the host device rather than trusting the stored device string.
- **Temporal infrastructure.** `TemporalSplit`, `TemporalTripleSplit`, `WalkForwardValidator`, `TrainingMetadata` with `from_fit()` / `to_dict()` / `from_dict()` and runtime overlap validation. `WalkForwardValidator` accepts `snap_to_day: bool = False`; when true, every fold's `train_end` snaps back to a day close and `test_start` is pushed forward by `gap` **trading days** (not bars), so intraday walk-forward folds never straddle a day boundary.
- **Persistence layout** (`src/core/persistence.py`). Directory-based: `<root>/config.json`, `<root>/metadata.json` (from `TrainingMetadata.to_dict()`), and model-specific weight files — `weights.json` for GARCH and ARMA, `weights.pt` for LSTM, `model.ubj` for XGBoost, `scaler.json` for `StandardScaler`. No pickle, no joblib. `ensure_model_dir` refuses non-empty targets so a stale directory never silently shadows a fresh save.
- **Data layer.** `CSVSource`, `DataNormalizer` (handles both yfinance and polygon column conventions), `DataCache`, and a `validate_bars` ingestion-time quality check (NaN, non-positive prices, OHLC ordering, duplicate timestamps) that runs once per fetch before the cache write so bad data never reaches the strategies or the C++ engine.
- **Registries.** `model_registry`, `classifier_registry`, `strategy_registry`, `data_source_registry`.

## Getting Started

### Prerequisites

- **C++:** CMake 3.20+ and a C++20 compiler (Clang 15+, GCC 12+, or Apple Clang 14+).
- **Python:** 3.12 or newer.
- **macOS only:** `brew install libomp` (XGBoost wheels need OpenMP runtime).

### Clone and install

```bash
git clone git@github.com:Alextz307/quantforge.git
cd quantforge

# Python package in editable mode, plus dev tools (mypy, ruff, pytest)
pip install -e ".[dev]"

# C++ build — CMake FetchContent pulls GoogleTest, Google Benchmark, and pybind11
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Debug
cmake --build cpp/build -j
```

### Run the tests

```bash
make test           # Full gate: C++ ctest + pytest + mypy strict
make test-cpp       # GoogleTest suite
make test-python    # pytest suite
make typecheck      # mypy --strict src/ tests/ scripts/
make lint           # ruff check + ruff format --check
make bench-cpp      # Google Benchmark indicator + engine + metrics + filter + state-machine + spread + C++ strategy micro-benches
make thesis-demo    # End-to-end pipeline smoke on cached SPY (offline, ~20s)

# Optional: verify each C++ path still beats its Python baseline
PERF_GUARD=1 pytest tests/benchmarks/    # opt-in; CI does not gate on timing
```

### Benchmarking

Every C++ micro-benchmark emits wall time alongside `Cycles` / `CyclesPerItem` custom counters (sourced from `__rdtsc` on x86, `CNTVCT_EL0` on arm64, `steady_clock` elsewhere). The Python orchestrator subprocesses `quant_bench --benchmark_format=json`, parses it into `BenchmarkResult` dataclasses, persists under `benchmark_results/runs/`, and drives the comparator / reporter.

```mermaid
graph LR
    cpp["quant_bench<br/>(Google Benchmark)"]
    cc["CycleCounter<br/>rdtsc · CNTVCT · steady_clock"]
    runner["BenchmarkRunner<br/>subprocesses + parses JSON"]
    store["BenchmarkStore<br/>JSONL runs + baselines"]
    analyzer["BenchmarkAnalyzer<br/>z-test · scaling fits"]
    reporter["BenchmarkReporter<br/>LaTeX · plots"]
    cc -. cycle counters .-> cpp
    cpp --> runner --> store
    store --> analyzer --> reporter
```

```bash
make bench                                                   # build + run; dumps JSONL under benchmark_results/runs/
make bench-baseline NAME=my-baseline                         # capture a named baseline (NAME required)
python -m scripts.benchmark run --save-baseline my-baseline  # same, via the CLI directly
python -m scripts.benchmark compare pre-optimization <run>   # regression gate (z-test + pct delta)
python -m scripts.benchmark latex                            # LaTeX summary table of the newest run
```

Runs and reports are gitignored; `benchmark_results/baselines/` is tracked so the thesis has a reproducible anchor across machines.

### Minimal example — fit a strategy and generate signals

```python
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_close_df

train = make_synthetic_close_df(n_rows=500)
eval_df = make_synthetic_close_df(n_rows=100, start="2021-01-04", seed=99)

strategy = AdaptiveBollingerStrategy(window=20, k=2.0, trend_window=100)
strategy.train(train)
signals = strategy.generate_signals(eval_df)      # pd.Series in {-1, 0, +1}

strategy.save("/tmp/ab_model")                    # metadata + config + GARCH subdir
reloaded = AdaptiveBollingerStrategy.load("/tmp/ab_model")
# reloaded.generate_signals(eval_df) is bit-identical to strategy.generate_signals(eval_df)
```

Every strategy exposes the same four-verb API — `train(data)`, `generate_signals(data)`, `save(path)`, `load(path)` — plus a static `suggest_params(trial)` so Optuna can tune the entire stack (feature periods, model hyperparameters, and strategy thresholds) end to end.

### End-to-end demo

The `make thesis-demo` target runs the full Python orchestration stack
on a committed SPY parquet fixture (`tests/fixtures/SPY.parquet`) so a
fresh checkout can verify every wire connects without network access.
Three CLI invocations land back-to-back: `experiment run` (single
walk-forward), `experiment compare` (cross-strategy ranking), and
`experiment regime` (per-regime split). Total wall time on a 2024
laptop is well under a minute.

```bash
make thesis-demo
```

> ⚠️ **The demo's output is illustrative — not a benchmark and not an
> empirical claim.** Strategies are not tuned, the walk-forward window
> is short (~7 years of daily SPY across 4 expanding folds), and only
> one regime detector is exercised. The comprehensive empirical study
> will land separately under `experiment_results/studies/`.

A curated subset of one demo run is committed under
[`experiment_results/thesis_demo/sample/`](experiment_results/thesis_demo/README.md)
so a casual reader sees the shape of the output without needing to run
anything. Two of the figures — the walk-forward equity curves and the
regime-vs-metric heatmap — give the quickest read on what the pipeline
actually produces:

![Per-fold equity curves](experiment_results/thesis_demo/sample/plots/run_equity_curves.png)

![Per-regime metric heatmap](experiment_results/thesis_demo/sample/plots/regime_metric_heatmap.png)

The full `sample/` index (every committed plot + LaTeX table + the
aggregated metrics JSON) lives in
[`experiment_results/thesis_demo/README.md`](experiment_results/thesis_demo/README.md).

## Project Structure

```
cpp/
  include/quant/
    core/                Bar, TimeSeries, Interval, tagged series
    indicators/          IIndicator, IVolatilityEstimator, RSI, MACD, Bollinger, GK, Parkinson
    indicators/detail/   Shared helpers (Welford rolling, annualization)
    filters/             garch_filter (GARCH inference σ² recurrence)
    statistics/          SpreadCalculator (spread + rolling z-score, shared by pairs)
    strategies/          IStrategy mixin, state machines, PairsTradingStrategy, AdaptiveBollingerStrategy
    engine/              SlippageConfig, BacktestEngine
    metrics/             MetricsCalculator, PerformanceMetrics
  src/                   Implementation files
  bindings/              pybind11 module entry point (python_module.cpp)
  tests/                 GoogleTest suite
  benchmarks/            Google Benchmark micro-benches
  benchmarks/detail/     Shared bench helpers (seeded RNG, cycle-counter measure wrapper)

src/
  core/                  Types, constants, temporal contracts, registry, device selection, exceptions, persistence helpers, config schema
  data/                  Sources (yfinance, CSV, parquet), normalizer, cache, loader, fingerprint
  features/              FeatureEngineeringPipeline
  models/                GARCH, ARMA, LSTM, XGBoost classifier, hybrids, cointegration, dataset (each with save / load)
  strategies/            Five strategies + IStrategy interface (each with save / load)
  engine/                CppBacktestEngine adapter, slippage scenarios, walk-forward orchestrator
  quant_engine/          pybind11 module re-exports + checked-in mypy stubs
  orchestration/         Builder, Experiment + RunOptions, comparison, regime-run, manifest, model-artifact, standalone training
  optimization/          Optuna StrategyTuner + samplers / pruners / objectives + checkpointing
  analysis/              Fold aggregator, ranking, regime split, paired-bootstrap significance
  visualization/         Strategy / Comparison / Regime / HPO reporters (plots + booktabs LaTeX)
  benchmarking/          Runner + store + analyzer + reporter + comparator

tests/
  unit/                  One unit-test file per component
  integration/           pybind11 module load + engine/indicator/filter/state-machine binding parity + walk-forward orchestrator
  benchmarks/            Opt-in perf guards (PERF_GUARD=1) verifying C++ still beats Python baselines
  fixtures/              Committed offline fixtures (e.g. SPY.parquet for `make thesis-demo`)
  conftest.py            Shared fixtures (synthetic data, global seeds)

scripts/                 experiment + benchmark CLIs + stdlib-only drift guards
config/                  Strategy / HPO / regime / universe YAMLs + thesis-demo entry config
benchmark_results/
  baselines/             Tracked JSONL anchors (pre-/post-optimization)
  runs/                  Per-run JSONL (gitignored)
  reports/               LaTeX + plots (gitignored)
experiment_results/
  thesis_demo/           Tracked: README + curated `sample/` from one demo run
  thesis_demo/runs/      Fresh per-`make thesis-demo` outputs (gitignored)
  runs/, comparisons/, regime_reports/, hpo/, models/  Ephemeral per-developer artefacts (gitignored)
.github/workflows/ci.yml Lint, typecheck, C++ matrix, Python matrix
Makefile                 Canonical build/test entry points
pyproject.toml           Python deps + scikit-build-core config
mypy.ini                 Strict settings + per-module ignore_missing_imports
```

### Subsystem navigation

Each subsystem ships its own `README.md` — purpose, public surface,
layout table, one runnable snippet, and cross-links. Use these as
navigation aids; function signatures and detailed docstrings live in the
code.

- [`cpp/`](cpp/README.md) — C++20 engine: indicators, filters, state machines, backtest engine, metrics, pybind11 module.
- [`src/orchestration/`](src/orchestration/README.md) — config → wired experiment, walk-forward driver, comparison + regime + holdout pipelines.
- [`src/strategies/`](src/strategies/README.md) — `IStrategy` + the five concrete strategies (incl. pairs).
- [`src/engine/`](src/engine/README.md) — `CppBacktestEngine` adapter + walk-forward orchestrator (single-leg / pairs dispatch).
- [`src/features/`](src/features/README.md) — `FeatureEngineeringPipeline` + fit-once anti-leakage scaler.
- [`src/optimization/`](src/optimization/README.md) — Optuna `StrategyTuner` + samplers / pruners / objectives.
- [`src/data/`](src/data/README.md) — sources, normaliser, cache, fingerprint (single + pair).
- [`src/core/`](src/core/README.md) — types, constants, registry, temporal primitives, persistence layout, exceptions, config.
- [`src/models/`](src/models/README.md) — leaf predictors / classifiers / hybrids / cointegration / dataset.
- [`src/analysis/`](src/analysis/README.md) — fold aggregator, ranking, regime split, significance.
- [`src/visualization/`](src/visualization/README.md) — strategy / comparison / regime / HPO reporters.
- [`src/benchmarking/`](src/benchmarking/README.md) — benchmark runner / store / analyzer / reporter.
- [`scripts/`](scripts/README.md) — `experiment` + `benchmark` CLIs and drift guards.
- [`config/`](config/README.md) — strategy / HPO / regime / model / universe YAMLs.
- [`webapp/`](webapp/README.md) — FastAPI backend + React/Vite SPA: read-only artifact viewer, configurable runner (run + tune), live job + HPO monitors with WebSocket streaming.

## Tech Stack

| Layer     | Technology                                                                                       |
|-----------|--------------------------------------------------------------------------------------------------|
| C++ engine| C++20, CMake 3.20+, GoogleTest, Google Benchmark                                                  |
| Python    | pandas 2.2+, numpy 1.26+, Pydantic v2, PyTorch 2.2+, XGBoost 2.x, arch, statsmodels, pmdarima, scikit-learn, Optuna |
| Bridge    | pybind11 2.12+, scikit-build-core                                                                |
| Quality   | mypy (strict), ruff (check + format), pandas-stubs, ASan/UBSan-ready C++ flags                   |
| CI        | GitHub Actions on an `ubuntu-latest` and `macos-latest` matrix                                   |

## License

This project is part of a university thesis. Not licensed for redistribution.
