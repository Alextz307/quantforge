# `src/features/`

Feature engineering for the predictive strategies. The pipeline owns
the engineered columns (returns, momentum, volatility, range/gap,
trend, volume) and the fit-once `StandardScaler`; the engine drives a
fresh pipeline per fold via a factory closure to keep walk-forward
leakage-free.

## Public surface

| Symbol | Role |
| --- | --- |
| `IFeaturePipeline` | ABC: `fit`, `transform`, default `fit_transform` (calls fit then transform). |
| `FeatureEngineeringPipeline` | Sole concrete pipeline, registered as `"standard"` on `feature_registry`. Computes 17 engineered features from full OHLCV; scales them with a fit-once `StandardScaler`. Raises `ValueError` if any OHLCV column is missing, or if no row survives feature warmup (frame shorter than the longest warmup, or a degenerate column such as constant volume). |
| `FeatureEngineeringPipeline.scaler` | Property: read the fitted scaler (or `None` before `fit`); setter is used by `load()` paths. |
| `FeatureEngineeringPipeline.hard_nan_warmup_bars` | Number of leading NaN bars across all features (max of every per-feature warmup window). Strategies use this to align their `required_warmup_bars`. |
| `FeatureEngineeringPipeline(keep_ohlc=...)` | When `True`, `transform` concatenates raw OHLCV (un-scaled) alongside scaled engineered features. |

## Layout

| File | Role |
| --- | --- |
| `interface.py` | `IFeaturePipeline` ABC. |
| `pipeline.py` | `FeatureEngineeringPipeline` + helper `_compute_rsi` / `_compute_macd` (thin wrappers around `quant_engine` C++ bindings). |

## Engineered columns

| Column | Formula |
| --- | --- |
| `return_1d` | `close.pct_change(1)` |
| `return_{short}d` | `close.pct_change(short_return_period)` (default 5) |
| `return_{long}d` | `close.pct_change(long_return_period)` (default 21) |
| `roc_{roc_period}` | `close.pct_change(roc_period)` (quarter-horizon momentum, default 63) |
| `vol_{vol_window}` | rolling std of `return_1d` (default 20) |
| `garman_klass` | C++ `GarmanKlass(window).compute(o,h,l,c)` (range-based annualized vol, default window 20) |
| `intraday_range` | `(high - low) / close` |
| `overnight_gap` | `open / close.shift(1) - 1` |
| `ma_ratio` | `close / SMA(close, ma_ratio_window)` |
| `bb_pctb` | C++ `BollingerBands(period, num_std).compute_all(close)` -> `(close - lower) / (upper - lower)` (default 20/2.0) |
| `rsi_{rsi_period}` | C++ `RSI(period).compute(close)` (Wilder's smoothing, default 14) |
| `macd`, `macd_signal`, `macd_hist` | C++ `MACD(fast, slow, signal).compute_all(close)` (default 12/26/9) |
| `adx_{adx_period}` | Wilder ADX via `ewm(adjust=False)` (trend strength, default 14; first `2*period-1` bars NaN-masked) |
| `volume_zscore` | `(volume - roll_mean) / roll_std` over `volume_zscore_window` (default 20) |
| `obv_z` | rolling z-score of on-balance volume `sign(close.diff()) * volume).cumsum()` (default window 20) |

Leading NaN from warmup is preserved, never `.bfill()` or `.fillna(0)`.
Every feature reads only bars `<= t` (the engine shifts the signal
`t -> t+1`); a parametrized causality test asserts prefix-vs-full
equality per column.

## `keep_ohlc` flag

Some strategies (`ReturnForecastStrategy`, `VolatilityTargetingStrategy`)
need both raw price (for log-returns or Garman-Klass volatility) and
engineered features in the same frame at `train()` time. Passing
`keep_ohlc=True` on the pipeline ctor concatenates the source frame's
OHLCV columns un-scaled alongside the scaled engineered features.

## Anti-leakage discipline

- **Fit-once.** `fit` calls `guard_scaler_fit_once` before fitting; a
  second `fit` on the same instance raises `LeakageError`. The
  walk-forward orchestrator never reuses an instance; it calls a
  zero-arg factory to get a fresh pipeline per fold and fits on
  `fold.train` only.
- **Transform-time alignment.** Transform applies the fitted scaler to
  non-NaN rows in place; warmup rows stay NaN.
- **Persistence round-trip.** When a strategy persists its pipeline
  (e.g. `MomentumGatekeeperStrategy`), the scaler is round-tripped via
  the public `scaler` property, never reaching into `_scaler`.

## Snippet

```python
from datetime import datetime

from src.data.loader import YFinanceSource
from src.features.pipeline import FeatureEngineeringPipeline

bars = YFinanceSource().fetch("SPY", datetime(2020, 1, 1), datetime(2024, 12, 31))
pipeline = FeatureEngineeringPipeline(keep_ohlc=True)
holdout = 63
train_features = pipeline.fit_transform(bars.iloc[:-holdout])
test_features = pipeline.transform(bars.iloc[-holdout:])
print(train_features.columns.tolist())
```

## Cross-links

- Numerical kernels (RSI, MACD, Garman-Klass, Bollinger Bands) live in
  `cpp/src/indicators/` and are exposed through `quant_engine`;
  Python-side rolling primitives are thin wrappers over those bindings.
  ADX and the volume features stay pure-pandas (`ewm(adjust=False)`
  runs the Wilder recursion at C speed; features compute once per fold,
  never inside the backtest hot loop).
- Registered on `feature_registry` (`src/core/registry.py`) for
  config-driven instantiation; `build_experiment` wires the factory.
- `LeakageError` and `guard_scaler_fit_once` come from
  `src/core/exceptions.py`.
