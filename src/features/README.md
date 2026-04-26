# `src/features/`

Feature engineering for the predictive strategies. The pipeline owns
the engineered columns (returns, volatility proxies, RSI, MACD) and
the fit-once `StandardScaler`; the engine drives a fresh pipeline per
fold via a factory closure to keep walk-forward leakage-free.

## Public surface

| Symbol | Role |
| --- | --- |
| `IFeaturePipeline` | ABC: `fit`, `transform`, default `fit_transform` (calls fit then transform). |
| `FeatureEngineeringPipeline` | Sole concrete pipeline, registered as `"standard"` on `feature_registry`. Computes 9 engineered features from `close`; scales them with a fit-once `StandardScaler`. |
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
| `vol_{vol_window}` | rolling std of `return_1d` (default 20) |
| `ma_ratio` | `close / SMA(close, ma_ratio_window)` |
| `rsi_{rsi_period}` | C++ `RSI(period).compute(close)` (Wilder's smoothing, default 14) |
| `macd`, `macd_signal`, `macd_hist` | C++ `MACD(fast, slow, signal).compute_all(close)` (default 12/26/9) |

Leading NaN from warmup is preserved — never `.bfill()` or `.fillna(0)`.

## `keep_ohlc` flag

Some strategies (`ReturnForecastStrategy`, `VolatilityTargetingStrategy`)
need both raw price (for log-returns or Garman-Klass volatility) and
engineered features in the same frame at `train()` time. Passing
`keep_ohlc=True` on the pipeline ctor concatenates the source frame's
OHLCV columns un-scaled alongside the scaled engineered features.

## Anti-leakage discipline

- **Fit-once.** `fit` calls `guard_scaler_fit_once` before fitting; a
  second `fit` on the same instance raises `LeakageError`. The
  walk-forward orchestrator never reuses an instance — it calls a
  zero-arg factory to get a fresh pipeline per fold and fits on
  `fold.train` only.
- **Transform-time alignment.** Transform applies the fitted scaler to
  non-NaN rows in place; warmup rows stay NaN.
- **Persistence round-trip.** When a strategy persists its pipeline
  (e.g. `MomentumGatekeeperStrategy`), the scaler is round-tripped via
  the public `scaler` property — no reaching into `_scaler`.

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

- Numerical kernels (RSI, MACD) live in `cpp/src/indicators/` and are
  exposed through `quant_engine`; Python-side rolling primitives are
  thin wrappers over those bindings.
- Registered on `feature_registry` (`src/core/registry.py`) for
  config-driven instantiation; `build_experiment` wires the factory.
- `LeakageError` and `guard_scaler_fit_once` come from
  `src/core/exceptions.py`.
