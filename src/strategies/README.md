# `src/strategies/`

Concrete trading strategies. Each one trains its own (composite) model
and emits position signals; the engine shifts signals by one bar and
runs the backtest.

## Public surface

| Symbol | Role |
| --- | --- |
| `IStrategy` | Abstract base. Required: `train`, `generate_signals`, `name`, `required_warmup_bars`, `suggest_params`. Default: `save`/`load` (raise), `hedge_ratio` (raise unless overridden), `get_all_training_metadata`, `_assert_fitted_with_metadata` (read-side guard), `_set_fitted_with_metadata` (atomic write-side commit). |
| `IStrategy.is_pairs_strategy` | `ClassVar[bool]` — `True` only on pairs strategies. The walk-forward dispatcher branches on this to call `engine.run` vs `engine.run_pairs`. |
| `IStrategy.is_multi_feature_strategy` | `ClassVar[bool]` — `True` for single-asset traded strategies that read N feature tickers from a wide `<ohlcv>_<TICKER>` frame. Mutually exclusive with `is_pairs_strategy`; the dispatcher slices the primary asset's OHLCV before calling `engine.run`. |
| `IStrategy.primary_ticker` | Property — the asset a multi-feature strategy trades; default raises (override required). |
| `AdaptiveBollingerStrategy` | Mean-reversion Bollinger bands with GARCH-scaled widths + SMA trend filter. |
| `PairsTradingStrategy` | Cointegration (Engle-Granger) + rolling z-score on the spread. The only `is_pairs_strategy = True` member. |
| `MomentumGatekeeperStrategy` | Long-only momentum gated by a 200-MA trend filter and an XGBoost `DirectionalClassifier`. |
| `CrossAssetMomentumStrategy` | Single-asset traded; XGBoost `DirectionalClassifier` over lagged returns of N feature tickers. The `is_multi_feature_strategy = True` exemplar. Methodology adapted from Rapach et al. (2019, *JFE* 135). |
| `ReturnForecastStrategy` | Sign-of-forecast positions from a `HybridReturnModel` (ARMA + LSTM residual correction). |
| `VolatilityTargetingStrategy` | Position size = `target_vol / forecast_vol` from a `HybridVolatilityModel` (GARCH + LSTM residual correction). |

All six register themselves at import time on `strategy_registry`
(name → class) for config-driven instantiation.

## Layout

| File | Role |
| --- | --- |
| `interface.py` | `IStrategy` ABC + the `is_pairs_strategy` capability flag + the deep-metadata collector. |
| `adaptive_bollinger.py` | Single-asset, owns a `GARCHPredictor`. No pretrained leaves. |
| `pairs_trading.py` | Two-asset (`close_a` / `close_b`); owns `CointegrationTester` and a C++ `PairsTradingStrategy` for signal generation. |
| `momentum_gatekeeper.py` | Owns a `FeatureEngineeringPipeline` + `DirectionalClassifier` (XGBoost). |
| `cross_asset_momentum.py` | Owns a `DirectionalClassifier` (XGBoost) fed lagged log-returns of N feature tickers; reads the wide `<ohlcv>_<TICKER>` frame directly. |
| `return_forecast.py` | Owns a `HybridReturnModel` rebuilt from a frozen `_HybridReturnParams` bundle each `train()`. |
| `volatility_targeting.py` | Owns a `HybridVolatilityModel` rebuilt from a frozen `_HybridVolParams` bundle each `train()`. |

## Patterns

- **Capability flag dispatch.** Pairs strategies set
  `is_pairs_strategy: ClassVar[bool] = True` on the class — no string
  check by name anywhere. `build_experiment` and `walk_forward` read
  the flag to decide between single-leg and two-leg paths.
- **Pretrained-leaf injection.** Composite strategies declare
  `_leaf_keys: ClassVar[frozenset[str]]`; non-composite strategies
  declare an empty frozenset. The ctor's `pretrained_leaves` kwarg is
  validated by `normalize_pretrained_leaves` (extra / missing keys
  raise). Per-leaf shape (interval, feature columns, lookback) is
  validated by `validate_pretrained_leaf`.
- **Composite passthrough bundle.** When a strategy owns a leaf with a
  fit-once scaler and >5 ctor kwargs, the leaf is rebuilt at the top
  of `train()` from a module-private frozen `@dataclass` (e.g.,
  `_HybridReturnParams`, `_HybridVolParams`, `_MomentumConfig`). Tests
  call `assert_params_match_constructor` to drift-guard the bundle
  fields against the leaf's ctor signature.
- **`training_metadata` is the transactional commit.** `train()` and
  `load()` end with `self._set_fitted_with_metadata(metadata)`; that
  helper is the only legal mutator of the slot, refuses ``None``, and is
  the single fitted-state signal — there is no separate boolean flag, so
  `training_metadata is not None` IS "fitted." Read-side guards call
  `self._assert_fitted_with_metadata()` (the caller name is auto-derived
  from the calling frame); composites layer leaf-presence checks
  (`_classifier is None`, `_cpp_coint is None`) as separate statements.
- **No signal shift inside the strategy.** The engine shifts
  positions to `t+1`. Strategies that compute `next_bar_direction`-style
  targets must drop the trailing row, never `fillna(0)` it.
- **`suggest_params` is static.** Each strategy declares its own
  Optuna search space — leaf hyperparameters that pass through to a
  wrapped model (e.g. `arma_p_max` on `ReturnForecast`) are flattened
  into the strategy's space, not resolved separately.

## Snippet

```python
from datetime import datetime

from src.data.loader import YFinanceSource
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy

bars = YFinanceSource().fetch("SPY", datetime(2020, 1, 1), datetime(2024, 12, 31))
holdout = 63
strat = AdaptiveBollingerStrategy(window=20, k=2.0, trend_window=100)
strat.train(bars.iloc[:-holdout])
positions = strat.generate_signals(bars.iloc[-holdout:])  # {-1, 0, +1}
```

## Cross-links

- Inherits the leakage discipline from `src/core/temporal.py`
  (`TrainingMetadata`, deep-metadata `TrackedMetadata`).
- Composes models from `src/models/` (single ownership: the strategy
  owns its leaf instances and calls public API only).
- Consumes engineered features from `src/features/` (where applicable).
- Persistence helpers live in `src/core/persistence.py`
  (`save_model_skeleton`, scaler / classifier subdirs, JSON keys).
