# `src/strategies/`

Concrete trading strategies. Each one trains its own (composite) model
and emits position signals; the engine shifts signals by one bar and
runs the backtest.

## Public surface

| Symbol | Role |
| --- | --- |
| `IStrategy` | Abstract base. Required: `train`, `generate_signals`, `name`, `required_warmup_bars`, `suggest_params`. Default: `save`/`load` (raise), `hedge_ratio` (raise unless overridden), `get_all_training_metadata`, `_assert_fitted_with_metadata` (read-side guard), `_set_fitted_with_metadata` (atomic write-side commit), and the feature-importance hooks (`feature_columns`, `feature_importance_frame`, `feature_importance_score`, `feature_gain`) - all default to "skip" so rule-based strategies opt out for free. |
| `IStrategy.is_pairs_strategy` | `ClassVar[bool]` - `True` only on pairs strategies. The walk-forward dispatcher branches on this to call `engine.run` vs `engine.run_pairs`. |
| `IStrategy.is_multi_feature_strategy` | `ClassVar[bool]` - `True` for single-asset traded strategies that read N feature tickers from a wide `<ohlcv>_<TICKER>` frame. Mutually exclusive with `is_pairs_strategy`; the dispatcher slices the primary asset's OHLCV before calling `engine.run`. |
| `IStrategy.primary_ticker` | Property - the asset a multi-feature strategy trades; default raises (override required). |
| `AdaptiveBollingerStrategy` | Mean-reversion Bollinger bands with GARCH-scaled widths + SMA trend filter. |
| `PairsTradingStrategy` | Cointegration (Engle-Granger) + rolling z-score on the spread. The only `is_pairs_strategy = True` member. |
| `MomentumGatekeeperStrategy` | Long-only momentum gated by a 200-MA trend filter and an XGBoost `DirectionalClassifier`. |
| `CrossAssetMomentumStrategy` | Single-asset traded; XGBoost `DirectionalClassifier` over lagged returns of N feature tickers. The `is_multi_feature_strategy = True` exemplar. Methodology adapted from Rapach et al. (2019, *JFE* 135). |
| `ReturnForecastStrategy` | Sign-of-forecast positions from a `HybridReturnModel` (ARMA + LSTM residual correction). |
| `VolatilityTargetingStrategy` | Position size = `target_vol / forecast_vol` from a `HybridVolatilityModel` (GARCH + LSTM residual correction). |

All six register themselves at import time on `strategy_registry`
(name -> class) for config-driven instantiation.

## Layout

| File | Role |
| --- | --- |
| `interface.py` | `IStrategy` ABC + the `is_pairs_strategy` capability flag + the deep-metadata collector. |
| `adaptive_bollinger.py` | Single-asset, owns a `GARCHPredictor`. |
| `pairs_trading.py` | Two-asset (`close_a` / `close_b`); owns `CointegrationTester` and a C++ `PairsTradingStrategy` for signal generation. |
| `momentum_gatekeeper.py` | Owns a `FeatureEngineeringPipeline` + `DirectionalClassifier` (XGBoost). |
| `cross_asset_momentum.py` | Owns a `DirectionalClassifier` (XGBoost) fed lagged log-returns of N feature tickers; reads the wide `<ohlcv>_<TICKER>` frame directly. |
| `return_forecast.py` | Owns a `HybridReturnModel` rebuilt from a frozen `_HybridReturnParams` bundle each `train()`. |
| `volatility_targeting.py` | Owns a `HybridVolatilityModel` rebuilt from a frozen `_HybridVolParams` bundle each `train()`. |

## Patterns

- **Capability flag dispatch.** Pairs strategies set
  `is_pairs_strategy: ClassVar[bool] = True` on the class, with no
  string check by name. `build_experiment` and `walk_forward` read
  the flag to decide between single-leg and two-leg paths.
- **Composite passthrough bundle.** When a strategy owns a leaf with a
  fit-once scaler and >5 ctor kwargs, the leaf is rebuilt at the top
  of `train()` from a module-private frozen `@dataclass` (e.g.,
  `_HybridReturnParams`, `_HybridVolParams`, `_MomentumConfig`). Tests
  call `assert_params_match_constructor` to drift-guard the bundle
  fields against the leaf's ctor signature.
- **`training_metadata` is the transactional commit.** `train()` and
  `load()` end with `self._set_fitted_with_metadata(metadata)`; that
  helper is the only legal mutator of the slot, refuses ``None``, and is
  the single fitted-state signal; there is no separate boolean flag, so
  `training_metadata is not None` means "fitted." Read-side guards call
  `self._assert_fitted_with_metadata()` (the caller name is auto-derived
  from the calling frame); composites layer leaf-presence checks
  (`_classifier is None`, `_cpp_coint is None`) as separate statements.
- **No signal shift inside the strategy.** The engine shifts
  positions to `t+1`. Strategies that compute `next_bar_direction`-style
  targets must drop the trailing row, never `fillna(0)` it.
- **`suggest_params` is static.** Each strategy declares its own
  Optuna search space; leaf hyperparameters that pass through to a
  wrapped model (e.g. `arma_p_max` on `ReturnForecast`) are flattened
  into the strategy's space rather than resolved separately.
- **Feature-importance hooks are opt-in.** A feature-consuming strategy
  overrides `feature_columns()` (the permutable columns),
  `feature_importance_frame(data)` (identity for the hybrids, which
  already receive engineered columns; transform-and-attach-`close` for
  the classifier strategies, which build features internally), and
  `feature_importance_score(frame)` (directional hit-rate for
  return/probability models, negative QLIKE for the volatility
  forecaster). XGBoost-backed strategies also override `feature_gain()`.
  The score derives its realised target only from `close`, so it is
  invariant to permuting any feature. Rule-based strategies inherit the
  defaults and are skipped.

## Adding a new strategy

1. **Pick a shape.** All three shapes inherit from `IStrategy`; the only
   difference is which class flag(s) they set and what frame the engine
   passes to `generate_signals`.

   | Shape | Capability flag | `primary_ticker` | Engine entry | Exemplar |
   | --- | --- | --- | --- | --- |
   | Single-asset (default) | none | not used | `engine.run` (single-leg) | `adaptive_bollinger.py` |
   | Pairs (two-leg) | `is_pairs_strategy = True` | not used | `engine.run_pairs` | `pairs_trading.py` |
   | Multi-feature single-asset | `is_multi_feature_strategy = True` | required override | `engine.run` after `slice_primary_ohlcv` | `cross_asset_momentum.py` |

   The two flags are mutually exclusive; `_validate_strategy_data_shape`
   in `src/orchestration/builder.py` rejects a class that sets both.

2. **Copy `_template.py` -> `<your_strategy>.py`** in this directory and
   drop the leading underscore. The autoloader skips `_`-prefixed
   modules on purpose, so the template never registers itself. Renaming
   makes the autoloader import the module; to actually register the
   class, also uncomment the `@strategy_registry.register("...")`
   decorator at the top of the file.

3. **Implement the five abstract methods** declared on `IStrategy`:
   `train`, `generate_signals`, `name`, `required_warmup_bars`,
   `suggest_params`. The template stubs each one with a
   `NotImplementedError` that explains what goes there. See *Hidden
   contracts* below for the rules each method must respect.

4. **(Pairs / multi-feature only) Set the capability flag and required
   overrides.** Pairs strategies set `is_pairs_strategy: ClassVar[bool]
   = True` and override the `hedge_ratio` property. Multi-feature
   strategies set `is_multi_feature_strategy: ClassVar[bool] = True`
   and override the `primary_ticker` property; the value MUST appear in
   the experiment's `data.tickers` list (validated at config-build time).

5. **Add `config/strategies/<name>.yaml`** with the default ctor kwargs,
   plus a corresponding HPO YAML if the strategy will be tuned. The
   YAML schema is enforced by Pydantic - string values for `Interval`
   and `Device` fields are auto-coerced to the matching `StrEnum`.

6. **Add `tests/unit/test_<your_strategy>.py`.** Minimum coverage:
   train + signals smoke (warmup-NaN check, signal range check),
   `assert_params_match_constructor` drift guard if you used a frozen
   passthrough dataclass, `assert "<Name>" in strategy_registry`,
   `suggest_params` keys match ctor kwargs.

## Hidden contracts

These are invariants the framework relies on but the type system can't
fully express. Violating any of them produces silently-wrong predictions
or surfaces only at backtest time, far from the bug.

- **Atomic fitted-state commit.** `_set_fitted_with_metadata(metadata)`
  is the *only* legal mutator of `_training_metadata`. It refuses
  `None` and must be the last line of `train()` and `load()`. Never
  assign `self._training_metadata = ...` directly. There is no separate
  `_fitted` boolean; `training_metadata is not None` means "fitted".
- **Read-side guard.** Every method that requires a completed `train()`
  (`generate_signals`, `save`, `hedge_ratio`) starts with
  `self._assert_fitted_with_metadata()`. The helper auto-derives the
  caller name from the calling frame for clean tracebacks. Composite
  strategies layer leaf-presence checks (e.g. `if self._classifier is
  None: raise RuntimeError(...)`) as separate statements after the
  metadata guard.
- **Engine shifts signals.** Positions returned at time `t` are
  shifted to `t+1` outside the strategy. Do NOT call `.shift(-1)`
  inside `generate_signals`. Strategies that compute targets like
  `(close[t+1] > close[t])` must drop the trailing row, never
  `fillna(0)` it.
- **`suggest_params` keys <-> ctor kwargs.** The dict returned by
  `suggest_params` has its KEYS consumed as ctor kwargs by
  `StrategyTuner`; mismatched keys surface as `TypeError` at trial-build
  time. The Optuna parameter NAMES (the strings passed to
  `trial.suggest_*`) are global identifiers in a study; prefix-namespace
  them by strategy (e.g. `"bollinger_window"`, `"cross_asset_n_estimators"`)
  to avoid cross-strategy collisions when multiple strategies share an
  Optuna study.
- **Composite passthrough bundle.** When a leaf has a fit-once scaler
  AND >5 ctor kwargs (Hybrid* models, FeaturePipeline + classifier),
  freeze every passthrough in a module-private
  `@dataclass(frozen=True)` with `tuple[str, ...]` for any list-typed
  field, rebuild the leaf at the top of `train()` from
  `**asdict(self._params)`, and drift-guard with
  `assert_params_match_constructor(_LeafParams, LeafClass)`. Exemplars:
  `_HybridReturnParams`, `_HybridVolParams`, `_MomentumConfig`. For
  <=5 passthrough kwargs, plain `self._x` attributes are simpler.
- **Autoload skip rules.** `autoload_package` (in
  `src/core/registry.py`) imports every module in this package EXCEPT
  ones whose name starts with `_` (templates, helpers) and the literal
  `interface` (the ABC). Spelling a strategy module name with a leading
  underscore produces "strategy not found" at YAML-load time; rename to
  drop the underscore.

## Templates and exemplars

| Need | Look at |
| --- | --- |
| Skeleton to copy | `_template.py` |
| Single-asset, no ML leaves | `adaptive_bollinger.py` |
| Single-asset composite (owns pipeline + classifier) | `momentum_gatekeeper.py` |
| Single-asset composite (passthrough bundle pattern) | `return_forecast.py`, `volatility_targeting.py` |
| Pairs (two-leg) | `pairs_trading.py` |
| Multi-feature single-asset (wide `<ohlcv>_<TICKER>` frame) | `cross_asset_momentum.py` |

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
