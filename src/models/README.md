# `src/models/`

Predictors and classifiers used by the strategies. Statistical leaves
(GARCH, ARMA), neural leaves (LSTM, XGBoost), composite hybrids
(GARCH+LSTM, ARMA+LSTM), and the Engle-Granger cointegration tester.
Every fitted model carries `TrainingMetadata` and supports JSON /
native-format persistence.

## Public surface

| Symbol | Role |
| --- | --- |
| `IPredictor` | ABC: `fit(train, target, *, checkpoint_path, **kw)`, `predict`, `predict_single`, default `save` / `load` (raise), `training_metadata` property, `get_all_training_metadata` (composite-aware). |
| `IClassifier` | ABC: `fit`, `predict_proba`, `predict`, default `save` / `load`, same metadata plumbing. |
| `GARCHPredictor` (`"garch"`) | GARCH(p,q) volatility predictor; AIC/BIC order selection; params frozen after `fit`. |
| `ARMAPredictor` (`"arma"`) | ARMA(p,q) return predictor; `pmdarima.auto_arima` order selection; one-step-ahead `predict`. |
| `LSTMPredictor` (`"lstm"`) | torch LSTM with early stopping + best-state checkpointing (`best_state.pt`). The training module is wrapped in `torch.compile(mode="reduce-overhead")` for kernel reordering; the `amp: bool = False` ctor kwarg opts into mixed-precision (CUDA-only — MPS / CPU silently no-op even when `True`). |
| `DirectionalClassifier` (`"directional_classifier"`) | XGBoost binary classifier (price-direction); early stopping + best-iteration checkpointing (`best_iteration.ubj`). |
| `HybridVolatilityModel` (`"hybrid_volatility"`) | GARCH conditional variance + LSTM residual correction. Owns both leaves; rebuilt fresh per `fit`. |
| `HybridReturnModel` (`"hybrid_return"`) | ARMA conditional mean + LSTM residual correction. Same composition pattern. |
| `CointegrationTester` + `CointegrationResult` | `engle_granger(series_a, series_b, p_value_threshold)` static helper. |
| `TemporalDataset` | `torch.utils.data.Dataset` with sliding-window features + next-step target; the only dataset shape any LSTM here consumes. |

All model and classifier classes register themselves on
`model_registry` / `classifier_registry` for config-driven
instantiation and standalone-training artifacts.

## Layout

| File | Role |
| --- | --- |
| `interface.py` | `IPredictor`, `IClassifier` ABCs + composite-aware `get_all_training_metadata`. |
| `garch.py` | `GARCHPredictor` + arch-library wrapper. |
| `arma.py` | `ARMAPredictor` + `_StatsmodelsARMAAdapter` (round-trips pmdarima → statsmodels at load time). |
| `lstm.py` | `LSTMPredictor` + `MarketLSTM` torch module + loss-function dispatch. |
| `xgboost_classifier.py` | `DirectionalClassifier` + `_ProgressAndCheckpointCallback`. |
| `hybrid_volatility.py` | `HybridVolatilityModel` + frozen `_HybridVolConfig`. |
| `hybrid_return.py` | `HybridReturnModel` + frozen `_HybridReturnConfig`. |
| `cointegration.py` | `CointegrationTester.engle_granger`. |
| `dataset.py` | `TemporalDataset` (anti-leakage sliding window). |
| `_garch_cache.py` | Module-private `GarchGridCache` + `garch_cache_context`; `StrategyTuner.run` binds the cache for an entire HPO study so two trials whose `(p_max, q_max)` grids overlap on the same fold reuse one another's `(p, q)` AIC tables. Outside HPO the `ContextVar` is unset and `_grid_search` runs a fresh sweep — same path as before the cache shipped. |

## Fit / persistence patterns

- **`training_metadata` is the transactional commit.** Every model
  ends `fit()` with `self._set_fitted_with_metadata(metadata)` only
  after every dependent piece of state (scaler, weights, frozen params)
  is in place. The helper is the only legal mutator of the slot;
  `training_metadata is not None` is the single fitted-state signal.
  Read-side guards call `self._assert_fitted_with_metadata()` (caller
  name auto-derived from the calling frame); composites (Hybrid models)
  layer leaf-presence checks (`_scaler is None`, `_model is None`) as
  separate statements.
- **Statistical leaves freeze params after `fit`.** GARCH conditional
  variance and ARMA one-step forecast use only the fitted params —
  no re-estimation during `predict`.
- **Required `feature_columns`.** LSTM, XGBoost, hybrids take
  `feature_columns: list[str]` as a required ctor parameter — never
  inferred from `train_data.columns` (the caller may legitimately
  carry extra columns like raw `close` alongside features).
- **Best-state checkpointing.** `LSTMPredictor.fit` and
  `DirectionalClassifier.fit` accept a `checkpoint_path`; on every
  validation-metric improvement they write `best_state.pt` /
  `best_iteration.ubj`. The walk-forward orchestrator wires these per
  fold under `<run_dir>/checkpoints/fold_<i>/`.
- **Composite black-box composition.** `HybridVolatilityModel` and
  `HybridReturnModel` own their leaves and call public API only —
  never reach into private state. Each rebuilds its leaves from a
  frozen `_Hybrid*Config` at the top of `fit` so a fresh scaler is
  created (the leaves' fit-once guard rejects a second `fit` on the
  same instance).

## Snippet

```python
from datetime import datetime

from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.data.loader import YFinanceSource
from src.models.garch import GARCHPredictor

bars = YFinanceSource().fetch("SPY", datetime(2020, 1, 1), datetime(2024, 12, 31))
log_returns = compute_log_returns(bars["close"])

holdout = 63
predictor = GARCHPredictor(p_max=2, q_max=2, interval=Interval.DAILY)
predictor.fit(bars.iloc[:-holdout], log_returns.iloc[:-holdout])
sigma_forecast = predictor.predict(bars.iloc[-holdout:])
```

## Cross-links

- All persistence helpers (`save_model_skeleton`, scaler round-trip,
  filename + subdir constants) live in `src/core/persistence.py`.
- Anti-leakage primitives (`TrainingMetadata`, `TrackedMetadata`,
  `mark_pretrained`) live in `src/core/temporal.py`.
- Strategies in `src/strategies/` own composite leaves via the
  `pretrained_leaves` injection workflow handled by
  `src/orchestration/pretrained_leaves.py` and
  `src/orchestration/standalone_training.py`.
