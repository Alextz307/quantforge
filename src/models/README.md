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
| `LSTMPredictor` (`"lstm"`) | torch LSTM with early stopping + best-state checkpointing (`best_state.pt`). |
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

## Fit / persistence patterns

- **`_fitted` is the transactional commit.** Every model sets
  `self._fitted = True` only after every dependent piece of state
  (scaler, weights, frozen params, `_training_metadata`) is in place.
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
