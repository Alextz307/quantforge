# `src/core/`

Foundational primitives shared across the codebase: domain types,
constants, registries, the temporal-leakage primitives, persistence
layout, exceptions, the pydantic config schema, contextual logging,
device selection, and small JSON / FS helpers.

## Public surface

| Symbol | Role |
| --- | --- |
| `Interval` (StrEnum) | Bar timeframe; `annualization_factor()` returns bars/year. |
| `LossFunction`, `InformationCriterion`, `Device` | Strict StrEnum types used in ctor params + Optuna search spaces. |
| `BarData`, `Signal`, `PairSignal` | Pydantic v2 frozen value types with bounds + invariants. |
| `OHLCV_COLUMNS`, `PAIRS_LEG_SUFFIXES`, `TRADING_DAYS_PER_YEAR`, `DEFAULT_REALIZED_VOL_WINDOW`, ... | Centralised constants: magic numbers live here, never inline. |
| `ComponentRegistry[T]` + globals | `strategy_registry`, `model_registry`, `classifier_registry`, `data_source_registry`, `feature_registry`. `create_from_config(ComponentConfig)` is the config-layer entry point. |
| `TemporalSplit`, `TemporalTripleSplit`, `WalkForwardValidator`, `PurgedGroupTimeSeriesSplit`, `resolve_holdout_boundary` | Anti-leakage temporal primitives. |
| `TrainingMetadata`, `TrackedMetadata`, `collect_metadata` | Per-component training-window record + composite-strategy aggregation. |
| `ExperimentConfig` + `load_experiment_config(path)` + `write_frozen_yaml(...)` | Pydantic root config; loads the canonical `config/strategies/*.yaml`. |
| `apply_overrides(payload, overrides)` | Dotted-path mutation of a config dict (e.g. `data.tickers=[QQQ]`); powers every CLI's `--override` flag. |
| `HPOConfig` + `load_hpo_config(path)` | HPO study spec (sampler, pruner, n_trials, objective). |
| `LeakageError`, `DataQualityError`, `WarmupInsufficientError`, `guard_scaler_fit_once` | Custom exceptions + the centralised fit-once helper. |
| `get_logger(name, **context)` | Contextual logger wrapper that prepends `[k=v ...]` to every message. |
| `select_device`, `select_xgboost_device` | Auto-detect CUDA / MPS / CPU; honour user preference. |
| `seed_all(seed)` | Lazy-imports torch and seeds numpy / random / torch deterministically. |
| `json_io.read_dict / write / get_int / get_float / get_str / get_bool` | Stable JSON helpers used by every model's `save` / `load`. |
| `fs`, `utils`, `contracts` | Small focused helpers (atomic file ops, bounded-range validators, contract assertions). |

## Layout

| File | Role |
| --- | --- |
| `types.py` | StrEnums + Pydantic value types. |
| `constants.py` | All numeric / string magic values (calendar, position limits, OHLCV column tuple, pairs suffixes). |
| `registry.py` | `ComponentRegistry[T]` + global registries + `autoload_package` (used by `src/<sub>/__init__.py` to fire decorator side-effects). |
| `temporal.py` | `TemporalSplit`, `WalkForwardValidator`, `TrainingMetadata`, `TrackedMetadata`, `resolve_holdout_boundary`. |
| `persistence.py` | Canonical filenames + subdirs (`config.json`, `weights.json`, `garch/`, `lstm/`, ...) + `save_model_skeleton` + scaler round-trip helpers + `write_experiment_manifest`. |
| `exceptions.py` | `LeakageError`, `DataQualityError`, `WarmupInsufficientError`, `guard_scaler_fit_once`. |
| `config.py` | `ExperimentConfig` + Pydantic validators. |
| `config_overrides.py` | `apply_overrides(payload, overrides)` - dotted-path mutation of a config dict before pydantic re-validation, used by every CLI's `--override` flag. |
| `hpo_config.py` | Pydantic config for the `experiment tune` CLI. |
| `logging.py` | `_ContextAdapter` + `get_logger` + `attach_cli_log_file` (CLI-scoped) + `attach_run_log_file` (per-experiment) + `log_stage`. |
| `device.py` | torch / xgboost device selection helpers. |
| `seeding.py` | `seed_all` (lazy torch import). |
| `json_io.py` | Typed JSON read / write helpers (no pickle). |
| `fs.py` | Atomic write / mkdir helpers. |
| `utils.py` | `validate_open_unit_interval`, `compute_log_returns`, `next_bar_direction`, `annualized_garman_klass`. |
| `contracts.py` | `assert_*` runtime contract helpers. |

## Anti-leakage primitives

`TemporalSplit.__post_init__` rejects `train_max >= test_min`, so
construction fails fast on overlap. `TrainingMetadata.from_fit(df, ...)`
captures `train_start` / `train_end` / `n_samples` / `interval` /
`feature_columns`; the walk-forward orchestrator runs
`validate_no_overlap(test_data)` against every metadata entry returned
by `IStrategy.get_all_training_metadata()` (composite leaves included).

## Persistence convention

No pickle. No joblib. Every artifact is JSON (metadata + configs +
small numeric weights), torch `.pt` for LSTMs, or XGBoost native
`.ubj`. Filename + subdir constants live in
`persistence.py` and are imported by every model and strategy that
saves to disk, the single source of truth for the on-disk layout.

## Snippet

```python
from datetime import datetime

from src.core.config import load_experiment_config
from src.core.logging import get_logger
from src.core.registry import strategy_registry

logger = get_logger(__name__, run="demo")
cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
strategy_cls = strategy_registry.get(cfg.strategy.name)
logger.info("resolved strategy %s with params %s", strategy_cls.__name__, cfg.strategy.params)
```

## Cross-links

- C++ side mirrors `constants.py` in `cpp/include/quant/core/types.hpp`
  (the Python check in `tests/unit/test_constants_drift.py` enforces
  parity).
- Every concrete subsystem (`src/strategies/`, `src/models/`,
  `src/data/`, `src/features/`) registers itself on the registries
  defined here; the autoload helper fires the decorator side-effects.
- The orchestration layer (`src/orchestration/`) drives all of these
  primitives; `Manifest` is the on-disk consumer of `TrainingMetadata`.
