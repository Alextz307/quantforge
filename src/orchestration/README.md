# `src/orchestration/`

End-to-end coordination layer: turns a validated config into a wired
experiment, runs the walk-forward, persists every artifact, and composes
multiple runs into comparison / regime reports.

## Public surface

| Entry point | Role |
| --- | --- |
| `build_experiment(cfg)` | `ExperimentConfig` → wired `Experiment` (resolves data source, strategy, validator, engine, slippage, optional feature-pipeline factory, optional pretrained leaves). |
| `Experiment.run(options)` | Execute walk-forward; write `config.yaml`, `manifest.json`, `fold_results.jsonl`, `metrics.json`, `strategy_state/`, optional report. |
| `run_comparison(configs, ..., regime_config=...)` | Run N strategies on aligned data, rank them, pairwise-bootstrap Sharpe differentials. With `regime_config` set, also tags every fold by regime and emits a strategy × regime aggregate map. |
| `run_regime_report(run_dir, regime_cfg, ...)` | Re-fetch a saved run's bars, tag them with a regime detector, split folds, aggregate per regime. |
| `train_model_standalone(...)` | Fit a leaf model (HybridReturn, HybridVolatility, DirectionalClassifier) on the dev region and persist it for `pretrained_leaves` injection. |
| `load_model_artifact(path)` | Inverse of the standalone training save — returns `(model, artifact_manifest)`. |
| `Manifest`, `PretrainedLeafRecord` | Frozen, round-tripable provenance dataclasses. |

## Layout

| File | Role |
| --- | --- |
| `builder.py` | Capability-flag dispatch (`is_pairs_strategy`); 1- or 2-ticker validation; pretrained-leaf loading and ctor injection. |
| `experiment.py` | `Experiment` dataclass + `RunOptions`; ticker-count dispatch in `_fetch_bars`; `_fetch_pair_bars` builds the wide-format `_a`/`_b` join for the 2-ticker path; persistence pipeline. |
| `manifest.py` | `Manifest` (run-level provenance) + `PretrainedLeafRecord` (per-leaf provenance, used by holdout-eval). |
| `comparison.py` | `run_comparison` + sequential / `ProcessPoolExecutor` paths + paired stationary bootstrap. |
| `regime.py` | `regime_registry` for detector classes (`PeriodRegimeDetector`, `BullBear200MA`, `VolatilityQuintile`). |
| `regime_run.py` | `run_regime_report` driver + `load_run_from_disk` reader for persisted runs. |
| `standalone_training.py` | Backend for `experiment train-model`; fits a leaf, writes the artifact + manifest. |
| `model_artifact.py` | Artifact loader + `ArtifactManifest` (pairs with `standalone_training`). |
| `pretrained_leaves.py` | `normalize_pretrained_leaves(...)`: validates the injected leaf map against the strategy's `_leaf_keys`. |
| `git_info.py` | `read_git_sha()` — best-effort short SHA, `"unknown"` outside git. |
| `types.py` | `ExperimentResult`, `FoldRecord`, `StrategyComparisonReport`, `PairwiseSignificance`, `RegimeReport`, `RegimeSlice`. |

## Multi-ticker / pairs flow

Two-ticker configs (e.g. `PairsTrading`) are dispatched by
`build_experiment`'s capability check on
`IStrategy.is_pairs_strategy`. `_fetch_pair_bars` inner-joins both legs
on shared timestamps and suffixes the OHLCV columns via
`PAIRS_LEG_SUFFIXES` (`_a`, `_b`). Single-leg strategies see the
1-ticker frame unchanged.

## Pretrained-leaf injection

When `cfg.pretrained_leaves` is non-empty, `build_experiment` loads each
artifact, validates the strategy ctor accepts a `pretrained_leaves`
kwarg, and threads the loaded models in. Each leaf's
`training_metadata` is recorded on the manifest as a
`PretrainedLeafRecord` so holdout-eval can verify temporal separation
without reloading the artifact.

## Snippet

```python
from pathlib import Path

from src.core.config import load_experiment_config
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import RunOptions

cfg = load_experiment_config(Path("config/strategies/adaptive_bollinger.yaml"))
result = build_experiment(cfg).run(RunOptions(write_report=True))
print(result.experiment_id, len(result.folds))
```

## Cross-links

- Drives `src/engine/` (walk-forward) and `src/strategies/` (concrete strategies).
- Reads from `src/data/` (sources, fingerprint) and `src/core/` (config, persistence, registry, temporal).
- Writes to `src/visualization/` (`StrategyReporter`, `ComparisonReporter`, `RegimeReporter`) on report opt-in.
- HPO entry point lives in `src/optimization/tuner.py` and reuses `Experiment` per trial.
