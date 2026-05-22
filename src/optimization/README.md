# `src/optimization/`

Optuna-backed joint hyperparameter search over an `ExperimentConfig`.
Each trial materialises a fresh config, runs a full walk-forward
experiment under the study directory, and returns a scalar objective.

## Public surface

| Symbol | Role |
| --- | --- |
| `StrategyTuner` | Frozen dataclass: `experiment_cfg`, `hpo_cfg`, `store_root`. `run(progress=...)` returns the completed `optuna.Study`. |
| `sample_trial_params(cfg, trial)` | Delegates to the strategy's static `suggest_params(trial)` and returns a fresh dict. |
| `build_objective(kind)` | `ObjectiveKind` → `IObjective` (`SharpeObjective`, `CalmarObjective`, `SortinoMinusDrawdownPenaltyObjective`). All objectives MAXIMIZE. |
| `build_sampler(kind, seed)` | `SamplerKind` → `BaseSampler` (TPE, Random, CMA-ES, QMC). Always seeded. |
| `build_pruner(kind)` | `PrunerKind` → `BasePruner` (Median, Hyperband, Percentile@25, NopPruner). |
| `TrialCallback` | Optuna callback: appends `trials.jsonl`, refreshes `best_config.yaml` on improvement. |
| `BEST_CONFIG_YAML`, `TRIALS_JSONL` | Re-exports of the on-disk filenames. |

## Layout

| File | Role |
| --- | --- |
| `tuner.py` | `StrategyTuner`, `_materialize_trial_config`, study-dir lifecycle, resume + content-hash check. |
| `sampling.py` | `sample_trial_params` — thin wrapper over the strategy's `suggest_params`. |
| `objectives.py` | `IObjective` Protocol + the three concrete adapters. |
| `samplers.py` | `build_sampler` factory. |
| `pruners.py` | `build_pruner` factory. |
| `checkpointing.py` | `TrialCallback` writing append-only `trials.jsonl` + the latest `best_config.yaml`. |

## On-disk layout per study

```
<store_root>/hpo/<study_name>/
    optuna_study.db          # SQLite — cross-process resume
    experiment_config.yaml   # frozen base config (content-hash gated)
    hpo_config.yaml          # frozen HPO config
    trials.jsonl             # append-only per-trial record
    best_config.yaml         # refreshed per new best
    trials_artifacts/
        runs/<experiment_id>/
            ...              # per-trial Experiment.run() artifacts
    plots/  tables/          # produced by generate_hpo_report
```

## Resume + content-hash gate

A repeat call with the same `study_name` + `store_root` re-opens the
SQLite study and runs `n_trials` ADDITIONAL trials. If the on-disk
`experiment_config.yaml` content-hash doesn't match the in-memory
config, the tuner refuses (silently studying under a different
objective would invalidate the trial history).

## Snippet

```python
from src.core.config import load_experiment_config
from src.core.hpo_config import load_hpo_config
from src.optimization.tuner import StrategyTuner

study = StrategyTuner(
    experiment_cfg=load_experiment_config("config/strategies/adaptive_bollinger.yaml"),
    hpo_cfg=load_hpo_config("config/hpo/adaptive_bollinger.yaml"),
).run(progress=True)
print(study.best_value, study.best_params)
```

## Cross-links

- Per-trial run path goes through `src/orchestration/builder.py::build_experiment`
  → `Experiment.run` → `evaluate_walk_forward`.
- Aggregate metrics computed by `src/analysis/metrics_aggregator.py`
  feed every objective.
- Strategy / model search spaces are owned by each component's static
  `suggest_params` (under `src/strategies/` and `src/models/`).
- HPO report rendering lives in `src/visualization/hpo_reporter.py`.
