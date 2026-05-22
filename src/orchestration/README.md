# `src/orchestration/`

End-to-end coordination layer: turns a validated config into a wired
experiment, runs the walk-forward, persists every artifact, and composes
multiple runs into comparison / regime reports.

## Public surface

| Entry point | Role |
| --- | --- |
| `build_experiment(cfg)` | `ExperimentConfig` → wired `Experiment` (resolves data source, strategy, validator, engine, slippage, optional feature-pipeline factory). |
| `Experiment.run(options)` | Execute walk-forward; write `config.yaml`, `manifest.json`, `fold_results.jsonl`, `metrics.json`, `strategy_state/`, optional report. |
| `run_comparison(configs, ..., regime_config=..., reused_results=...)` | Run N strategies on aligned data, rank them, pairwise-bootstrap Sharpe differentials. With `regime_config` set, also tags every fold by regime and emits a strategy × regime aggregate map. With `reused_results=` set (one `ExperimentResult` per config), the per-strategy walk-forward step is skipped and prior fold artifacts feed the rest of the pipeline. |
| `load_experiment_result(run_dir)` / `load_experiment_config_from_run(run_dir)` | Reconstruct an `ExperimentResult` (or its frozen `ExperimentConfig`) from a persisted run directory; powers `compare --reuse-runs`. |
| `run_regime_report(run_dir, regime_cfg, ...)` | Re-fetch a saved run's bars, tag them with a regime detector, split folds, aggregate per regime. |
| `run_holdout_eval(source, out_name, store_root)` | Refit a fresh strategy on the full dev region, evaluate once on the reserved holdout. Cross-checks `data_hash` + `holdout_start` against the source manifest before fitting. |
| `run_study(spec_path, ...)` | Drive a full `StudySpec` end-to-end — cross-strategy × cross-universe sweep with per-leg resume. |
| `consolidate_study(study_dir)` | Walk a completed study tree (`runs/`, `regime_reports/`, `holdout_evals/`, `comparisons/`); return a `ConsolidatedStudyReport` value object covering every completed leg. |
| `plan_clean(store_root)` / `apply_clean(plan)` | Tidy ephemeral subdirs under an `experiment_results/` tree; default dry-run, refuse to delete dirs containing git-tracked files. |
| `Manifest` | Frozen, round-tripable provenance dataclass. |
| `LegState`, `StudyState`, `ConsolidatedStudyReport`, `HoldoutSnapshot` | Frozen, round-tripable resume / consolidation dataclasses. |

## Layout

| File | Role |
| --- | --- |
| `builder.py` | Capability-flag dispatch (`is_pairs_strategy`); 1- or 2-ticker validation. |
| `experiment.py` | `Experiment` dataclass + `RunOptions`; ticker-count dispatch in `fetch_bars`; `_fetch_pair_bars` builds the wide-format `_a`/`_b` join for the 2-ticker path; persistence pipeline. |
| `manifest.py` | `Manifest` (run-level provenance). |
| `comparison.py` | `run_comparison` + sequential / `ProcessPoolExecutor` paths + paired stationary bootstrap. |
| `regime.py` | `regime_registry` for detector classes (`PeriodRegimeDetector`, `BullBear200MA`, `VolatilityQuintile`). |
| `regime_run.py` | `run_regime_report` driver + `load_run_from_disk` reader for persisted runs. |
| `run_loader.py` | `load_experiment_result` / `load_experiment_config_from_run` — read manifest + folds + frozen YAML back into typed objects, used by `compare --reuse-runs`. |
| `holdout_eval.py` | `run_holdout_eval` (one-shot honest-OOS) + `resolve_source` (CLI source-pair resolver). Writes a `holdout_eval.json` payload with the `is_holdout_eval: true` marker — does NOT write a `Manifest` (post-hoc evaluation, not a new experiment). |
| `study.py` | Empirical-study orchestrator: leg expansion, universe-profile composition, per-universe cross-strategy compare. |
| `study_state.py` | `LegState` + `StudyState` resume dataclasses; atomic write via `os.replace`; spec-hash guard refuses to resume against a mutated spec. |
| `study_report.py` | `consolidate_study(study_dir)` walker + `ConsolidatedStudyReport` / `HoldoutSnapshot` value types — collapses per-leg artifacts into a cross-leg view consumed by `StudyReportReporter`. |
| `clean.py` | `plan_clean` / `apply_clean` / `format_plan` — tidy ephemeral `experiment_results/` subdirs with a hard preserve on `thesis_demo/` and a `git ls-files` refusal on tracked content. |
| `git_info.py` | `read_git_sha()` — best-effort short SHA, `"unknown"` outside git. |
| `types.py` | `ExperimentResult`, `FoldRecord`, `StrategyComparisonReport`, `PairwiseSignificance` (round-trips via `to_dict` / `from_dict`), `RegimeReport`, `RegimeSlice`. |

## Multi-ticker / pairs flow

Two-ticker configs (e.g. `PairsTrading`) are dispatched by
`build_experiment`'s capability check on
`IStrategy.is_pairs_strategy`. The internal `_fetch_pair_bars` inner-joins both legs
on shared timestamps and suffixes the OHLCV columns via
`PAIRS_LEG_SUFFIXES` (`_a`, `_b`). Single-leg strategies see the
1-ticker frame unchanged.

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
