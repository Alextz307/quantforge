# `src/orchestration/`

End-to-end coordination layer: turns a validated config into a wired
experiment, runs the walk-forward, persists every artifact, and composes
multiple runs into cross-strategy comparison reports.

## Public surface

| Entry point | Role |
| --- | --- |
| `build_experiment(cfg)` | `ExperimentConfig` -> wired `Experiment` (resolves data source, strategy, validator, engine, slippage, optional feature-pipeline factory). |
| `Experiment.run(options)` | Execute walk-forward; write `config.yaml`, `manifest.json`, `fold_results.jsonl`, `metrics.json`, `strategy_state/`, `run.log` (root-logger tee for the duration of the run), optional report. `RunOptions.compute_feature_importance` (off by default; on for the study's final per-leg runs) additionally writes `feature_importance.json` for feature-consuming strategies. |
| `run_comparison(configs, ..., reused_results=...)` | Run N strategies on aligned data, rank them, pairwise-bootstrap Sharpe differentials. With `reused_results=` set, the per-strategy walk-forward step is skipped and prior fold artifacts feed the rest of the pipeline; `configs=` may be omitted on the reuse path and strategy names are read from each result's `manifest.name`. |
| `load_experiment_result(run_dir)` / `load_experiment_config_from_run(run_dir)` / `resolve_run_dir(store, id)` | Reconstruct an `ExperimentResult` (or its frozen `ExperimentConfig`) from a persisted run directory; powers `compare --reuse-runs`. `resolve_run_dir` resolves a run by id under both the flat `runs/<id>` and study-nested `studies/<x>/runs/<id>` layouts (flat-first, recursive fallback), so a deployment can point at a study-internal run. |
| `load_strategy_from_run_dir(run_dir)` | Registry-driven loader: returns a fully-trained `IStrategy` instance whose `generate_signals` works immediately. Used by the deployment layer. |
| `strategy_supports_feature_importance(name)` | Whether a registered strategy can produce feature importance (derived from the class overriding `feature_columns`, so it can't drift). Drives the `importance` CLI / webapp's pre-launch guard and the read endpoint's `computable` flag. |
| `create_deployment(...)` / `load_deployment(store, id)` / `predict(deployment_id, store_root, as_of=None)` / `read_signals(...)` | Live-deployment primitives: pin a trained run, generate today's signal, accumulate a signed log. Predict-only, no refit clock. |
| `next_signal_date(bar_ts, interval)` | The trading day a signal computed at `bar_ts` is *for* (next NYSE session after the last completed bar, per the engine's `t -> t+1` shift; exchange holidays + early closes included). Display-only; never a leakage boundary. |
| `resolve_strategy_state_path(source_kind, source_id, store_root)` | Single source of truth for "where does this source's `strategy_state/` live", handling both `run` and `hpo` (Optuna best-trial lookup). |
| `run_holdout_eval(source, out_name, store_root)` | Refit a fresh strategy on the full dev region, evaluate once on the reserved holdout. Cross-checks `data_hash` + `holdout_start` against the source manifest before fitting. |
| `run_study(spec_path, ...)` | Drive a full `StudySpec` end-to-end: cross-strategy x cross-universe sweep with per-leg resume. |
| `consolidate_study(study_dir)` | Walk a completed study tree (`runs/`, `holdout_evals/`, `comparisons/`); return a `ConsolidatedStudyReport` value object covering every completed leg. |
| `plan_clean(store_root)` / `apply_clean(plan)` | Tidy ephemeral subdirs under an `experiment_results/` tree; default dry-run, refuse to delete dirs containing git-tracked files. Also removes a stray top-level sweep-tracking-file allowlist (`.sweep_pid`, `sweep_*.log`, ...); other top-level files are preserved. |
| `Manifest` | Frozen, round-tripable provenance dataclass. |
| `LegState`, `StudyState`, `ConsolidatedStudyReport`, `HoldoutSnapshot` | Frozen, round-tripable resume / consolidation dataclasses. |

## Layout

| File | Role |
| --- | --- |
| `builder.py` | Capability-flag dispatch (`is_pairs_strategy`); 1- or 2-ticker validation. |
| `experiment.py` | `Experiment` dataclass + `RunOptions`; ticker-count dispatch in `fetch_bars`; `_fetch_pair_bars` builds the wide-format `_a`/`_b` join for the 2-ticker path; persistence pipeline. |
| `manifest.py` | `Manifest` (run-level provenance). |
| `comparison.py` | `run_comparison` + sequential / `ProcessPoolExecutor` paths + paired stationary bootstrap. |
| `deployment.py` | `Deployment` / `SignalRow` dataclasses + `create_deployment` / `predict` / `read_signals` / `resolve_strategy_state_path`. Live-inference layer over a frozen trained run. |
| `run_loader.py` | `load_experiment_result` / `load_experiment_config_from_run` / `load_strategy_from_run_dir` / `resolve_run_dir` / `strategy_supports_feature_importance` - read manifest + folds + frozen YAML back into typed objects; the strategy loader is registry-driven and powers `deployment.predict`. |
| `holdout_eval.py` | `run_holdout_eval` (one-shot honest-OOS) + `resolve_source` (CLI source-pair resolver). Writes a `holdout_eval.json` payload with the `is_holdout_eval: true` marker - does NOT write a `Manifest` (post-hoc evaluation, not a new experiment). |
| `study.py` | Empirical-study orchestrator: leg expansion, universe-profile composition, per-universe cross-strategy compare. |
| `study_state.py` | `LegState` + `StudyState` resume dataclasses; atomic write via `os.replace`; spec-hash guard refuses to resume against a mutated spec. |
| `study_report.py` | `consolidate_study(study_dir)` walker + `ConsolidatedStudyReport` / `HoldoutSnapshot` value types - collapses per-leg artifacts (metrics, holdout, DSR, floor-bind, feature-importance) into a cross-leg view consumed by `StudyReportReporter`. |
| `clean.py` | `plan_clean` / `apply_clean` / `format_plan` - tidy ephemeral `experiment_results/` subdirs with a `--keep` preserve set and a `git ls-files` refusal on tracked content; `CleanPlan.stray_files` carries a top-level sweep-tracking-file allowlist removed alongside the dirs. |
| `git_info.py` | `read_git_sha()` - best-effort short SHA, `"unknown"` outside git. |
| `types.py` | `ExperimentResult`, `FoldRecord`, `StrategyComparisonReport`, `PairwiseSignificance` (round-trips via `to_dict` / `from_dict`). |

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

## Live-deployment flow

`deployment.py` bridges the backtest world (frozen trained strategies
on disk) and live decision support. A `Deployment` is *only* a pinned
pointer to a trained run plus an accumulating `signals.jsonl`. No
refit clock. Refreshing a stale model is a new experiment run via
the existing flow, then a new deployment pointed at it.

`predict(deployment_id, as_of=None, store_root=...)` flow:

1. Load the deployment manifest; resolve the source's run-dir via
   `resolve_strategy_state_path`.
2. Load the trained strategy via `run_loader.load_strategy_from_run_dir`
   (registry-driven, supports every strategy that implements `save`/`load`).
3. Read `training_metadata` (frozen on disk) for `train_end` + `interval`.
4. Fetch a warmup window through `as_of` via the cadence-specific
   `LiveBarFetcher`; the window may overlap training.
5. **Anti-leakage guard**: `bars.index[-1] > train_end`, strictly
   weaker than backtest's `validate_no_overlap` but the right
   contract in a live setting where the model is already validated.
6. `strategy.generate_signals(bars).iloc[-1]` -> today's signal.
   NaN raises `WarmupInsufficientError`.
7. Idempotent append to `signals.jsonl` (dedup by `bar_ts`).

## Cross-links

- Drives `src/engine/` (walk-forward) and `src/strategies/` (concrete strategies).
- Reads from `src/data/` (sources, fingerprint, `LiveBarFetcher`) and `src/core/` (config, persistence, registry, temporal).
- Writes to `src/visualization/` (`StrategyReporter`, `ComparisonReporter`) on report opt-in.
- HPO entry point lives in `src/optimization/tuner.py` and reuses `Experiment` per trial.
- The live deployment surface (`deployment.py`) shares storage primitives with the experiment runner (`save_model_skeleton`, `attach_run_log_file`).
