# `src/analysis/`

Numeric post-processing layer between fold records and reports:
fold-level -> run-level metric aggregation, strategy ranking, and
pairwise / forecaster significance tests.

## Public surface

| Symbol | Role |
| --- | --- |
| `aggregate_folds(folds, *, annualization_factor, risk_free_rate=0.0, rng=None)` -> `AggregateStats` | Collapse `tuple[FoldRecord, ...]` to per-metric mean / std / 95% CI + run-wide scalars, plus the pooled-OOS Sharpe (`sharpe_pooled`) and its PSR (`psr_pooled`) over the stitched fold returns. `risk_free_rate` keeps the pooled Sharpe on the same scale as the per-fold metrics. Deterministic by default (fixed RNG seed). |
| `AggregateStats.to_dict()` / `AggregateStats.from_dict(d)` | Flat dict consumed by HPO objectives + `metrics.json`; `from_dict` reconstructs it (the study report reads each run's persisted aggregate rather than recomputing). |
| `rank_strategies(per_strategy_stats)` -> `pd.DataFrame` | Tidy ranking table sorted by mean-of-folds Sharpe (Sortino then name as tie-breaks), via a stable mergesort. |
| `paired_bootstrap_sharpe_differential(returns_a, returns_b, ...)` | Stationary bootstrap on aligned bar-level returns; returns a 95% CI on the Sharpe differential. |
| `bootstrap_sharpe_ci(returns, ...)` | 95% CI on a single strategy's Sharpe via stationary bootstrap. |
| `diebold_mariano_test(forecasts_a, forecasts_b, realised, *, loss=...)` | DM test on aligned forecaster outputs (with Harvey-Leybourne-Newbold small-sample correction). |
| `deflated_sharpe_ratio(trial_sharpes, *, sample_length)` -> `DeflatedSharpe` | Bailey-López de Prado 2014 multiple-testing-adjusted significance for an HPO study's best Sharpe. Inputs come from the Optuna `study.db`. |
| `compute_pooled_sharpe(returns, *, annualization_factor, risk_free_rate=0.0)` -> `PooledSharpe` | Pooled-OOS Sharpe + zero-benchmark Probabilistic Sharpe Ratio for one leg's stitched (seams-dropped) fold returns. |
| `deflate_pooled_across_legs(legs)` -> `tuple[float, ...]` | Selection-deflated pooled-Sharpe probability per leg (benchmark = expected-max Sharpe across all legs) - the "best of many pairs" guard, aligned to input order. |
| `compute_buy_and_hold(bars, *, slippage, interval, ...)` -> `BaselineResult` | Long-only "do nothing" baseline on canonical OHLCV; runs through the same `IBacktestEngine` + slippage scenario as the strategy. |
| `percentile_ci(samples, confidence)` | Symmetric percentile bounds; reused by aggregator and bootstrap helpers. |
| `evaluate_signals(bar_timestamps, signal_values, opens)` -> `SignalEvaluation` | Backward open->open scoring of a deployment's emitted signals against realised session opens: per-signal realised return + directional hit + cumulative growth, plus hit-rate / cumulative-return headline stats. Pure and read-only, the evaluation counterpart to signal generation. |
| `compute_fold_importance(strategy, test_frame, fold_index, *, n_repeats, rng)` -> `FoldImportance \| None` | Out-of-sample feature importance for a feature-consuming strategy on one fold's frozen model: permutation importance (model-agnostic) plus XGBoost native gain when the strategy exposes a booster. Skips rule-based strategies. |
| `permutation_importance(score_fn, features, feature_columns, *, n_repeats, rng, baseline=None)` | Per-column OOS permutation importance against a higher-is-better score; never mutates the input frame. |
| `xgb_gain_importance(gain, feature_columns)` / `aggregate_fold_importance(folds)` / `build_importance_artifact(folds)` / `read_aggregated_importance(payload)` | Gain-map wrapper (0.0-filled), cross-fold mean +/- std aggregation, and the `feature_importance.json` write/read pair. |

## Layout

| File | Role |
| --- | --- |
| `metrics_aggregator.py` | `AggregateStats` (+ `from_dict`), `aggregate_folds` (mean-of-folds CIs + pooled-OOS Sharpe/PSR), `_mean_std_ci` (IID percentile bootstrap over fold means). |
| `ranking.py` | `rank_strategies` - Sharpe-sorted ranking table (stable mergesort tie-break). |
| `significance.py` | Stationary bootstrap (Politis-Romano), Diebold-Mariano (HLN-corrected), deflated Sharpe + pooled-OOS Sharpe / PSR / cross-leg deflation (Bailey-López de Prado); result types `BootstrapCI`, `DMResult`, `DeflatedSharpe`, `PooledSharpe` + enums `DMLoss`, `DMDirection`. |
| `baselines.py` | `BaselineResult` + `compute_buy_and_hold` - per-universe long-only reference. |
| `signal_evaluation.py` | `ScoredSignal`, `SignalEvaluation`, `evaluate_signals` - open->open scoring of a deployment's signal log. |
| `feature_importance.py` | `ImportanceMethod` (StrEnum) + `FeatureImportance` / `FoldImportance` / `AggregatedImportance` result types; permutation + XGBoost-gain importance, per-fold driver, cross-fold aggregation, and `feature_importance.json` round-trip. OOS-only, never mutates inputs. |

## Aggregation choices worth knowing

- **Two RNGs.** `metrics_aggregator` and `significance` use distinct
  fixed seeds so a joint invocation (aggregate -> bootstrap) doesn't
  share a draw sequence by accident.
- **IID resampling at fold level.** The 95% CIs on Sharpe / Sortino /
  Calmar come from an IID percentile bootstrap over fold means. Folds
  are disjoint walk-forward windows, so the IID assumption holds at
  fold granularity. Autocorrelation-aware bootstrap on raw bar returns
  is reserved for `significance.py`.
- **Block size heuristic.** Bootstrap helpers default
  `block_size = max(1, round(2 * sqrt(n)))` (a widely-used heuristic);
  callers can pass an explicit block size when they know the
  autocorrelation structure.

## Snippet

```python
from src.analysis.metrics_aggregator import aggregate_folds
from src.analysis.ranking import rank_strategies
from src.orchestration.run_loader import load_experiment_result

run_dir = "experiment_results/runs/<exp_id>"
result = load_experiment_result(run_dir)
# Interval + risk-free rate come from the manifest (no config reload needed).
stats = aggregate_folds(
    result.folds,
    annualization_factor=result.manifest.interval.annualization_factor(),
    risk_free_rate=result.manifest.risk_free_rate,
)
# Mean-of-folds = stability; pooled = realised OOS track record.
print(stats.sharpe_mean, stats.sharpe_pooled, stats.psr_pooled)

# Two-strategy ranking (sorted by Sharpe)
ranking = rank_strategies({"A": stats, "B": stats})
print(ranking.to_string(index=False))
```

## Cross-links

- Drives `src/visualization/comparison_reporter.py` (consumes
  `rank_strategies` output).
- Powers `src/orchestration/comparison.py` pairwise significance and
  every HPO objective in `src/optimization/objectives.py`.
- Fold record shape is owned by `src/orchestration/types.py::FoldRecord`.
