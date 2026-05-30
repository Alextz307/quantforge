# `src/analysis/`

Numeric post-processing layer between fold records and reports:
fold-level -> run-level metric aggregation, strategy ranking, and
pairwise / forecaster significance tests.

## Public surface

| Symbol | Role |
| --- | --- |
| `aggregate_folds(folds, *, rng=None)` -> `AggregateStats` | Collapse `tuple[FoldRecord, ...]` to per-metric mean / std / 95% CI + run-wide scalars. Deterministic by default (fixed RNG seed). |
| `AggregateStats.to_dict()` | Flat dict consumed by HPO objectives + `metrics.json`. |
| `rank_strategies(per_strategy_stats, *, by=...)` -> `pd.DataFrame` | Tidy ranking table with deterministic tie-break. `RankingMetric` selects the primary axis (Sharpe / Sortino / Calmar). |
| `paired_bootstrap_sharpe_differential(returns_a, returns_b, ...)` | Stationary bootstrap on aligned bar-level returns; returns a 95% CI on the Sharpe differential. |
| `bootstrap_sharpe_ci(returns, ...)` | 95% CI on a single strategy's Sharpe via stationary bootstrap. |
| `diebold_mariano_test(forecasts_a, forecasts_b, realised, *, loss=...)` | DM test on aligned forecaster outputs (with Harvey-Leybourne-Newbold small-sample correction). |
| `deflated_sharpe_ratio(trial_sharpes, *, sample_length)` -> `DeflatedSharpe` | Bailey-López de Prado 2014 multiple-testing-adjusted significance for an HPO study's best Sharpe. Inputs come from the Optuna `study.db`. |
| `compute_buy_and_hold(bars, *, slippage, interval, ...)` -> `BaselineResult` | Long-only "do nothing" baseline on canonical OHLCV; runs through the same `IBacktestEngine` + slippage scenario as the strategy. |
| `percentile_ci(samples, confidence)` | Symmetric percentile bounds; reused by aggregator and bootstrap helpers. |
| `evaluate_signals(bar_timestamps, signal_values, opens)` -> `SignalEvaluation` | Backward open->open scoring of a deployment's emitted signals against realised session opens: per-signal realised return + directional hit + cumulative growth, plus hit-rate / cumulative-return headline stats. Pure and read-only, the evaluation counterpart to signal generation. |
| `compute_fold_importance(strategy, test_frame, fold_index, *, n_repeats, rng)` -> `FoldImportance \| None` | Out-of-sample feature importance for a feature-consuming strategy on one fold's frozen model: permutation importance (model-agnostic) plus XGBoost native gain when the strategy exposes a booster. Skips rule-based strategies. |
| `permutation_importance(score_fn, features, feature_columns, *, n_repeats, rng, baseline=None)` | Per-column OOS permutation importance against a higher-is-better score; never mutates the input frame. |
| `xgb_gain_importance(gain, feature_columns)` / `aggregate_fold_importance(folds)` / `build_importance_artifact(folds)` / `read_aggregated_importance(payload)` | Gain-map wrapper (0.0-filled), cross-fold mean +/- std aggregation, and the `feature_importance.json` write/read pair. |

## Layout

| File | Role |
| --- | --- |
| `metrics_aggregator.py` | `AggregateStats`, `aggregate_folds`, `_mean_std_ci` (IID percentile bootstrap over fold means). |
| `ranking.py` | `RankingMetric` (StrEnum) + `rank_strategies` (stable mergesort tie-break). |
| `significance.py` | Stationary bootstrap (Politis-Romano), Diebold-Mariano (HLN-corrected), deflated Sharpe (Bailey-López de Prado); result types `BootstrapCI`, `DMResult`, `DeflatedSharpe` + enums `DMLoss`, `DMDirection`. |
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
from src.analysis.ranking import RankingMetric, rank_strategies
from src.orchestration.run_loader import load_experiment_result

result = load_experiment_result("experiment_results/runs/<exp_id>")
stats = aggregate_folds(result.folds)
print(stats.sharpe_mean, stats.sharpe_ci95_low, stats.sharpe_ci95_high)

# Two-strategy ranking
ranking = rank_strategies({"A": stats, "B": stats}, by=RankingMetric.SHARPE)
print(ranking.to_string(index=False))
```

## Cross-links

- Drives `src/visualization/comparison_reporter.py` (consumes
  `rank_strategies` output).
- Powers `src/orchestration/comparison.py` pairwise significance and
  every HPO objective in `src/optimization/objectives.py`.
- Fold record shape is owned by `src/orchestration/types.py::FoldRecord`.
