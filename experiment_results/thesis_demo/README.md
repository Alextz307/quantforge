# `experiment_results/thesis_demo/`

End-to-end pipeline smoke. The point of `make thesis-demo` is to prove
that a fresh checkout can run **data → walk-forward → engine → metrics →
reporters → comparison → regime split** offline, on cached data. It is
**not** an empirical claim. Strategy parameters are not tuned, the
walk-forward window is short, and a single regime detector is exercised.
The comprehensive empirical study will land separately under
`experiment_results/studies/`.

## Layout

| Path | Status | What's there |
| --- | --- | --- |
| `sample/` | Tracked | Curated artifacts captured from one demo run, kept in repo so a casual reader sees the shape of the output without running anything. |
| `runs/<exp_id>/` | Gitignored | Fresh per-invocation walk-forward bundle for the single-strategy `experiment run` step (AdaptiveBollinger). |
| `comparisons/pipeline_compare/` | Gitignored | Fresh per-invocation cross-strategy comparison (AdaptiveBollinger vs MomentumGatekeeper). |
| `regime_reports/pipeline_regime/` | Gitignored | Fresh per-invocation regime split of the AdaptiveBollinger run (`bull_bear_200ma` detector). |

Everything under the runtime dirs (`runs/`, `comparisons/`,
`regime_reports/`) is wiped at the start of each `make thesis-demo` so
the demo is repeatable; the committed `sample/` is left alone.

## `sample/` index

| File | Source | What it shows |
| --- | --- | --- |
| `plots/run_equity_curves.png` | `runs/<id>/plots/equity_curves.*` | Per-fold equity curves of the AdaptiveBollinger walk-forward. |
| `plots/run_fold_stability.png` | `runs/<id>/plots/fold_stability.*` | Cross-fold scatter of Sharpe vs max-drawdown — quick read on whether the strategy is fold-stable. |
| `tables/run_metrics_summary.tex` | `runs/<id>/tables/metrics_summary.tex` | Booktabs table of per-fold metrics. |
| `run_metrics.json` | `runs/<id>/metrics.json` | Aggregated metrics (Sharpe / Sortino / Calmar / max-DD with 95% CIs). |
| `plots/compare_equity_overlay.png` | `comparisons/pipeline_compare/plots/equity_overlay.*` | Two-strategy equity overlay normalised to 1.0 at fold start. |
| `tables/compare_ranking.tex` | `comparisons/pipeline_compare/tables/ranking.tex` | Ranked Sharpe / Sortino / Calmar across the two strategies. |
| `tables/compare_pairwise_significance.tex` | `comparisons/pipeline_compare/tables/pairwise_significance.tex` | Paired stationary-bootstrap CI on the Sharpe differential. |
| `plots/regime_metric_heatmap.png` | `regime_reports/pipeline_regime/plots/regime_metric_heatmap.*` | Bull vs bear performance per metric. |
| `plots/regime_timeline.png` | `regime_reports/pipeline_regime/plots/regime_timeline.*` | When each regime occurred over the test window. |
| `tables/regime_summary.tex` | `regime_reports/pipeline_regime/tables/regime_summary.tex` | Per-regime aggregate metrics with bootstrap dispersion. |

The captions in the regime LaTeX tables reference the original
experiment_id (timestamp + random suffix) of the committed sample run.
That id will not exist after a re-run — the committed artifacts are
historical snapshots, not live links.

## Reproducing

```bash
make thesis-demo
```

Reads the cached `tests/fixtures/SPY.parquet`, runs the three CLI
invocations, and writes fresh artifacts into the runtime subdirs. Total
wall time: under one minute on a 2024 laptop. Exit status non-zero if
any step fails.

To regenerate the *committed* `sample/` artifacts from a fresh demo run
(when the strategy code or the report layout changes), re-run
`make thesis-demo` and copy the relevant subset by hand —
`scripts/` does not currently expose a one-shot "refresh sample"
command.
