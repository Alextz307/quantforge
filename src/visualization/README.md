# `src/visualization/`

Thesis-ready report renderers. Six sibling reporters consume the
in-memory result types from `src/orchestration/types.py` (and the
cross-leg `ConsolidatedStudyReport` from
`src/orchestration/study_report.py`) and emit PNG + SVG plots plus
booktabs LaTeX tables under each report's `plots/` and `tables/`
subdirectories.

## Public surface

| Symbol | Role |
| --- | --- |
| `StrategyReporter.generate_full_report(result, out_dir, *, publish_label=None)` | Per-experiment artefacts: equity curves overlay, fold-stability scatter, per-fold metrics LaTeX. |
| `ComparisonReporter.generate_full_report(report, folds_by_strategy, out_dir, *, publish_label=None)` | Cross-strategy artefacts: ranking LaTeX, pairwise-significance LaTeX, equity overlay normalised at 1.0, optional strategy × regime heatmap + LaTeX table when `report.per_strategy_per_regime_stats` is populated, comparison `manifest.json`. |
| `RegimeReporter.generate_full_report(report, out_dir, *, publish_label=None)` | Per-regime artefacts: regime summary LaTeX, regime × metric heatmap, regime timeline tape, regime `manifest.json`. |
| `HPOReporter.generate_full_report(study, out_dir)` | Optuna-study artefacts: convergence curve, parameter-importance bars, top-trials LaTeX. |
| `HoldoutEvalReporter.generate_full_report(result, out_dir, *, publish_label=None)` | One-shot holdout-eval artefacts: holdout-metrics LaTeX (two-column metric · value), normalised holdout-equity curve. |
| `StudyReportReporter.generate_full_report(report, out_dir, *, publish_label=None)` | Cross-leg consolidated artefacts: master / per-universe / per-regime / holdout rankings (`.tex` + `.csv`), strategy × universe + strategy × regime heatmaps, dev-vs-holdout scatter, per-universe equity-overlay copies, regime-timeline copies, per-leg holdout-equity copies. |
| `build_booktabs_table(df, *, caption, label, ...)` / `write_booktabs_table` | Single LaTeX styling entry point — every reporter routes through here. |
| `validate_publish_label(slug)` | Shared regex gate for the citation slug accepted by every reporter's `publish_label` kwarg. |
| `save_png_and_svg(fig, png_path)` | PNG + SVG twin-write helper; pinned `FIGURE_WIDTH_IN`, `FIGURE_HEIGHT_IN`, `FIGURE_DPI`. |
| `PLOTS_SUBDIR`, `TABLES_SUBDIR`, `MANIFEST_FILENAME` | Shared on-disk layout constants. |

## Layout

| File | Role |
| --- | --- |
| `plots.py` | Pins matplotlib's Agg backend before `pyplot` is imported anywhere; figure geometry constants; `save_png_and_svg` helper. |
| `latex.py` | `build_booktabs_table` + `write_booktabs_table` (pandas → booktabs LaTeX). |
| `strategy_reporter.py` | `StrategyReporter` (one experiment). |
| `comparison_reporter.py` | `ComparisonReporter` (multi-strategy ranking + significance). |
| `regime_reporter.py` | `RegimeReporter` (per-regime aggregation + timeline). |
| `hpo_reporter.py` | `HPOReporter` (Optuna study convergence + importance + top trials). |
| `holdout_eval_reporter.py` | `HoldoutEvalReporter` (one-shot holdout-eval table + equity plot). |
| `study_report_reporter.py` | `StudyReportReporter` (cross-leg consolidated tables + heatmaps + dev-vs-holdout scatter + per-leg artefact copies). |

## Conventions enforced here

- **Agg backend, before pyplot.** `plots.py` sets `matplotlib.use("Agg")`
  unconditionally before importing pyplot — required for headless CI
  runs and consistent rendering across macOS / Linux.
- **Twin write.** Every plot writes both PNG (for inline preview /
  thesis embed) and SVG (for redrawing without resampling).
- **Equity overlays normalise to 1.0.** Per-fold and per-strategy
  overlays are normalised at series start so the eye reads
  fold-internal / strategy-internal performance, not the compounding
  baseline.
- **booktabs everywhere.** All `.tex` outputs use the booktabs style
  via `build_booktabs_table`; caption/label conventions live in one
  place.
- **Stable citation slugs via `publish_label`.** Every reporter that
  emits LaTeX accepts a `publish_label: str | None` kwarg. When set,
  it replaces the volatile `experiment_id` / `out_name` in
  `\caption` / `\label` so a thesis-prose `\ref{tab:metrics_…}` keeps
  resolving across reruns. The slug regex (`validate_publish_label`)
  rejects anything that breaks LaTeX (spaces, braces, `%`, `#`).
- **`manifest.json` per report bundle.** Each reporter writes a small
  identity sidecar (out_name, timestamp, git sha, key stats) so a
  reader can inspect the bundle without re-running.

## Snippet

```python
from pathlib import Path

from src.orchestration.regime_run import load_run_from_disk
from src.visualization.strategy_reporter import StrategyReporter

run = load_run_from_disk(Path("experiment_results/runs/<exp_id>"))
StrategyReporter().generate_full_report(
    result=run,  # ExperimentResult-shaped (folds + manifest)
    out_dir=Path("experiment_results/runs/<exp_id>"),
)
```

## Cross-links

- Consumes `AggregateStats` from `src/analysis/metrics_aggregator.py`,
  ranking DataFrames from `src/analysis/ranking.py`, and `RegimeReport`
  / `StrategyComparisonReport` / `ExperimentResult` from
  `src/orchestration/types.py`.
- Benchmark reporting under `src/benchmarking/reporter.py` reuses
  these figure geometry / LaTeX constants — keep them in sync if
  changed.
