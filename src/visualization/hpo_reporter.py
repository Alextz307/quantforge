"""HPO study report generator.

Consumes an :class:`optuna.Study` and writes thesis-ready artifacts
under ``<study_dir>/plots/`` + ``<study_dir>/tables/``:

* ``plots/convergence.png/svg`` — per-trial objective value + rolling
  best-so-far line. The story is "did the study converge, and how
  quickly did it find the best region".
* ``plots/param_importance.png/svg`` — horizontal bar chart from
  ``optuna.importance.get_param_importances``. Skipped (with an info log)
  for studies with <2 completed trials where the importance computation
  is undefined.
* ``tables/top_trials.tex`` — booktabs LaTeX ranking the top-N trials
  by objective value, one row per trial with number + value + key
  params. Row count capped at :data:`_TOP_TRIALS_N`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.trial import TrialState

from src.core.logging import get_logger
from src.visualization.latex import write_booktabs_table
from src.visualization.plots import (
    FIGURE_DPI,
    FIGURE_HEIGHT_IN,
    FIGURE_WIDTH_IN,
    save_png_and_svg,
)

_logger = get_logger(__name__)

_PLOTS_SUBDIR = "plots"
_TABLES_SUBDIR = "tables"
_CONVERGENCE_FILENAME = "convergence.png"
_IMPORTANCE_FILENAME = "param_importance.png"
_TOP_TRIALS_FILENAME = "top_trials.tex"
_TOP_TRIALS_N = 10
# Minimum completed trials for param-importance to be meaningful.
# get_param_importances raises below this; we skip with an info log.
_MIN_TRIALS_FOR_IMPORTANCE = 2


class HPOReporter:
    """Generate the HPO-study report bundle."""

    def generate_full_report(self, study: optuna.Study, out_dir: Path) -> Path:
        """Write every artifact under ``out_dir/{plots,tables}/`` and return ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / _PLOTS_SUBDIR
        tables_dir = out_dir / _TABLES_SUBDIR

        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        _logger.info(
            "generating HPO report: study=%s trials=%d completed=%d",
            study.study_name,
            len(study.trials),
            len(completed),
        )

        if completed:
            self._plot_convergence(completed, plots_dir / _CONVERGENCE_FILENAME)
            write_booktabs_table(
                self._build_top_trials_df(completed),
                tables_dir / _TOP_TRIALS_FILENAME,
                caption=f"Top {_TOP_TRIALS_N} trials — study {study.study_name}",
                label=f"tab:hpo_{study.study_name}",
            )

        if len(completed) >= _MIN_TRIALS_FOR_IMPORTANCE:
            self._plot_param_importance(study, plots_dir / _IMPORTANCE_FILENAME)
        else:
            _logger.info(
                "skipping param-importance plot: need ≥%d completed trials, have %d",
                _MIN_TRIALS_FOR_IMPORTANCE,
                len(completed),
            )

        return out_dir

    def _plot_convergence(self, completed: list[optuna.trial.FrozenTrial], path: Path) -> None:
        trial_numbers = [t.number for t in completed]
        values = [t.value for t in completed if t.value is not None]
        # best-so-far: running max over trial order
        best_so_far: list[float] = []
        running_best = float("-inf")
        for v in values:
            running_best = max(running_best, v)
            best_so_far.append(running_best)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        ax.scatter(trial_numbers, values, alpha=0.5, label="trial value", zorder=2)
        ax.plot(
            trial_numbers,
            best_so_far,
            linestyle="--",
            color="tab:red",
            label="best-so-far",
            zorder=3,
        )
        ax.set_xlabel("trial number")
        ax.set_ylabel("objective value")
        ax.set_title("HPO convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_png_and_svg(fig, path)
        plt.close(fig)

    def _plot_param_importance(self, study: optuna.Study, path: Path) -> None:
        importances = get_param_importances(study)
        if not importances:
            return
        # Sorted descending by importance — bar chart reads top-to-bottom as
        # most-to-least important
        names = list(importances.keys())
        scores = list(importances.values())

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI)
        ax.barh(range(len(names)), scores, color="tab:blue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("importance")
        ax.set_title("Parameter importance (fANOVA)")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        save_png_and_svg(fig, path)
        plt.close(fig)

    def _build_top_trials_df(self, completed: list[optuna.trial.FrozenTrial]) -> pd.DataFrame:
        ranked = sorted(
            completed,
            key=lambda t: t.value if t.value is not None else float("-inf"),
            reverse=True,
        )[:_TOP_TRIALS_N]
        rows = [
            {
                "trial": t.number,
                "value": t.value,
                "params": _format_params(t.params),
            }
            for t in ranked
        ]
        return pd.DataFrame(rows)


def _format_params(params: dict[str, object]) -> str:
    """Compact key=val,key=val rendering for a LaTeX table cell."""
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return ", ".join(parts)
