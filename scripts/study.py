"""
Click adapter for the empirical-study orchestrator.

Two subcommands under the ``experiment study`` group:

* ``study run``           Drive the full sweep: tune -> run -> holdout-eval
                          per leg, then per-universe cross-strategy
                          compare. Resumable via ``study_state.json`` under
                          ``<study_dir>/``.
* ``study report``        Walk a completed study directory and consolidate
                          per-leg artifacts into ``<study_dir>/{tables,
                          plots,manifest.json}``. Read-only with respect
                          to the per-leg tree.

Logic lives in ``src/orchestration/study.py`` and
``src/orchestration/study_report.py``; this module is a thin flag-parser
+ error-wrapper, mirroring the rest of ``experiment``'s subcommand layout.
"""

from __future__ import annotations

from pathlib import Path

import click
from pydantic import ValidationError

from scripts._attribution import attribute_via_username, default_username
from src.core.exceptions import LeakageError
from src.core.logging import attach_cli_log_file
from src.orchestration.study import run_study
from src.orchestration.study_report import consolidate_study
from src.visualization.study_report_reporter import StudyReportReporter
from webapp.backend.app.schemas.jobs import JobKind

DEFAULT_STORE_ROOT = Path("experiment_results")


@click.group("study")
def study() -> None:
    """
    Empirical-study orchestrator (cross-strategy x cross-universe sweep).
    """


@study.command("run")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a StudySpec YAML.",
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Override the experiment_results/ directory (artifacts root).",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Ignore is_complete markers; re-run every leg from scratch.",
)
@click.option(
    "--only-leg",
    "only_legs",
    multiple=True,
    help=(
        "Filter: only run legs whose leg_id matches. Repeatable. Useful "
        "for re-running a single failed leg without restarting the sweep."
    ),
)
@click.option(
    "--skip-compares",
    is_flag=True,
    default=False,
    help="Run per-leg workflows only; skip the per-universe cross-strategy compares.",
)
@click.option(
    "--skip-holdout-eval",
    is_flag=True,
    default=False,
    help="Skip the holdout-eval step on every leg (early-iteration knob).",
)
@click.option(
    "--user",
    "username",
    default=None,
    help="Webapp username to attribute this study to (see ``experiment run --user``).",
)
def run_cmd(
    spec_path: Path,
    store_root: Path,
    force_rerun: bool,
    only_legs: tuple[str, ...],
    skip_compares: bool,
    skip_holdout_eval: bool,
    username: str | None,
) -> None:
    """
    Drive the empirical study end-to-end.

    Per-leg failures are isolated (one bad leg does not abort the
    sweep); rerun the same command to retry only the failed legs.
    """

    with attach_cli_log_file(store_root, "study_run") as log_path:
        click.echo(
            f"running study from spec '{spec_path}' "
            f"(store_root={store_root}, force_rerun={force_rerun}, "
            f"only_legs={list(only_legs) or 'all'}, "
            f"skip_compares={skip_compares}, skip_holdout_eval={skip_holdout_eval}) "
            f"-> log: {log_path}"
        )
        try:
            result = run_study(
                spec_path,
                store_root=store_root,
                force_rerun=force_rerun,
                only_legs=list(only_legs) if only_legs else None,
                skip_compares=skip_compares,
                skip_holdout_eval=skip_holdout_eval,
            )
        except LeakageError as e:
            raise click.ClickException(f"leakage tripwire fired: {e}") from e
        except (ValidationError, FileNotFoundError, ValueError, RuntimeError) as e:
            raise click.ClickException(f"study run failed: {e}") from e

        click.echo(f"study_dir:    {result.study_dir}")
        click.echo(f"completed:    {result.n_legs_completed}")
        click.echo(f"failed:       {result.n_legs_failed}")
        click.echo(f"skipped:      {result.n_legs_skipped}")
        click.echo(f"compares:     {result.n_compares_done}")

        attribute_via_username(
            username=username or default_username(),
            kind=JobKind.STUDY,
            experiment_id=result.study_dir.name,
            command="study run",
        )

        if result.n_legs_failed > 0:
            raise click.ClickException(
                f"{result.n_legs_failed} leg(s) failed - inspect study_state.json "
                f"under {result.study_dir} for per-leg error messages, then rerun "
                f"the same command to retry."
            )


@study.command("report")
@click.option(
    "--study-dir",
    "study_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to a completed study directory (contains study_state.json).",
)
@click.option(
    "--publish-label",
    "publish_label",
    default=None,
    type=str,
    help=(
        "Optional slug used in every emitted LaTeX caption / label. "
        "Pass when committed artifacts are referenced from prose so "
        "re-running the consolidator doesn't churn citation slugs."
    ),
)
def report_cmd(study_dir: Path, publish_label: str | None) -> None:
    """
    Consolidate a completed study's per-leg artifacts into one tree.

    Reads ``runs/``, ``holdout_evals/``, and ``comparisons/`` under
    ``--study-dir``; writes ``<study-dir>/{manifest.json,tables/,plots/}``
    with cross-leg rankings, heatmaps, and per-universe equity / holdout
    plot copies. Read-only with respect to the per-leg tree - safe to
    rerun.
    """

    with attach_cli_log_file(study_dir, "study_report") as log_path:
        click.echo(f"consolidating study at {study_dir} -> log: {log_path}")
        try:
            report = consolidate_study(study_dir)
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"consolidation failed: {e}") from e

        StudyReportReporter().generate_full_report(report, study_dir, publish_label=publish_label)
        click.echo(f"study_name:       {report.study_name}")
        click.echo(f"strategies:       {len(report.strategies)}")
        click.echo(f"universes:        {len(report.universes)}")
        click.echo(f"completed legs:   {len(report.per_leg_aggregate)}")
        click.echo(f"incomplete legs:  {len(report.incomplete_leg_ids)}")
        click.echo(f"legs w/ holdout:  {len(report.per_leg_holdout)}")
        click.echo(f"output:           {study_dir}")
