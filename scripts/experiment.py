"""
Command-line driver for the experiment runner.

Subcommands:

* ``run``          Fetch data, walk-forward, persist artifacts to
                   ``experiment_results/runs/<experiment_id>/``.
* ``tune``         Drive an Optuna study over a config's HPO space,
                   persisting to ``experiment_results/hpo/<study>/``.
* ``compare``      Run N configs, rank them, compute pairwise Sharpe
                   significance, write the bundle to
                   ``experiment_results/comparisons/<out_name>/``.
* ``holdout-eval`` Take a completed run or HPO study, refit the strategy
                   on the full dev region, and evaluate once on the
                   reserved holdout window. Writes the honest OOS bundle
                   to ``experiment_results/holdout_evals/<out_name>/``.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import click
from pydantic import ValidationError

from scripts._attribution import attribute_via_username, default_username
from scripts.study import study
from src.core import json_io
from src.core.config import (
    ExperimentConfig,
    load_experiment_config,
)
from src.core.config_overrides import apply_overrides
from src.core.constants import IMPORTANCE_REPRODUCTION_ABS_TOL, IMPORTANCE_REPRODUCTION_RTOL
from src.core.exceptions import LeakageError
from src.core.hpo_config import HPOConfig, load_hpo_config
from src.core.logging import CLI_LOG_FORMAT, attach_cli_log_file, get_logger
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    EXPERIMENT_METRICS_JSON,
    FEATURE_IMPORTANCE_DIVERGED_JSON,
    FEATURE_IMPORTANCE_JSON,
    HOLDOUT_EVALS_SUBDIR,
    HPO_SUBDIR,
    RUNS_SUBDIR,
    read_experiment_manifest,
)
from src.core.types import JobKind
from src.orchestration.builder import build_experiment
from src.orchestration.comparison import SignificanceTest, run_comparison
from src.orchestration.experiment import RunOptions
from src.orchestration.holdout_eval import resolve_source, run_holdout_eval
from src.orchestration.run_loader import (
    load_experiment_config_from_run,
    load_experiment_result,
    strategy_supports_feature_importance,
)
from src.orchestration.types import ExperimentResult

# ``optuna`` is deferred into ``tune_cmd`` so it does not load on every CLI
# invocation. Visualization reporters are also lazy for symmetry; matplotlib
# already lands at top-level via the ``pmdarima`` model registry side-effect.

DEFAULT_STORE_ROOT = Path("experiment_results")

logger = get_logger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Python logging level for the runner.",
)
def cli(log_level: str) -> None:
    """
    Quant-engine experiment orchestrator.
    """

    logging.basicConfig(level=log_level.upper(), format=CLI_LOG_FORMAT)


@cli.command("run")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to an ExperimentConfig YAML.",
)
@click.option(
    "--name",
    default=None,
    help="Override the config's `name` field (does not affect experiment_id).",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Override the config's `seed` field for this invocation.",
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Override the experiment_results/ directory.",
)
@click.option(
    "--report/--no-report",
    "write_report",
    default=True,
    help="Generate strategy_reporter artifacts (plots + LaTeX tables) under the run dir.",
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=False,
    help="Show a tqdm progress bar over the walk-forward fold loop (TTY only).",
)
@click.option(
    "--checkpoint/--no-checkpoint",
    "checkpoint",
    default=False,
    help=(
        "Per-fold mid-fit best-state checkpoints under <run>/checkpoints/fold_N/. "
        "Useful for long LSTM/XGBoost fits where Ctrl+C should leave the best-so-far "
        "weights recoverable. HPO trials never checkpoint regardless of this flag."
    ),
)
@click.option(
    "--feature-importance/--no-feature-importance",
    "feature_importance",
    default=False,
    help=(
        "Compute out-of-sample feature importance per fold (permutation + "
        "XGBoost gain) and write feature_importance.json under the run dir. "
        "Off by default; the study turns it on for its final per-leg runs."
    ),
)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help=(
        "Dotted-path override applied to the loaded config (e.g. "
        "data.tickers=[QQQ]). Value parsed as YAML; intermediate keys "
        "must exist. Repeatable."
    ),
)
@click.option(
    "--publish-label",
    "publish_label",
    default=None,
    help=(
        "Stable LaTeX caption + label slug for the metrics table. When "
        "set, replaces the volatile experiment_id in \\caption / \\label "
        "so thesis prose can \\ref the table across reruns. Allowed "
        "chars: letters, digits, _, -, :."
    ),
)
@click.option(
    "--user",
    "username",
    default=None,
    help=(
        "Webapp username to attribute this artifact to. Defaults to the OS "
        "user (``getpass.getuser()``). Auto-creates the webapp account on "
        "first use when stdin is a TTY (prompts for a password); errors "
        "out in non-interactive contexts pointing at scripts/create_user.py."
    ),
)
def run_cmd(
    config_path: Path,
    name: str | None,
    seed: int | None,
    store_root: Path,
    write_report: bool,
    progress: bool,
    checkpoint: bool,
    feature_importance: bool,
    overrides: tuple[str, ...],
    publish_label: str | None,
    username: str | None,
) -> None:
    """
    Execute a single walk-forward experiment end-to-end.
    """

    with attach_cli_log_file(store_root, "experiment_run") as log_path:
        try:
            cfg = load_experiment_config(config_path)
        except (ValidationError, FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"failed to load config {config_path}: {e}") from e

        if name is not None or seed is not None:
            cfg = _override_experiment(cfg, name=name, seed=seed)
        cfg = _apply_dotted_overrides(cfg, overrides)

        try:
            experiment = build_experiment(cfg)
        except (ValidationError, ValueError) as e:
            raise click.ClickException(f"failed to build experiment: {e}") from e

        click.echo(
            f"running experiment '{cfg.name}' ({cfg.strategy.name} x {cfg.data.tickers[0]}) "
            f"-> log: {log_path}"
        )
        try:
            result = experiment.run(
                RunOptions(
                    store_root=store_root,
                    write_report=write_report,
                    progress=progress,
                    checkpoint=checkpoint,
                    publish_label=publish_label,
                    compute_feature_importance=feature_importance,
                )
            )
        except LeakageError as e:
            raise click.ClickException(f"leakage tripwire fired: {e}") from e
        except (NotImplementedError, ValueError, RuntimeError) as e:
            raise click.ClickException(f"experiment failed: {e}") from e

        run_dir = store_root / RUNS_SUBDIR / result.experiment_id
        click.echo(f"experiment_id: {result.experiment_id}")
        click.echo(f"artifacts:    {run_dir}")
        click.echo(f"folds:        {len(result.folds)}")

        attribute_via_username(
            username=username or default_username(),
            kind=JobKind.RUN,
            experiment_id=result.experiment_id,
            command="experiment run",
        )


@cli.command("tune")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to an ExperimentConfig YAML (the base config HPO searches over).",
)
@click.option(
    "--hpo-config",
    "hpo_config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to an HPOConfig YAML (study_name, n_trials, sampler, objective, ...).",
)
@click.option(
    "--trials",
    "n_trials_override",
    default=None,
    type=click.IntRange(min=1),
    help="Override hpo.n_trials for this invocation.",
)
@click.option(
    "--n-jobs",
    "n_jobs_override",
    default=None,
    type=int,
    help="Override hpo.n_jobs. Pass -1 for os.cpu_count().",
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Override the experiment_results/ directory.",
)
@click.option(
    "--report/--no-report",
    "write_report",
    default=True,
    help="Generate HPO convergence + top-trials report after the study finishes.",
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=False,
    help="Show Optuna's per-trial progress bar (TTY only).",
)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help=(
        "Dotted-path override applied to the experiment config (e.g. "
        "data.tickers=[QQQ]). Value parsed as YAML; intermediate keys "
        "must exist. Repeatable. Does not modify the HPO config - use "
        "--trials / --n-jobs for that."
    ),
)
@click.option(
    "--user",
    "username",
    default=None,
    help="Webapp username to attribute this study to (see ``experiment run --user``).",
)
def tune_cmd(
    config_path: Path,
    hpo_config_path: Path,
    n_trials_override: int | None,
    n_jobs_override: int | None,
    store_root: Path,
    write_report: bool,
    progress: bool,
    overrides: tuple[str, ...],
    username: str | None,
) -> None:
    """
    Run an Optuna study over an ExperimentConfig's hyperparameter space.

    The study is persisted to a SQLite file under
    ``<store_root>/hpo/<study_name>/optuna_study.db`` - re-running with
    the same ``--config`` + ``--hpo-config`` resumes from the last
    completed trial (Optuna's semantics: ``n_trials`` is the number of
    NEW trials to run each invocation, not a cap on total trials).
    """

    # Defer optuna so unrelated subcommands don't pay its import cost.
    from src.optimization.tuner import StrategyTuner
    from src.visualization.hpo_reporter import HPOReporter

    with attach_cli_log_file(store_root, "experiment_tune") as log_path:
        try:
            experiment_cfg = load_experiment_config(config_path)
        except (ValidationError, FileNotFoundError, ValueError) as e:
            raise click.ClickException(
                f"failed to load experiment config {config_path}: {e}"
            ) from e
        try:
            hpo_cfg = load_hpo_config(hpo_config_path)
        except (ValidationError, FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"failed to load hpo config {hpo_config_path}: {e}") from e

        experiment_cfg = _apply_dotted_overrides(experiment_cfg, overrides)
        hpo_cfg = _apply_hpo_overrides(hpo_cfg, n_trials=n_trials_override, n_jobs=n_jobs_override)

        tuner = StrategyTuner(
            experiment_cfg=experiment_cfg,
            hpo_cfg=hpo_cfg,
            store_root=store_root,
        )

        click.echo(
            f"tuning '{experiment_cfg.strategy.name}' on study '{hpo_cfg.study_name}' "
            f"for {hpo_cfg.n_trials} trial(s) (n_jobs={hpo_cfg.n_jobs}) -> log: {log_path}"
        )
        try:
            study = tuner.run(progress=progress)
        except LeakageError as e:
            raise click.ClickException(f"leakage tripwire fired: {e}") from e
        except (NotImplementedError, ValueError, RuntimeError) as e:
            raise click.ClickException(f"tuning failed: {e}") from e

        if write_report:
            HPOReporter().generate_full_report(study, tuner.study_dir)

        study_dir = store_root / HPO_SUBDIR / hpo_cfg.study_name
        click.echo(f"study_name:  {study.study_name}")
        click.echo(f"artifacts:   {study_dir}")
        click.echo(f"trials:      {len(study.trials)}")
        try:
            best = study.best_trial
            click.echo(f"best_value:  {best.value}")
            click.echo(f"best_trial:  {best.number}")
        except ValueError:
            click.echo("best_value:  n/a (no completed trials)")

        attribute_via_username(
            username=username or default_username(),
            kind=JobKind.TUNE,
            experiment_id=hpo_cfg.study_name,
            command="experiment tune",
        )


@cli.command("compare")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to an ExperimentConfig YAML. Pass multiple times for a multi-strategy compare.",
)
@click.option(
    "--out-name",
    required=True,
    help="Directory name under experiment_results/comparisons/ for the report bundle.",
)
@click.option(
    "--significance-test",
    type=click.Choice([m.value for m in SignificanceTest], case_sensitive=False),
    default=SignificanceTest.BOOTSTRAP.value,
    help="Pairwise Sharpe-differential test (paired stationary bootstrap) or skip.",
)
@click.option(
    "--n-jobs",
    default=1,
    type=int,
    help="1 = in-process sequential; >1 fans out via ProcessPoolExecutor.",
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Override the experiment_results/ directory.",
)
@click.option(
    "--report/--no-report",
    "write_report",
    default=True,
    help="Write ranking.tex, pairwise_significance.tex, and equity overlay plot.",
)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help=(
        "Dotted-path override applied to every --config in turn (e.g. "
        "data.tickers=[QQQ]). Value parsed as YAML; intermediate keys "
        "must exist. Repeatable."
    ),
)
@click.option(
    "--reuse-runs",
    "reuse_runs",
    default=None,
    help=(
        "Comma-separated list of completed run directories (one per "
        "--config in matching order). When set, the per-strategy "
        "walk-forward is skipped and ranking + bootstrap run against "
        "the prior fold artifacts."
    ),
)
@click.option(
    "--publish-label",
    "publish_label",
    default=None,
    help=(
        "Stable LaTeX caption + label slug for ranking and pairwise "
        "tables. When set, replaces out_name in every emitted "
        "\\caption / \\label so a re-run committed under a different "
        "--out-name still \\refs the original prose citation. Allowed "
        "chars: letters, digits, _, -, :."
    ),
)
@click.option(
    "--user",
    "username",
    default=None,
    help="Webapp username to attribute this comparison to (see ``experiment run --user``).",
)
def compare_cmd(
    config_paths: tuple[Path, ...],
    out_name: str,
    significance_test: str,
    n_jobs: int,
    store_root: Path,
    write_report: bool,
    overrides: tuple[str, ...],
    reuse_runs: str | None,
    publish_label: str | None,
    username: str | None,
) -> None:
    """
    Run N configs, rank, optionally test pairwise significance.

    The comparison directory is ``<store_root>/comparisons/<out_name>/``.
    Each strategy's walk-forward results land under ``runs/`` inside
    that directory (instead of the top-level ``experiment_results/runs/``)
    so the comparison bundle is self-contained.
    """

    from src.visualization.comparison_reporter import ComparisonReporter

    if len(config_paths) < 2:
        raise click.ClickException(
            f"compare needs at least 2 --config paths, got {len(config_paths)}; "
            f"pass the option multiple times."
        )

    with attach_cli_log_file(store_root, "experiment_compare") as log_path:
        configs: list[ExperimentConfig] = []
        for path in config_paths:
            try:
                cfg = load_experiment_config(path)
            except (ValidationError, FileNotFoundError, ValueError) as e:
                raise click.ClickException(f"failed to load config {path}: {e}") from e
            configs.append(_apply_dotted_overrides(cfg, overrides))

        reused_results = _load_reused_runs(reuse_runs, n_configs=len(configs))

        sig = SignificanceTest(significance_test)
        click.echo(
            f"comparing {len(configs)} strategies under '{out_name}' "
            f"(n_jobs={n_jobs}, significance={sig.value}, "
            f"reuse={'yes' if reused_results is not None else 'no'}) -> log: {log_path}"
        )
        try:
            report, folds_by_strategy = run_comparison(
                configs,
                out_name=out_name,
                store_root=store_root,
                n_jobs=n_jobs,
                significance_test=sig,
                reused_results=reused_results,
            )
        except LeakageError as e:
            raise click.ClickException(f"leakage tripwire fired: {e}") from e
        except (NotImplementedError, ValueError, RuntimeError) as e:
            raise click.ClickException(f"comparison failed: {e}") from e

        cmp_dir = store_root / COMPARISONS_SUBDIR / out_name
        if write_report:
            ComparisonReporter().generate_full_report(
                report,
                cmp_dir,
                folds_by_strategy=folds_by_strategy,
                publish_label=publish_label,
            )

        click.echo(f"out_name:   {report.out_name}")
        click.echo(f"artifacts:  {cmp_dir}")
        click.echo(f"strategies: {', '.join(report.per_strategy_stats.keys())}")
        if report.pairwise:
            n_sig = sum(1 for p in report.pairwise if p.significant)
            click.echo(f"pairwise:   {len(report.pairwise)} comparisons, {n_sig} significant")

        attribute_via_username(
            username=username or default_username(),
            kind=JobKind.COMPARE,
            experiment_id=report.out_name,
            command="experiment compare",
        )


@cli.command("holdout-eval")
@click.option(
    "--run-dir",
    "run_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "Source: a completed `experiment run` directory (must contain "
        "config.yaml + manifest.json with a non-null holdout_start). "
        "Mutually exclusive with --hpo-best."
    ),
)
@click.option(
    "--hpo-best",
    "hpo_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "Source: a completed `experiment tune` study directory (must "
        "contain best_config.yaml + at least one trial under "
        "trials_artifacts/runs/). Mutually exclusive with --run-dir."
    ),
)
@click.option(
    "--out-name",
    default=None,
    help=(
        "Directory name under <store-root>/holdout_evals/ for the bundle. "
        "Defaults to the source's basename (run id or study name)."
    ),
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Override the experiment_results/ directory.",
)
@click.option(
    "--report/--no-report",
    "write_report",
    default=True,
    help="Write the holdout-metrics LaTeX table and the holdout-equity plot.",
)
@click.option(
    "--publish-label",
    "publish_label",
    default=None,
    help=(
        "Stable LaTeX caption + label slug for the holdout-metrics table. "
        "When set, replaces source_id / out_name in \\caption / \\label "
        "so thesis prose can \\ref the table across reruns. Allowed "
        "chars: letters, digits, _, -, :."
    ),
)
@click.option(
    "--user",
    "username",
    default=None,
    help="Webapp username to attribute this holdout-eval to (see ``experiment run --user``).",
)
def holdout_eval_cmd(
    run_dir: Path | None,
    hpo_dir: Path | None,
    out_name: str | None,
    store_root: Path,
    write_report: bool,
    publish_label: str | None,
    username: str | None,
) -> None:
    """
    Refit on full dev, evaluate once on the reserved holdout - honest OOS.

    The source's manifest.json is the source of truth for the dev/holdout
    boundary timestamp and the data fingerprint; the command refuses on
    any drift. The strategy is refit from scratch on the FULL dev region
    (not reused from the source's last-fold state) so the OOS number
    reflects the strongest honest fit the framework can produce.
    """

    if (run_dir is None) == (hpo_dir is None):
        raise click.ClickException(
            "holdout-eval requires exactly one of --run-dir / --hpo-best; pass exactly one source."
        )

    with attach_cli_log_file(store_root, "holdout_eval") as log_path:
        try:
            source = resolve_source(run_dir=run_dir, hpo_dir=hpo_dir)
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"failed to resolve source: {e}") from e

        resolved_out_name = out_name if out_name is not None else source.source_id
        click.echo(
            f"holdout-eval: source={source.kind} '{source.source_id}' "
            f"-> out_name='{resolved_out_name}' -> log: {log_path}"
        )
        try:
            result, out_dir = run_holdout_eval(
                source=source,
                out_name=resolved_out_name,
                store_root=store_root,
            )
        except LeakageError as e:
            raise click.ClickException(f"leakage tripwire fired: {e}") from e
        except (
            FileNotFoundError,
            ValidationError,
            NotImplementedError,
            ValueError,
            RuntimeError,
        ) as e:
            raise click.ClickException(f"holdout-eval failed: {e}") from e

        if write_report:
            # Lazy: matplotlib's cold import is ~4s; --no-report skips it.
            from src.visualization.holdout_eval_reporter import HoldoutEvalReporter

            HoldoutEvalReporter().generate_full_report(result, out_dir, publish_label=publish_label)

        artifact_dir = store_root / HOLDOUT_EVALS_SUBDIR / resolved_out_name
        click.echo(f"out_name:        {result.out_name}")
        click.echo(f"artifacts:       {artifact_dir}")
        click.echo(f"holdout_start:   {result.holdout_start.isoformat()}")
        click.echo(f"dev_bars:        {result.n_dev_bars}")
        click.echo(f"holdout_bars:    {result.n_holdout_bars}")
        click.echo(f"sharpe:          {result.sharpe_ratio:+.4f}")
        click.echo(f"total_return:    {result.total_return:+.4f}")
        click.echo(f"max_drawdown:    {result.max_drawdown:+.4f}")

        attribute_via_username(
            username=username or default_username(),
            kind=JobKind.HOLDOUT,
            experiment_id=result.out_name,
            command="experiment holdout-eval",
        )


@cli.command("importance")
@click.option(
    "--run-dir",
    "run_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "A completed `experiment run` directory (must contain config.yaml + "
        "manifest.json + metrics.json). Its strategy must consume engineered "
        "features; rule-based strategies have no importance to compute."
    ),
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="The experiment_results/ directory a diverged re-run is saved under.",
)
@click.option(
    "--name",
    default=None,
    help=(
        "Override the re-run's `name` field. The webapp passes the job id so a "
        "diverged re-run's manifest.name resolves the job back to its new run."
    ),
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=False,
    help="Show a tqdm progress bar over the re-run's walk-forward fold loop (TTY only).",
)
@click.option(
    "--user",
    "username",
    default=None,
    help="Webapp username to attribute a diverged re-run to (see ``experiment run --user``).",
)
def importance_cmd(
    run_dir: Path,
    store_root: Path,
    name: str | None,
    progress: bool,
    username: str | None,
) -> None:
    """
    Compute feature importance for a finished run that lacks it.

    Importance is computed on each fold's freshly-trained model, so it cannot
    be derived from a finished run's artifacts - the models were discarded.
    This re-runs the run's exact config (same seed, cached data) with
    importance on, into a throwaway store, then decides where the result
    lands by comparing the re-run's aggregated metrics against the original:

    * **Reproduced** (deterministic training, e.g. XGBoost): the re-run's
      metrics match the original, so ``feature_importance.json`` is written
      into the original run dir, leaving every other artifact untouched.
    * **Diverged** (non-deterministic training, e.g. an accelerator-trained
      LSTM): the re-run's metrics differ, so attaching importance here would
      misrepresent the original's models. The re-run is saved as a separate
      run instead, tagged with its ``source_run``.
    """

    with attach_cli_log_file(store_root, "feature_importance") as log_path:
        try:
            cfg = load_experiment_config_from_run(run_dir)
        except (ValidationError, FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"failed to load run config from {run_dir}: {e}") from e

        if not strategy_supports_feature_importance(cfg.strategy.name):
            raise click.ClickException(
                f"strategy {cfg.strategy.name!r} consumes no engineered features, so it "
                f"produces no feature importance; nothing to compute for this run."
            )

        try:
            original_id = read_experiment_manifest(run_dir).experiment_id
            original_metrics = json_io.read_dict(run_dir / EXPERIMENT_METRICS_JSON)
        except FileNotFoundError as e:
            raise click.ClickException(f"run at {run_dir} is missing manifest/metrics: {e}") from e

        if name is not None:
            cfg = _override_experiment(cfg, name=name, seed=None)

        try:
            experiment = build_experiment(cfg)
        except (ValidationError, ValueError) as e:
            raise click.ClickException(f"failed to build experiment: {e}") from e

        click.echo(
            f"recomputing feature importance for '{original_id}' "
            f"({cfg.strategy.name} x {cfg.data.tickers[0]}) -> log: {log_path}"
        )

        final_id = original_id
        diverged_run_id: str | None = None
        with tempfile.TemporaryDirectory(prefix="importance_") as tmp:
            scratch_root = Path(tmp)
            try:
                result = experiment.run(
                    RunOptions(
                        store_root=scratch_root,
                        write_report=False,
                        progress=progress,
                        compute_feature_importance=True,
                    )
                )
            except LeakageError as e:
                raise click.ClickException(f"leakage tripwire fired: {e}") from e
            except (NotImplementedError, ValueError, RuntimeError) as e:
                raise click.ClickException(f"re-run failed: {e}") from e

            scratch_dir = scratch_root / RUNS_SUBDIR / result.experiment_id
            importance_path = scratch_dir / FEATURE_IMPORTANCE_JSON
            if not importance_path.is_file():
                raise click.ClickException(
                    "the re-run produced no feature_importance.json; every fold was "
                    "skipped (scored region too short after warmup) - widen the run's "
                    "date range so each fold has rows past the feature warmup."
                )

            try:
                scratch_metrics = json_io.read_dict(scratch_dir / EXPERIMENT_METRICS_JSON)
                reproduced = _metrics_reproduced(
                    original_metrics, scratch_metrics, IMPORTANCE_REPRODUCTION_RTOL
                )

                payload = json_io.read_dict(importance_path)
                payload["recomputed"] = True
                payload["reproduced"] = reproduced

                if reproduced:
                    json_io.write(run_dir / FEATURE_IMPORTANCE_JSON, payload)
                    # Importance now lives here; drop any earlier divergence pointer + run.
                    _discard_prior_diverged_run(run_dir, store_root, original_id)
                    (run_dir / FEATURE_IMPORTANCE_DIVERGED_JSON).unlink(missing_ok=True)
                    click.echo(
                        "reproduced the original run's metrics exactly; "
                        f"importance attached to '{original_id}' in place."
                    )
                else:
                    payload["source_run"] = original_id
                    json_io.write(importance_path, payload)
                    # Drop the run an earlier divergence saved, so re-divergence can't orphan it.
                    _discard_prior_diverged_run(run_dir, store_root, original_id)
                    dest = store_root / RUNS_SUBDIR / result.experiment_id
                    try:
                        shutil.copytree(scratch_dir, dest)
                    except (OSError, shutil.Error):
                        shutil.rmtree(dest, ignore_errors=True)
                        raise
                    final_id = result.experiment_id
                    diverged_run_id = final_id
                    # Record where importance landed so the run page links to it after a reload.
                    json_io.write(
                        run_dir / FEATURE_IMPORTANCE_DIVERGED_JSON,
                        {
                            "diverged_run_id": final_id,
                            "recomputed_at": datetime.now(UTC).isoformat(),
                        },
                    )
                    click.echo(
                        "re-run diverged from the original's metrics (training is "
                        "non-deterministic on this device); importance saved as a new "
                        f"run '{final_id}' so the original's metrics stay consistent."
                    )
            except (OSError, shutil.Error, ValueError) as e:
                raise click.ClickException(f"failed to persist recomputed importance: {e}") from e

        click.echo(f"experiment_id: {final_id}")

        # A reproduced backfill writes into the original run; attributing it here
        # would silently claim a run the caller only recomputed importance for.
        if diverged_run_id is not None:
            attribute_via_username(
                username=username or default_username(),
                kind=JobKind.RUN,
                experiment_id=diverged_run_id,
                command="experiment importance",
            )


def _discard_prior_diverged_run(run_dir: Path, store_root: Path, original_id: str) -> None:
    """
    Remove the run a previous diverged recompute saved importance under.

    A re-divergence (or a later reproduction) supersedes that earlier run;
    leaving it behind would strand an owned-but-unreferenced run dir. Guarded
    by the ``source_run`` tag so only an importance container this command
    created for ``original_id`` is ever deleted.
    """

    try:
        pointer = json_io.read_dict(run_dir / FEATURE_IMPORTANCE_DIVERGED_JSON)
    except (FileNotFoundError, ValueError):
        return
    prior_id = pointer.get("diverged_run_id")
    if not isinstance(prior_id, str) or not prior_id:
        return
    prior_dir = store_root / RUNS_SUBDIR / prior_id
    try:
        prior_payload = json_io.read_dict(prior_dir / FEATURE_IMPORTANCE_JSON)
    except (FileNotFoundError, ValueError):
        return
    if prior_payload.get("source_run") != original_id:
        return
    shutil.rmtree(prior_dir, ignore_errors=True)


def _metrics_reproduced(
    original: dict[str, object],
    candidate: dict[str, object],
    rtol: float,
) -> bool:
    """
    Whether two ``metrics.json`` payloads describe the same models.

    ``metrics.json`` is ``AggregateStats.to_dict()`` - a flat map of floats /
    ints that is a deterministic function of the fold values (the CI bootstrap
    is seeded internally), so identical fold models yield bit-identical
    aggregates. ``rtol`` absorbs benign run-to-run numerical noise (BLAS
    summation order, the hybrids' GARCH/ARMA MLE converging within its own
    optimizer tolerance) while a genuinely different fit moves the metrics far
    above it - see ``IMPORTANCE_REPRODUCTION_RTOL``. Two NaNs at the same key
    count as equal (a degenerate fold the original already carried), and the
    key sets must match.
    """

    if original.keys() != candidate.keys():
        # Can't compare different schemas, so treat as diverged rather than backfill blind.
        return False
    for key, original_value in original.items():
        candidate_value = candidate[key]
        if isinstance(original_value, bool) or isinstance(candidate_value, bool):
            # A bool vs a number is a type change (True == 1 in Python), so it
            # counts as divergent; bool-to-bool then compares by value.
            same_type = type(original_value) is type(candidate_value)
            if not same_type or original_value != candidate_value:
                return False
        elif isinstance(original_value, (int, float)) and isinstance(candidate_value, (int, float)):
            if math.isnan(original_value) and math.isnan(candidate_value):
                continue
            if not math.isclose(
                original_value,
                candidate_value,
                rel_tol=rtol,
                abs_tol=IMPORTANCE_REPRODUCTION_ABS_TOL,
            ):
                return False
        elif original_value != candidate_value:
            return False
    return True


def _apply_hpo_overrides(cfg: HPOConfig, *, n_trials: int | None, n_jobs: int | None) -> HPOConfig:
    """
    Rebuild the HPO config with CLI overrides, re-running validators.
    """

    if n_trials is None and n_jobs is None:
        return cfg
    payload = cfg.model_dump(mode="json")
    if n_trials is not None:
        payload["n_trials"] = n_trials
    if n_jobs is not None:
        if n_jobs == -1:
            resolved = os.cpu_count() or 1
        elif n_jobs < 1:
            raise click.ClickException(
                f"--n-jobs must be -1 (auto) or a positive int; got {n_jobs}."
            )
        else:
            resolved = n_jobs
        payload["n_jobs"] = resolved
    return HPOConfig.model_validate(payload)


def _override_experiment(
    cfg: ExperimentConfig, *, name: str | None, seed: int | None
) -> ExperimentConfig:
    """
    Rebuild ``cfg`` with CLI ``--name`` / ``--seed`` overrides applied.

    Re-runs the pydantic validator so an override never leaves the config
    in a half-validated state (e.g. an empty-string ``name`` from the CLI
    would be caught by ``min_length=1``).
    """

    payload = cfg.model_dump(mode="json")
    if name is not None:
        payload["name"] = name
    if seed is not None:
        payload["seed"] = seed
    return ExperimentConfig.model_validate(payload)


def _load_reused_runs(raw: str | None, *, n_configs: int) -> list[ExperimentResult] | None:
    """
    Resolve ``--reuse-runs <a,b,c>`` into the list ``run_comparison`` needs.

    Returns ``None`` when the flag is absent. Raises
    :class:`click.ClickException` on count mismatch or unreadable run
    dirs so the user sees the usual CLI-error framing instead of a raw
    traceback.
    """

    if raw is None:
        return None
    raw_paths = [p.strip() for p in raw.split(",") if p.strip()]
    if len(raw_paths) != n_configs:
        raise click.ClickException(
            f"--reuse-runs has {len(raw_paths)} path(s) but --config has "
            f"{n_configs}; pass one --reuse-runs path per --config in "
            f"matching order."
        )
    results: list[ExperimentResult] = []
    for raw_path in raw_paths:
        run_dir = Path(raw_path)
        try:
            results.append(load_experiment_result(run_dir))
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"failed to load reused run {run_dir}: {e}") from e
    return results


def _apply_dotted_overrides(cfg: ExperimentConfig, overrides: tuple[str, ...]) -> ExperimentConfig:
    """
    Apply repeatable ``--override key.path=value`` flags via dict round-trip.

    Empty ``overrides`` is a no-op; otherwise we ``model_dump`` to JSON-safe
    primitives, mutate the dict via :func:`apply_overrides`, then ``model_validate``
    so the returned object is fully validated against the same schema.
    """

    if not overrides:
        return cfg
    payload = cfg.model_dump(mode="json")
    try:
        payload = apply_overrides(payload, overrides)
    except ValueError as e:
        raise click.ClickException(f"--override failed: {e}") from e
    try:
        return ExperimentConfig.model_validate(payload)
    except ValidationError as e:
        raise click.ClickException(f"--override re-validation failed: {e}") from e


cli.add_command(study)


@cli.command("clean")
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to clean (default: experiment_results/).",
)
@click.option(
    "--apply",
    "apply",
    is_flag=True,
    default=False,
    help="Actually wipe (default: dry-run that prints what would be wiped).",
)
@click.option(
    "--keep",
    "keep",
    multiple=True,
    help="Directory name to preserve under <store-root>. Repeatable.",
)
def clean_cmd(store_root: Path, apply: bool, keep: tuple[str, ...]) -> None:
    """
    Wipe the contents of ephemeral subdirs under ``--store-root`` (default: experiment_results/).

    Each candidate directory survives as an empty placeholder so the
    canonical store layout (``runs/``, ``hpo/``, ``studies/``, ...) is
    intact after a wipe. Refuses to wipe any directory containing
    git-tracked files; pass ``--keep <name>`` for each to exclude them
    and rerun. A short allowlist of stray top-level sweep-tracking files
    (``.sweep_pid``, ``.sweep_started_at``, ``.sweep_log_path``,
    ``sweep_*.log``) is also removed; every other top-level file is left
    in place.
    """

    from src.orchestration.clean import apply_clean, format_plan, plan_clean

    plan = plan_clean(store_root, keep=keep)
    click.echo(format_plan(plan))
    if not apply:
        return
    if plan.refused:
        names = ", ".join(c.path.name for c in plan.refused)
        raise click.ClickException(
            f"refusing to apply: tracked files under {names}. "
            f"`git rm` first or pass --keep for each."
        )
    wiped = apply_clean(plan)
    suffix = f" and removed {len(plan.stray_files)} stray file(s)" if plan.stray_files else ""
    click.echo(f"wiped {len(wiped)} directory(ies){suffix}.")


if __name__ == "__main__":
    cli()
