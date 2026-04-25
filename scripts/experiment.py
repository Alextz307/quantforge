"""Command-line driver for the experiment runner.

Subcommands:

* ``run``          Fetch data, walk-forward, persist artifacts to
                   ``experiment_results/runs/<experiment_id>/``.
* ``train-model``  Fit one model standalone and persist to
                   ``experiment_results/models/<name>/`` for later
                   injection via ``ExperimentConfig.pretrained_leaves``.
* ``list-models``  Enumerate saved model artifacts for discovery.
* ``tune``         Drive an Optuna study over a config's HPO space,
                   persisting to ``experiment_results/hpo/<study>/``.
* ``compare``      Run N configs, rank them, compute pairwise Sharpe
                   significance, write the bundle to
                   ``experiment_results/comparisons/<out_name>/``.

The CLI deliberately mirrors ``scripts/benchmark.py`` (same ``click.group`` +
``--store-root`` convention + ``ClickException`` wrapping of runtime errors)
so a user who knows ``make bench`` knows ``make experiment`` instantly.

Future subcommands (``holdout-eval``, ``forward-run``) land in later
batches — the group is extensible without breaking these.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click
from pydantic import ValidationError

from src.core import json_io
from src.core.config import (
    ExperimentConfig,
    StandaloneModelConfig,
    load_experiment_config,
    load_standalone_model_config,
)
from src.core.exceptions import LeakageError
from src.core.hpo_config import HPOConfig, load_hpo_config
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    HPO_SUBDIR,
    METADATA_JSON,
    MODEL_ARTIFACT_MANIFEST_JSON,
    MODEL_ARTIFACT_WEIGHTS_SUBDIR,
    MODELS_SUBDIR,
    RUNS_SUBDIR,
)
from src.orchestration.builder import build_experiment
from src.orchestration.comparison import SignificanceTest, run_comparison
from src.orchestration.model_artifact import ModelArtifactManifest, save_model_artifact
from src.orchestration.standalone_training import train_model_standalone

# ``optuna`` is deferred into ``tune_cmd`` so it does not load on every CLI
# invocation (e.g., ``--help`` or ``compare``). The visualization reporters
# import matplotlib transitively, but matplotlib already lands at top-level
# via ``pmdarima`` (model registry side-effect import), so deferring them
# here only saves the wrapper modules — kept lazy for symmetry with optuna.

DEFAULT_STORE_ROOT = Path("experiment_results")

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Python logging level for the runner.",
)
def cli(log_level: str) -> None:
    """Quant-engine experiment orchestrator."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
def run_cmd(
    config_path: Path,
    name: str | None,
    seed: int | None,
    store_root: Path,
    write_report: bool,
) -> None:
    """Execute a single walk-forward experiment end-to-end."""
    try:
        cfg = load_experiment_config(config_path)
    except (ValidationError, FileNotFoundError, ValueError) as e:
        raise click.ClickException(f"failed to load config {config_path}: {e}") from e

    if name is not None or seed is not None:
        cfg = _override_experiment(cfg, name=name, seed=seed)

    try:
        experiment = build_experiment(cfg)
    except (ValidationError, ValueError) as e:
        raise click.ClickException(f"failed to build experiment: {e}") from e

    click.echo(f"running experiment '{cfg.name}' ({cfg.strategy.name} × {cfg.data.tickers[0]}) ...")
    try:
        result = experiment.run(store_root=store_root, write_report=write_report)
    except LeakageError as e:
        raise click.ClickException(f"leakage tripwire fired: {e}") from e
    except (NotImplementedError, ValueError, RuntimeError) as e:
        raise click.ClickException(f"experiment failed: {e}") from e

    run_dir = store_root / RUNS_SUBDIR / result.experiment_id
    click.echo(f"experiment_id: {result.experiment_id}")
    click.echo(f"artifacts:    {run_dir}")
    click.echo(f"folds:        {len(result.folds)}")


@cli.command("train-model")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a StandaloneModelConfig YAML.",
)
@click.option(
    "--name",
    default=None,
    help="Override the config's `name` field (used as the artifact directory name).",
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
def train_model_cmd(
    config_path: Path,
    name: str | None,
    seed: int | None,
    store_root: Path,
) -> None:
    """Train one model standalone and persist to experiment_results/models/<name>/.

    The resulting artifact is reusable via ``ExperimentConfig.pretrained_leaves``
    — `experiment run --config strategy.yaml` with a matching ``pretrained_leaves``
    entry will load this model frozen into the strategy.
    """
    try:
        cfg = load_standalone_model_config(config_path)
    except (ValidationError, FileNotFoundError, ValueError) as e:
        raise click.ClickException(f"failed to load config {config_path}: {e}") from e

    if name is not None or seed is not None:
        cfg = _override_standalone(cfg, name=name, seed=seed)

    click.echo(
        f"training model '{cfg.name}' ({cfg.model.name} / "
        f"{cfg.model_kind.value}) on {cfg.data.tickers}..."
    )
    try:
        trained = train_model_standalone(cfg)
    except (NotImplementedError, ValueError, RuntimeError) as e:
        raise click.ClickException(f"standalone training failed: {e}") from e

    artifact_dir = store_root / MODELS_SUBDIR / cfg.name
    try:
        save_model_artifact(
            artifact_dir,
            model=trained.model,
            manifest=trained.manifest,
            config=cfg,
        )
    except FileExistsError as e:
        raise click.ClickException(
            f"artifact path {artifact_dir} already exists and is non-empty; "
            f"choose a fresh --name or delete the existing directory. ({e})"
        ) from e

    click.echo(f"artifact:  {artifact_dir}")
    click.echo(f"data_hash: {trained.manifest.data_hash[:12]}...")
    click.echo(f"git_sha:   {trained.manifest.git_sha}")


@cli.command("list-models")
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Root directory whose `models/` subdirectory is enumerated.",
)
def list_models_cmd(store_root: Path) -> None:
    """Enumerate saved model artifacts under experiment_results/models/."""
    models_root = store_root / MODELS_SUBDIR
    if not models_root.is_dir():
        click.echo(f"no models directory at {models_root} — nothing to list.")
        return

    rows: list[tuple[str, str, str, str, str, str]] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / MODEL_ARTIFACT_MANIFEST_JSON
        if not manifest_path.is_file():
            continue
        try:
            manifest = ModelArtifactManifest.from_dict(json_io.read_dict(manifest_path))
        except (ValueError, KeyError) as e:
            click.echo(f"  [skip] {entry.name}: manifest unreadable ({e})")
            continue

        train_end = "?"
        meta_path = entry / MODEL_ARTIFACT_WEIGHTS_SUBDIR / METADATA_JSON
        if meta_path.is_file():
            try:
                meta_raw = json_io.read_dict(meta_path)
                train_end = str(meta_raw.get("train_end", "?"))
            except (ValueError, KeyError):
                train_end = "?"

        rows.append(
            (
                manifest.name,
                manifest.model_name,
                manifest.model_kind.value,
                train_end,
                manifest.data_hash[:8],
                manifest.git_sha,
            )
        )

    if not rows:
        click.echo(f"no model artifacts under {models_root}.")
        return

    header = ("name", "model", "kind", "train_end", "data_hash", "git_sha")
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
    click.echo("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    click.echo("  ".join("-" * w for w in widths))
    for row in rows:
        click.echo("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


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
def tune_cmd(
    config_path: Path,
    hpo_config_path: Path,
    n_trials_override: int | None,
    n_jobs_override: int | None,
    store_root: Path,
    write_report: bool,
) -> None:
    """Run an Optuna study over an ExperimentConfig's hyperparameter space.

    The study is persisted to a SQLite file under
    ``<store_root>/hpo/<study_name>/optuna_study.db`` — re-running with
    the same ``--config`` + ``--hpo-config`` resumes from the last
    completed trial (Optuna's semantics: ``n_trials`` is the number of
    NEW trials to run each invocation, not a cap on total trials).
    """
    # Defer optuna so unrelated subcommands don't pay its import cost.
    from src.optimization.tuner import StrategyTuner
    from src.visualization.hpo_reporter import HPOReporter

    try:
        experiment_cfg = load_experiment_config(config_path)
    except (ValidationError, FileNotFoundError, ValueError) as e:
        raise click.ClickException(f"failed to load experiment config {config_path}: {e}") from e
    try:
        hpo_cfg = load_hpo_config(hpo_config_path)
    except (ValidationError, FileNotFoundError, ValueError) as e:
        raise click.ClickException(f"failed to load hpo config {hpo_config_path}: {e}") from e

    hpo_cfg = _apply_hpo_overrides(hpo_cfg, n_trials=n_trials_override, n_jobs=n_jobs_override)

    tuner = StrategyTuner(
        experiment_cfg=experiment_cfg,
        hpo_cfg=hpo_cfg,
        store_root=store_root,
    )

    click.echo(
        f"tuning '{experiment_cfg.strategy.name}' on study '{hpo_cfg.study_name}' "
        f"for {hpo_cfg.n_trials} trial(s) (n_jobs={hpo_cfg.n_jobs}) ..."
    )
    try:
        study = tuner.run()
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
def compare_cmd(
    config_paths: tuple[Path, ...],
    out_name: str,
    significance_test: str,
    n_jobs: int,
    store_root: Path,
    write_report: bool,
) -> None:
    """Run N configs, rank, optionally test pairwise significance.

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

    configs: list[ExperimentConfig] = []
    for path in config_paths:
        try:
            configs.append(load_experiment_config(path))
        except (ValidationError, FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"failed to load config {path}: {e}") from e

    # click ``Choice(case_sensitive=False)`` already returns the canonical
    # lowercase form of the choice, so ``SignificanceTest(...)`` round-trips.
    sig = SignificanceTest(significance_test)
    click.echo(
        f"comparing {len(configs)} strategies under '{out_name}' "
        f"(n_jobs={n_jobs}, significance={sig.value}) ..."
    )
    try:
        report, folds_by_strategy = run_comparison(
            configs,
            out_name=out_name,
            store_root=store_root,
            n_jobs=n_jobs,
            significance_test=sig,
        )
    except LeakageError as e:
        raise click.ClickException(f"leakage tripwire fired: {e}") from e
    except (NotImplementedError, ValueError, RuntimeError) as e:
        raise click.ClickException(f"comparison failed: {e}") from e

    cmp_dir = store_root / COMPARISONS_SUBDIR / out_name
    if write_report:
        ComparisonReporter().generate_full_report(
            report, cmp_dir, folds_by_strategy=folds_by_strategy
        )

    click.echo(f"out_name:   {report.out_name}")
    click.echo(f"artifacts:  {cmp_dir}")
    click.echo(f"strategies: {', '.join(report.per_strategy_stats.keys())}")
    if report.pairwise:
        n_sig = sum(1 for p in report.pairwise if p.significant)
        click.echo(f"pairwise:   {len(report.pairwise)} comparisons, {n_sig} significant")


def _apply_hpo_overrides(cfg: HPOConfig, *, n_trials: int | None, n_jobs: int | None) -> HPOConfig:
    """Rebuild the HPO config with CLI overrides, re-running validators."""
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


def _override[CfgT: (ExperimentConfig, StandaloneModelConfig)](
    cfg: CfgT, *, name: str | None, seed: int | None
) -> CfgT:
    """Rebuild ``cfg`` with CLI ``--name`` / ``--seed`` overrides applied.

    Re-runs the pydantic validator so an override never leaves the config
    in a half-validated state (e.g. an empty-string ``name`` from the CLI
    would be caught by ``min_length=1``).
    """
    payload = cfg.model_dump(mode="json")
    if name is not None:
        payload["name"] = name
    if seed is not None:
        payload["seed"] = seed
    return type(cfg).model_validate(payload)


# Type-narrowed aliases kept for call-site + test clarity — both are the
# same generic; a collision on the overload form (two concrete bindings
# sharing a symbol) is what we want to prevent.
def _override_experiment(
    cfg: ExperimentConfig, *, name: str | None, seed: int | None
) -> ExperimentConfig:
    return _override(cfg, name=name, seed=seed)


def _override_standalone(
    cfg: StandaloneModelConfig, *, name: str | None, seed: int | None
) -> StandaloneModelConfig:
    return _override(cfg, name=name, seed=seed)


if __name__ == "__main__":
    cli()
