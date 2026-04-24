"""Command-line driver for the experiment runner.

Subcommands:

* ``run``          Fetch data, walk-forward, persist artifacts to
                   ``experiment_results/runs/<experiment_id>/``.
* ``train-model``  Fit one model standalone and persist to
                   ``experiment_results/models/<name>/`` for later
                   injection via ``ExperimentConfig.pretrained_leaves``.
* ``list-models``  Enumerate saved model artifacts for discovery.

The CLI deliberately mirrors ``scripts/benchmark.py`` (same ``click.group`` +
``--store-root`` convention + ``ClickException`` wrapping of runtime errors)
so a user who knows ``make bench`` knows ``make experiment`` instantly.

Future subcommands (``tune``, ``compare``, ``holdout-eval``, ``forward-run``)
land in later batches — the group is extensible without breaking these.
"""

from __future__ import annotations

import logging
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
from src.core.persistence import (
    METADATA_JSON,
    MODEL_ARTIFACT_MANIFEST_JSON,
    MODEL_ARTIFACT_WEIGHTS_SUBDIR,
    MODELS_SUBDIR,
    RUNS_SUBDIR,
)
from src.orchestration.builder import build_experiment
from src.orchestration.model_artifact import ModelArtifactManifest, save_model_artifact
from src.orchestration.standalone_training import train_model_standalone

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
