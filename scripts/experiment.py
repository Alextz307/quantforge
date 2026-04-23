"""Command-line driver for the experiment runner.

Subcommands:

* ``run``  Fetch data, walk-forward, persist artifacts to
           ``experiment_results/runs/<experiment_id>/``.

The CLI deliberately mirrors ``scripts/benchmark.py`` (same ``click.group`` +
``--store-root`` convention + ``ClickException`` wrapping of runtime errors)
so a user who knows ``make bench`` knows ``make experiment`` instantly.

Future subcommands (``tune``, ``compare``, ``holdout-eval``, ``forward-run``,
``list-models``, ``train-model``) land in later batches — the group is
extensible without breaking this command.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from pydantic import ValidationError

from src.core.config import ExperimentConfig, load_experiment_config
from src.core.exceptions import LeakageError
from src.orchestration.builder import build_experiment

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
        cfg = _override(cfg, name=name, seed=seed)

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

    run_dir = store_root / "runs" / result.experiment_id
    click.echo(f"experiment_id: {result.experiment_id}")
    click.echo(f"artifacts:    {run_dir}")
    click.echo(f"folds:        {len(result.folds)}")


def _override(cfg: ExperimentConfig, *, name: str | None, seed: int | None) -> ExperimentConfig:
    """Rebuild a validated ``ExperimentConfig`` with CLI overrides applied.

    ``model_copy(update=...)`` bypasses validators, so we re-validate via
    ``model_validate`` to preserve the frozen-dataclass + registry-check
    guarantees.
    """
    payload = cfg.model_dump(mode="json")
    if name is not None:
        payload["name"] = name
    if seed is not None:
        payload["seed"] = seed
    return ExperimentConfig.model_validate(payload)


if __name__ == "__main__":
    cli()
