"""
Command-line driver for live deployments of trained strategies.

Subcommands:

* ``create``   Pin a deployment to a completed run (or HPO study). Writes
               ``<store>/deployments/<id>/``: manifest, YAML, empty signal log.
* ``predict``  Generate today's signal (or recall a cached one for the
               same target bar). Idempotent on ``--as-of``.
* ``list``     One line per deployment under ``<store>/deployments/``.
* ``show``     Print the typed manifest for a single deployment.
* ``signals``  Print the deployment's signal log (most-recent first).

A deployment is pinned to one trained run — training a fresher model is
a separate concern. Use ``quant experiment run`` and then create a new
deployment pointing at the resulting run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

from src.core.exceptions import LeakageError, WarmupInsufficientError
from src.core.logging import CLI_LOG_FORMAT, attach_cli_log_file, get_logger
from src.core.persistence import DEPLOYMENTS_SUBDIR
from src.orchestration.deployment import (
    create_deployment,
    load_deployment,
    predict,
    read_signals,
)
from src.orchestration.holdout_eval import SourceKind

DEFAULT_STORE_ROOT = Path("experiment_results")

logger = get_logger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Python logging level for the deploy CLI.",
)
def cli(log_level: str) -> None:
    """
    Live deployment management for trained strategies.
    """

    logging.basicConfig(level=log_level.upper(), format=CLI_LOG_FORMAT)


_STORE_OPTION = click.option(
    "--store",
    "store_root",
    default=str(DEFAULT_STORE_ROOT),
    type=click.Path(file_okay=False, path_type=Path),
    help="Root experiment_results/ directory (also holds deployments/).",
)


@cli.command("create")
@click.option(
    "--from-run",
    "from_run",
    default=None,
    help="experiment_id of a completed run under <store>/runs/.",
)
@click.option(
    "--from-hpo",
    "from_hpo",
    default=None,
    help="Study name of a completed HPO study under <store>/hpo/. The best trial's strategy is deployed.",
)
@click.option(
    "--name",
    "name",
    default=None,
    help="Display name for the deployment. Defaults to '<ticker>-<strategy>-<train_end>'.",
)
@click.option(
    "--warmup-bars",
    "warmup_bars",
    default=None,
    type=click.IntRange(min=1),
    help=(
        "Bars of warmup history per predict call. Default: auto-derive from "
        "the strategy's required_warmup_bars + convergence_margin_bars."
    ),
)
@_STORE_OPTION
def create_cmd(
    from_run: str | None,
    from_hpo: str | None,
    name: str | None,
    warmup_bars: int | None,
    store_root: Path,
) -> None:
    """
    Create a deployment pinned to a trained run or HPO study.
    """

    if (from_run is None) == (from_hpo is None):
        raise click.UsageError("pass exactly one of --from-run or --from-hpo.")

    source_kind: SourceKind
    if from_run is not None:
        source_kind = "run"
        source_id = from_run
    else:
        assert from_hpo is not None
        source_kind = "hpo"
        source_id = from_hpo

    with attach_cli_log_file(store_root, "deploy_create") as log_path:
        click.echo(
            f"creating deployment from {source_kind}={source_id!r} → log: {log_path}"
        )
        try:
            deployment = create_deployment(
                source_kind=source_kind,
                source_id=source_id,
                store_root=store_root,
                name=name,
                warmup_bars=warmup_bars,
            )
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"deploy create failed: {e}") from e

        click.echo(f"deployment_id: {deployment.deployment_id}")
        click.echo(f"name:          {deployment.name}")
        click.echo(f"source:        {deployment.source_kind}:{deployment.source_id}")
        click.echo(f"warmup_bars:   {deployment.warmup_bars}")


@cli.command("predict")
@click.argument("deployment_id")
@click.option(
    "--as-of",
    "as_of",
    default=None,
    help="ISO timestamp (UTC) of the bar to predict for. Defaults to wall-clock now.",
)
@_STORE_OPTION
def predict_cmd(deployment_id: str, as_of: str | None, store_root: Path) -> None:
    """
    Generate (or recall) the latest signal for a deployment.
    """

    resolved_as_of = pd.Timestamp(as_of, tz="UTC") if as_of is not None else None

    with attach_cli_log_file(store_root, "deploy_predict") as log_path:
        click.echo(
            f"predict deployment={deployment_id!r} → log: {log_path}", err=True
        )
        try:
            row = predict(
                deployment_id=deployment_id,
                store_root=store_root,
                as_of=resolved_as_of,
            )
        except LeakageError as e:
            raise click.ClickException(f"leakage tripwire fired: {e}") from e
        except WarmupInsufficientError as e:
            raise click.ClickException(f"warmup window too short: {e}") from e
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise click.ClickException(f"predict failed: {e}") from e

        click.echo(json.dumps(row.to_dict(), indent=2, sort_keys=True))


@cli.command("list")
@_STORE_OPTION
def list_cmd(store_root: Path) -> None:
    """
    List every deployment under <store>/deployments/.
    """

    root = store_root / DEPLOYMENTS_SUBDIR
    if not root.is_dir():
        click.echo("(no deployments)")
        return

    entries = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not entries:
        click.echo("(no deployments)")
        return

    for deployment_id in entries:
        try:
            deployment = load_deployment(store_root, deployment_id)
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"{deployment_id}: <manifest error: {e}>")
            continue
        click.echo(
            f"{deployment_id}  {deployment.name}  ({deployment.source_kind}:{deployment.source_id})"
        )


@cli.command("show")
@click.argument("deployment_id")
@_STORE_OPTION
def show_cmd(deployment_id: str, store_root: Path) -> None:
    """
    Print the typed manifest for one deployment.
    """

    try:
        deployment = load_deployment(store_root, deployment_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(f"{e}") from e
    click.echo(json.dumps(deployment.to_dict(), indent=2, sort_keys=True))


@cli.command("signals")
@click.argument("deployment_id")
@click.option(
    "--limit",
    "limit",
    default=None,
    type=click.IntRange(min=1),
    help="Print only the most recent N rows.",
)
@_STORE_OPTION
def signals_cmd(deployment_id: str, limit: int | None, store_root: Path) -> None:
    """
    Print a deployment's signal log (most-recent first).
    """

    try:
        rows = read_signals(store_root, deployment_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(f"{e}") from e

    ordered = list(reversed(rows))
    if limit is not None:
        ordered = ordered[:limit]
    for row in ordered:
        click.echo(json.dumps(row.to_dict(), sort_keys=True))


if __name__ == "__main__":
    cli()
