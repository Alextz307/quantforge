"""
Smoke tests for the ``scripts/deploy.py`` click CLI.

Exercises the create / list / show / signals / predict subcommand glue.
The predict path uses a monkeypatched stub fetcher so the test stays
offline — the predict op itself is covered by
``test_deployment_predict.py``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from scripts.deploy import cli
from src.core.persistence import (
    DEPLOYMENT_MANIFEST_JSON,
    DEPLOYMENTS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    RUNS_SUBDIR,
)
from src.core.types import Interval
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_ohlcv_df

_RUN_ID = "cli_run"
_TRAIN_END_INDEX = 250
_TOTAL_BARS = 400
_BOLLINGER_WINDOW = 20
_BOLLINGER_TREND = 50
_GARCH_P = 1
_GARCH_Q = 1
_WARMUP_BARS = 100


def _materialise_source(store: Path) -> pd.DataFrame:
    from src.core.config import load_experiment_config, write_frozen_yaml

    run_dir = store / RUNS_SUBDIR / _RUN_ID
    run_dir.mkdir(parents=True)
    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    bars = make_synthetic_ohlcv_df(n_rows=_TOTAL_BARS)
    train_df = bars.iloc[:_TRAIN_END_INDEX]
    strategy = AdaptiveBollingerStrategy(
        window=_BOLLINGER_WINDOW,
        trend_window=_BOLLINGER_TREND,
        garch_p_max=_GARCH_P,
        garch_q_max=_GARCH_Q,
    )
    strategy.train(train_df)
    strategy.save(run_dir / EXPERIMENT_STRATEGY_SUBDIR)
    return bars


def test_create_then_list_then_show(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _materialise_source(store)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "create",
            "--from-run",
            _RUN_ID,
            "--warmup-bars",
            str(_WARMUP_BARS),
            "--store",
            str(store),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "deployment_id:" in result.output

    deployment_dirs = list((store / DEPLOYMENTS_SUBDIR).iterdir())
    assert len(deployment_dirs) == 1
    deployment_id = deployment_dirs[0].name
    assert (deployment_dirs[0] / DEPLOYMENT_MANIFEST_JSON).is_file()

    list_result = runner.invoke(cli, ["list", "--store", str(store)])
    assert list_result.exit_code == 0
    assert deployment_id in list_result.output

    show_result = runner.invoke(cli, ["show", deployment_id, "--store", str(store)])
    assert show_result.exit_code == 0
    parsed = json.loads(show_result.output)
    assert parsed["deployment_id"] == deployment_id
    assert parsed["source_kind"] == "run"
    assert parsed["warmup_bars"] == _WARMUP_BARS


def test_create_requires_exactly_one_source(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()

    runner = CliRunner()
    neither = runner.invoke(cli, ["create", "--store", str(store)])
    assert neither.exit_code != 0
    assert "exactly one" in neither.output

    both = runner.invoke(
        cli,
        ["create", "--from-run", "a", "--from-hpo", "b", "--store", str(store)],
    )
    assert both.exit_code != 0
    assert "exactly one" in both.output


def test_predict_via_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = tmp_path / "store"
    bars = _materialise_source(store)

    cursor = {"last": _TRAIN_END_INDEX + 30}

    class _Stub:
        def fetch(
            self,
            ticker: str,
            start: datetime,
            end: datetime,
            interval: Interval,
        ) -> pd.DataFrame:
            del ticker, start, end, interval
            return bars.iloc[: cursor["last"] + 1]

    monkeypatch.setattr(
        "src.orchestration.deployment.resolve_fetcher", lambda _: _Stub()
    )

    runner = CliRunner()
    create = runner.invoke(
        cli,
        ["create", "--from-run", _RUN_ID, "--store", str(store)],
    )
    assert create.exit_code == 0, create.output
    deployment_id = next((store / DEPLOYMENTS_SUBDIR).iterdir()).name

    as_of = pd.Timestamp(bars.index[cursor["last"]]).tz_localize("UTC").isoformat()
    predict_result = runner.invoke(
        cli,
        [
            "predict",
            deployment_id,
            "--as-of",
            as_of,
            "--store",
            str(store),
        ],
    )
    assert predict_result.exit_code == 0, predict_result.output
    payload = json.loads(predict_result.stdout)
    assert payload["signal"] in (-1.0, 0.0, 1.0)

    signals = runner.invoke(
        cli,
        ["signals", deployment_id, "--store", str(store)],
    )
    assert signals.exit_code == 0
    assert payload["bar_ts"] in signals.output


def test_list_empty_store(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--store", str(tmp_path / "empty_store")])
    assert result.exit_code == 0
    assert "no deployments" in result.output


def test_show_missing_deployment(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli, ["show", "nope", "--store", str(tmp_path / "store")]
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
