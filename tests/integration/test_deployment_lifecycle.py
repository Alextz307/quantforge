"""
End-to-end deployment lifecycle: train -> save -> deploy -> predict -> recall.

Validates the framework primitive without yfinance - the stub fetcher
returns slices of the synthetic frame the source strategy was *not*
trained on, so the predict path exercises the same code that runs in
production (load registry-driven strategy, run warmup window through
``generate_signals``, append to ``signals.jsonl``) without network.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.core.persistence import (
    DEPLOYMENT_MANIFEST_JSON,
    DEPLOYMENT_SIGNALS_JSONL,
    DEPLOYMENTS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    RUNS_SUBDIR,
)
from src.core.types import Interval
from src.orchestration.deployment import (
    create_deployment,
    load_deployment,
    predict,
    read_signals,
)
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_ohlcv_df

_LIFECYCLE_TOTAL_BARS = 600
_LIFECYCLE_TRAIN_END = 400
_LIFECYCLE_WINDOW = 20
_LIFECYCLE_TREND = 50
_LIFECYCLE_GARCH_P = 1
_LIFECYCLE_GARCH_Q = 1
_LIFECYCLE_WARMUP_BARS = 200
_LIFECYCLE_RUN_ID = "lifecycle_run"
_LIFECYCLE_DEPLOYMENT_ID = "lifecycle_deployment"


class _SyntheticFetcher:
    """
    Slice the master frame up to a controlled cursor - no vendor calls.
    """

    def __init__(self, bars: pd.DataFrame, cursor: dict[str, int]) -> None:
        self._bars = bars
        self._cursor = cursor

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval,
    ) -> pd.DataFrame:
        del ticker, start, end, interval
        return self._bars.iloc[: self._cursor["last"] + 1]


def test_full_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Train, save, deploy, predict twice - and verify on-disk state.
    """

    from src.core.config import load_experiment_config, write_frozen_yaml

    store = tmp_path / "store"
    run_dir = store / RUNS_SUBDIR / _LIFECYCLE_RUN_ID
    run_dir.mkdir(parents=True)

    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    bars = make_synthetic_ohlcv_df(n_rows=_LIFECYCLE_TOTAL_BARS)
    train_df = bars.iloc[:_LIFECYCLE_TRAIN_END]
    strategy = AdaptiveBollingerStrategy(
        window=_LIFECYCLE_WINDOW,
        trend_window=_LIFECYCLE_TREND,
        garch_p_max=_LIFECYCLE_GARCH_P,
        garch_q_max=_LIFECYCLE_GARCH_Q,
    )
    strategy.train(train_df)
    strategy.save(run_dir / EXPERIMENT_STRATEGY_SUBDIR)

    deployment = create_deployment(
        source_kind="run",
        source_id=_LIFECYCLE_RUN_ID,
        store_root=store,
        deployment_id=_LIFECYCLE_DEPLOYMENT_ID,
        warmup_bars=_LIFECYCLE_WARMUP_BARS,
    )

    dep_dir = store / DEPLOYMENTS_SUBDIR / _LIFECYCLE_DEPLOYMENT_ID
    assert (dep_dir / DEPLOYMENT_MANIFEST_JSON).is_file()
    assert load_deployment(store, _LIFECYCLE_DEPLOYMENT_ID) == deployment

    cursor = {"last": _LIFECYCLE_TRAIN_END + 30}
    monkeypatch.setattr(
        "src.orchestration.deployment.resolve_fetcher",
        lambda _: _SyntheticFetcher(bars, cursor),
    )

    first_as_of = pd.Timestamp(bars.index[cursor["last"]])
    first_row = predict(
        deployment_id=_LIFECYCLE_DEPLOYMENT_ID,
        store_root=store,
        as_of=first_as_of,
    )
    assert first_row.bar_ts == first_as_of
    assert first_row.signal in {-1.0, 0.0, 1.0}
    assert first_row.warmup_bars_used == _LIFECYCLE_WARMUP_BARS

    cursor["last"] = _LIFECYCLE_TRAIN_END + 80
    second_as_of = pd.Timestamp(bars.index[cursor["last"]])
    second_row = predict(
        deployment_id=_LIFECYCLE_DEPLOYMENT_ID,
        store_root=store,
        as_of=second_as_of,
    )
    assert second_row.bar_ts == second_as_of
    assert second_row.bar_ts != first_row.bar_ts

    logged = read_signals(store, _LIFECYCLE_DEPLOYMENT_ID)
    assert len(logged) == 2
    assert {r.bar_ts for r in logged} == {first_as_of, second_as_of}

    log_path = dep_dir / DEPLOYMENT_SIGNALS_JSONL
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
