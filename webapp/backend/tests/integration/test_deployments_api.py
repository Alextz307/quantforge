"""
Integration tests for /api/deployments (auth-gated).

The tests materialise a real trained AdaptiveBollinger run on disk so
the framework's create + predict paths can resolve and run end-to-end.
The live fetcher is stubbed (no yfinance round-trip), but everything
between the HTTP boundary and the on-disk artifacts is real.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime
from http import HTTPStatus
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.core.config import load_experiment_config, write_frozen_yaml
from src.core.persistence import (
    DEPLOYMENT_SIGNALS_JSONL,
    DEPLOYMENTS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    RUNS_SUBDIR,
)
from src.core.types import Interval
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_ohlcv_df
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.core.types import Role
from webapp.backend.app.services.user_service import create_user
from webapp.backend.tests.conftest import (
    SECONDARY_PASSWORD,
    SECONDARY_USERNAME,
    TEST_USERNAME,
)

DEPLOYMENTS_PATH = "/api/deployments"

_RUN_ID = "20260101_120000_AdaptiveBollinger_test_synth"
_PAIRS_RUN_ID = "20260101_120000_PairsTrading_test_synth"
_TRAIN_ROWS = 250
_BOLLINGER_WINDOW = 20
_BOLLINGER_TREND = 50
_GARCH_P = 1
_GARCH_Q = 1
_PREDICT_BAR_OFFSET = 30


def _materialise_run(store_root: Path, run_id: str) -> Path:
    """
    Train a tiny AdaptiveBollinger and persist it under ``runs/<id>/``.
    """

    run_dir = store_root / RUNS_SUBDIR / run_id
    run_dir.mkdir(parents=True)
    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    df = make_synthetic_ohlcv_df(n_rows=_TRAIN_ROWS)
    strategy = AdaptiveBollingerStrategy(
        window=_BOLLINGER_WINDOW,
        trend_window=_BOLLINGER_TREND,
        garch_p_max=_GARCH_P,
        garch_q_max=_GARCH_Q,
    )
    strategy.train(df)
    strategy.save(run_dir / EXPERIMENT_STRATEGY_SUBDIR)
    return run_dir


def _materialise_pairs_run(store_root: Path) -> Path:
    """
    Write a config-only "pairs" run dir (no strategy_state).

    Suffices for the source-validation test: ``create_deployment`` reads
    the config to count tickers and rejects two-ticker sources before
    ever loading the strategy.
    """

    run_dir = store_root / RUNS_SUBDIR / _PAIRS_RUN_ID
    run_dir.mkdir(parents=True)
    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    pairs_cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"tickers": ["GLD", "SLV"]})}
    )
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, pairs_cfg)
    (run_dir / EXPERIMENT_STRATEGY_SUBDIR).mkdir()
    return run_dir


@pytest.fixture
def trained_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """
    Synthetic store with one trained AdaptiveBollinger run materialised.
    """

    store = tmp_path / "store"
    _materialise_run(store, _RUN_ID)
    monkeypatch.setenv("WEBAPP_STORE_ROOT", str(store))
    get_settings.cache_clear()
    yield store


@pytest.fixture
def stubbed_fetcher(
    trained_store: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Replace the live fetcher with a deterministic in-memory stub.

    The framework's ``predict()`` calls ``resolve_fetcher(interval).fetch``;
    the service's bar-ts probe calls ``_probe_latest_bar_ts``. Both are
    patched here so tests never touch yfinance and the latest-bar
    timestamp is deterministic.

    Returns ``(bars_df, latest_bar_ts)``. The bars cover enough history
    to satisfy the strategy's warmup; ``latest_bar_ts`` is the index of
    the bar we want predict-if-stale to act on (one past train_end).
    """

    bars = make_synthetic_ohlcv_df(n_rows=_TRAIN_ROWS + _PREDICT_BAR_OFFSET)
    latest_bar_ts = pd.Timestamp(bars.index[-1])

    class _StubFetcher:
        def fetch(
            self,
            ticker: str,
            start: datetime,
            end: datetime,
            interval: Interval,
        ) -> pd.DataFrame:
            del ticker, start, end, interval
            return bars

    def _stub_resolve_fetcher(_interval: Interval) -> _StubFetcher:
        return _StubFetcher()

    def _stub_probe(_ticker: str, _interval: Interval) -> pd.Timestamp:
        return latest_bar_ts

    monkeypatch.setattr(
        "src.orchestration.deployment.resolve_fetcher", _stub_resolve_fetcher
    )
    monkeypatch.setattr(
        "webapp.backend.app.services.deployment_service._probe_latest_bar_ts",
        _stub_probe,
    )
    return bars, latest_bar_ts


def _login(client: TestClient, username: str, password: str) -> None:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == HTTPStatus.OK


def _create_secondary_user(db_conn: sqlite3.Connection) -> None:
    create_user(
        db_conn,
        username=SECONDARY_USERNAME,
        password=SECONDARY_PASSWORD,
        role=Role.USER,
    )


# ---------------------------------------------------------------------------
# Auth + listing
# ---------------------------------------------------------------------------


def test_list_requires_auth(client: TestClient, trained_store: Path) -> None:
    response = client.get(DEPLOYMENTS_PATH)
    assert response.status_code == HTTPStatus.UNAUTHORIZED


def test_list_empty_for_new_user(authed_client: TestClient, trained_store: Path) -> None:
    response = authed_client.get(DEPLOYMENTS_PATH)
    assert response.status_code == HTTPStatus.OK
    assert response.json() == []


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_minimal_payload(authed_client: TestClient, trained_store: Path) -> None:
    response = authed_client.post(
        DEPLOYMENTS_PATH,
        json={"source_kind": "run", "source_id": _RUN_ID},
    )

    assert response.status_code == HTTPStatus.CREATED, response.text
    payload = response.json()
    assert payload["source_kind"] == "run"
    assert payload["source_id"] == _RUN_ID
    assert payload["ticker"] == "SPY"
    assert payload["strategy_name"] == "AdaptiveBollinger"
    assert payload["interval"] == "daily"
    assert payload["warmup_bars"] > 0
    assert payload["latest_signal"] is None
    assert payload["owner_username"] == TEST_USERNAME


def test_create_with_custom_name_and_warmup(
    authed_client: TestClient, trained_store: Path
) -> None:
    response = authed_client.post(
        DEPLOYMENTS_PATH,
        json={
            "source_kind": "run",
            "source_id": _RUN_ID,
            "name": "my deploy",
            "warmup_bars": 120,
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    payload = response.json()
    assert payload["name"] == "my deploy"
    assert payload["warmup_bars"] == 120


def test_create_writes_manifest_to_disk(
    authed_client: TestClient, trained_store: Path
) -> None:
    response = authed_client.post(
        DEPLOYMENTS_PATH,
        json={"source_kind": "run", "source_id": _RUN_ID},
    )
    deployment_id = response.json()["id"]

    dep_dir = trained_store / DEPLOYMENTS_SUBDIR / deployment_id
    assert (dep_dir / "manifest.json").is_file()
    assert (dep_dir / DEPLOYMENT_SIGNALS_JSONL).is_file()


def test_create_rejects_unknown_source(
    authed_client: TestClient, trained_store: Path
) -> None:
    response = authed_client.post(
        DEPLOYMENTS_PATH,
        json={"source_kind": "run", "source_id": "absent_run"},
    )
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_create_rejects_pairs_source(
    authed_client: TestClient, trained_store: Path
) -> None:
    _materialise_pairs_run(trained_store)
    response = authed_client.post(
        DEPLOYMENTS_PATH,
        json={"source_kind": "run", "source_id": _PAIRS_RUN_ID},
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "single-asset" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Get / rename / delete
# ---------------------------------------------------------------------------


def test_get_detail_round_trip(authed_client: TestClient, trained_store: Path) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()

    detail = authed_client.get(f"{DEPLOYMENTS_PATH}/{created['id']}").json()
    assert detail["id"] == created["id"]
    assert detail["latest_signal"] is None


def test_get_404_for_unknown(authed_client: TestClient, trained_store: Path) -> None:
    response = authed_client.get(f"{DEPLOYMENTS_PATH}/missing")
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_rename(authed_client: TestClient, trained_store: Path) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()

    renamed = authed_client.patch(
        f"{DEPLOYMENTS_PATH}/{created['id']}", json={"name": "renamed"}
    )
    assert renamed.status_code == HTTPStatus.OK
    assert renamed.json()["name"] == "renamed"

    refetched = authed_client.get(f"{DEPLOYMENTS_PATH}/{created['id']}").json()
    assert refetched["name"] == "renamed"


def test_delete_removes_row_and_dir(
    authed_client: TestClient, trained_store: Path
) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()
    deployment_id = created["id"]
    dep_dir = trained_store / DEPLOYMENTS_SUBDIR / deployment_id
    assert dep_dir.is_dir()

    response = authed_client.delete(f"{DEPLOYMENTS_PATH}/{deployment_id}")
    assert response.status_code == HTTPStatus.NO_CONTENT
    assert not dep_dir.exists()
    assert authed_client.get(f"{DEPLOYMENTS_PATH}/{deployment_id}").status_code == HTTPStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# Ownership scoping
# ---------------------------------------------------------------------------


def test_other_user_cannot_see_deployment(
    authed_client: TestClient,
    trained_store: Path,
    db_conn: sqlite3.Connection,
    client: TestClient,
) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()
    deployment_id = created["id"]

    _create_secondary_user(db_conn)
    authed_client.post("/api/auth/logout")
    _login(client, SECONDARY_USERNAME, SECONDARY_PASSWORD)

    assert client.get(DEPLOYMENTS_PATH).json() == []
    assert client.get(f"{DEPLOYMENTS_PATH}/{deployment_id}").status_code == HTTPStatus.NOT_FOUND
    assert (
        client.patch(f"{DEPLOYMENTS_PATH}/{deployment_id}", json={"name": "x"}).status_code
        == HTTPStatus.NOT_FOUND
    )
    assert client.delete(f"{DEPLOYMENTS_PATH}/{deployment_id}").status_code == HTTPStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# Signal log
# ---------------------------------------------------------------------------


def test_signals_empty_on_fresh_deployment(
    authed_client: TestClient, trained_store: Path
) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()
    response = authed_client.get(f"{DEPLOYMENTS_PATH}/{created['id']}/signals")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == []


# ---------------------------------------------------------------------------
# Predict-if-stale
# ---------------------------------------------------------------------------


def test_predict_if_stale_cache_miss_then_hit(
    authed_client: TestClient,
    trained_store: Path,
    stubbed_fetcher: tuple[pd.DataFrame, pd.Timestamp],
) -> None:
    """
    First call computes (stale=True); second call recalls (stale=False).
    """

    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()
    deployment_id = created["id"]

    first = authed_client.post(f"{DEPLOYMENTS_PATH}/{deployment_id}/predict-if-stale")
    assert first.status_code == HTTPStatus.OK, first.text
    first_payload = first.json()
    assert first_payload["stale"] is True
    first_signal = first_payload["signal"]
    assert isinstance(first_signal["signal"], int | float)

    second = authed_client.post(f"{DEPLOYMENTS_PATH}/{deployment_id}/predict-if-stale")
    assert second.status_code == HTTPStatus.OK
    second_payload = second.json()
    assert second_payload["stale"] is False
    assert second_payload["signal"]["bar_ts"] == first_signal["bar_ts"]
    assert second_payload["signal"]["signal"] == first_signal["signal"]


def test_predict_if_stale_idempotent_appends_one_row(
    authed_client: TestClient,
    trained_store: Path,
    stubbed_fetcher: tuple[pd.DataFrame, pd.Timestamp],
) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()
    deployment_id = created["id"]
    authed_client.post(f"{DEPLOYMENTS_PATH}/{deployment_id}/predict-if-stale")
    authed_client.post(f"{DEPLOYMENTS_PATH}/{deployment_id}/predict-if-stale")
    authed_client.post(f"{DEPLOYMENTS_PATH}/{deployment_id}/predict-if-stale")

    signals = authed_client.get(f"{DEPLOYMENTS_PATH}/{deployment_id}/signals").json()
    assert len(signals) == 1


def test_predict_if_stale_404_for_other_user(
    authed_client: TestClient,
    trained_store: Path,
    db_conn: sqlite3.Connection,
    client: TestClient,
    stubbed_fetcher: tuple[pd.DataFrame, pd.Timestamp],
) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()
    deployment_id = created["id"]

    _create_secondary_user(db_conn)
    authed_client.post("/api/auth/logout")
    _login(client, SECONDARY_USERNAME, SECONDARY_PASSWORD)

    response = client.post(f"{DEPLOYMENTS_PATH}/{deployment_id}/predict-if-stale")
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_detail_surfaces_latest_signal_after_predict(
    authed_client: TestClient,
    trained_store: Path,
    stubbed_fetcher: tuple[pd.DataFrame, pd.Timestamp],
) -> None:
    created = authed_client.post(
        DEPLOYMENTS_PATH, json={"source_kind": "run", "source_id": _RUN_ID}
    ).json()
    deployment_id = created["id"]
    authed_client.post(f"{DEPLOYMENTS_PATH}/{deployment_id}/predict-if-stale")

    detail = authed_client.get(f"{DEPLOYMENTS_PATH}/{deployment_id}").json()
    assert detail["latest_signal"] is not None
    assert "bar_ts" in detail["latest_signal"]


