"""End-to-end smoke test for the study orchestrator (``experiment study run``).

Drives the full CLI stack against a tiny synthetic CSV fixture: writes
two CSV-backed universe profiles + a 1-strategy x 2-universe study spec,
then runs ``experiment study run --skip-compares`` and asserts the
per-leg artifact tree, the ``study_state.json`` shape, and resume
semantics on a second invocation.

Opt-in via ``RUN_STUDY_SMOKE=1`` (mirrors ``RUN_EXP_SMOKE`` on the run
smoke) because the inner HPO + run stages take ~30-60s on a tiny config.
Cross-strategy compare is exercised by ``test_comparison.py`` and is
deliberately NOT in scope here — the orchestrator's compare step is a
thin wrapper over ``run_comparison``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from src.core.persistence import HPO_SUBDIR, RUNS_SUBDIR
from src.orchestration.study_state import (
    LEG_STEP_HOLDOUT_EVAL,
    LEG_STEP_REGIME,
    LEG_STEP_RUN,
    LEG_STEP_TUNE,
    read_study_state,
)
from tests.conftest import make_synthetic_ohlcv_df

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_STUDY_SMOKE") != "1",
    reason="set RUN_STUDY_SMOKE=1 to run the study orchestrator smoke test",
)

_TICKER_A = "MINIA"
_TICKER_B = "MINIB"
_N_BARS = 300
_N_TRIALS = 2
_N_SPLITS = 2
_TEST_SIZE = 50
_GAP = 1
_HOLDOUT_PCT = 0.20


def _write_universe(
    path: Path, ticker: str, csv_dir: Path, holdout_pct: float = _HOLDOUT_PCT
) -> None:
    """Render a universe profile YAML pointing at a CSV fixture."""
    payload: dict[str, object] = {
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": [ticker],
            "start": datetime(2020, 1, 2).isoformat(),
            "end": datetime(2022, 1, 1).isoformat(),
            "interval": "daily",
        },
        "validation": {
            "n_splits": _N_SPLITS,
            "test_size": _TEST_SIZE,
            "gap": _GAP,
            "holdout_pct": holdout_pct,
        },
    }
    path.write_text(yaml.safe_dump(payload, default_flow_style=False))


def _write_strategy(path: Path) -> None:
    """Render a tiny AdaptiveBollinger strategy YAML (no data block).

    The orchestrator's ``compose_leg_config`` overlays the universe's
    ``data`` + ``validation`` blocks, so the strategy YAML only needs
    the strategy block + slippage. ExperimentConfig validation requires
    a ``data`` block at parse time, so we leave a placeholder that the
    deep-merge always overrides.
    """
    payload: dict[str, object] = {
        "name": "_placeholder",
        "seed": 42,
        "data": {
            "source": "csv",
            "tickers": ["PLACEHOLDER"],
            "start": datetime(2020, 1, 2).isoformat(),
            "end": datetime(2022, 1, 1).isoformat(),
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {
                "window": 20,
                "trend_window": 50,
                "garch_p_max": 1,
                "garch_q_max": 1,
            },
        },
        "validation": {
            "n_splits": _N_SPLITS,
            "test_size": _TEST_SIZE,
            "gap": _GAP,
        },
        "slippage": {"scenario": "normal"},
    }
    path.write_text(yaml.safe_dump(payload, default_flow_style=False))


def _write_hpo(path: Path) -> None:
    """Tiny HPO config — n_trials=2 to keep the smoke under a minute."""
    payload: dict[str, object] = {
        "study_name": "_placeholder_will_be_overridden",
        "n_trials": _N_TRIALS,
        "n_jobs": 1,
        "sampler": "tpe",
        "pruner": "median",
        "objective": "sharpe",
        "seed": 42,
    }
    path.write_text(yaml.safe_dump(payload, default_flow_style=False))


def _write_csv(csv_dir: Path, ticker: str, *, seed_offset: int) -> None:
    """Synthesise a 300-bar OHLCV CSV under ``csv_dir/<ticker>.csv``."""
    df = make_synthetic_ohlcv_df(n_rows=_N_BARS, start="2020-01-02", seed=42 + seed_offset)
    df.index.name = "date"
    df.to_csv(csv_dir / f"{ticker}.csv")


def _build_smoke_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Materialise CSVs + universe / strategy / hpo / spec YAMLs.

    Layout (tests chdir to ``tmp_path`` so ``config/universes/`` resolves
    against the orchestrator's ``<repo_root>/config/universes/<name>.yaml``
    lookup convention)::

        tmp_path/
          csv_data/{MINIA,MINIB}.csv
          config/
            universes/{minia,minib}_smoke.yaml
            strategies/ab_smoke.yaml
            hpo/ab_smoke.yaml
            study/smoke_spec.yaml
          store/                    (populated by run_study)

    Returns ``(spec_path, store_root)`` ready to feed to the CLI.
    """
    csv_dir = tmp_path / "csv_data"
    csv_dir.mkdir()
    _write_csv(csv_dir, _TICKER_A, seed_offset=0)
    _write_csv(csv_dir, _TICKER_B, seed_offset=1)

    cfg_dir = tmp_path / "config"
    universes_dir = cfg_dir / "universes"
    strategies_dir = cfg_dir / "strategies"
    hpo_dir = cfg_dir / "hpo"
    study_dir = cfg_dir / "study"
    for d in (universes_dir, strategies_dir, hpo_dir, study_dir):
        d.mkdir(parents=True)

    _write_universe(universes_dir / f"{_TICKER_A.lower()}_smoke.yaml", _TICKER_A, csv_dir)
    _write_universe(universes_dir / f"{_TICKER_B.lower()}_smoke.yaml", _TICKER_B, csv_dir)

    strategy_path = strategies_dir / "ab_smoke.yaml"
    _write_strategy(strategy_path)

    hpo_path = hpo_dir / "ab_smoke.yaml"
    _write_hpo(hpo_path)

    spec_payload: dict[str, object] = {
        "name": "smoke_study",
        "output_dir": str(tmp_path / "store" / "studies" / "smoke"),
        "legs": [
            {
                "strategy": "AdaptiveBollinger",
                "strategy_config": str(strategy_path),
                "hpo_config": str(hpo_path),
                "universes": [f"{_TICKER_A.lower()}_smoke", f"{_TICKER_B.lower()}_smoke"],
            },
        ],
    }
    spec_path = study_dir / "smoke_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_payload, default_flow_style=False))

    return spec_path, tmp_path / "store"


def test_study_run_produces_complete_leg_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: 2 legs run, both complete, state file round-trips."""
    spec_path, store = _build_smoke_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    # We invoke the CLI directly (Click exit-code semantics catch any
    # ClickException; bare-fail orchestration bugs surface here, not silent).
    from scripts.experiment import cli as experiment_cli

    runner = CliRunner()
    result = runner.invoke(
        experiment_cli,
        [
            "study",
            "run",
            "--spec",
            str(spec_path),
            "--store-root",
            str(store),
            "--skip-compares",
        ],
    )
    assert result.exit_code == 0, result.output

    study_dir = store / "studies" / "smoke"
    state = read_study_state(study_dir / "study_state.json")
    assert state.spec_name == "smoke_study"
    assert len(state.legs) == 2
    for leg in state.legs:
        assert leg.is_complete, f"leg {leg.leg_id} did not complete: error={leg.error}"
        assert leg.run_experiment_id is not None
        # Holdout-eval was on (universe.holdout_pct=0.20), regime off.
        assert LEG_STEP_TUNE in leg.steps_completed
        assert LEG_STEP_RUN in leg.steps_completed
        assert LEG_STEP_REGIME in leg.steps_completed
        assert LEG_STEP_HOLDOUT_EVAL in leg.steps_completed

    # Per-leg artifact directories materialised.
    for leg in state.legs:
        assert (study_dir / HPO_SUBDIR / leg.leg_id).is_dir()
        assert leg.run_experiment_id is not None
        assert (study_dir / RUNS_SUBDIR / leg.run_experiment_id).is_dir()


def test_study_run_resume_skips_complete_legs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second invocation against the same store skips legs flagged complete."""
    spec_path, store = _build_smoke_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    from scripts.experiment import cli as experiment_cli

    runner = CliRunner()
    first = runner.invoke(
        experiment_cli,
        ["study", "run", "--spec", str(spec_path), "--store-root", str(store), "--skip-compares"],
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        experiment_cli,
        ["study", "run", "--spec", str(spec_path), "--store-root", str(store), "--skip-compares"],
    )
    assert second.exit_code == 0, second.output
    # Resume should report 0 newly-completed + 2 skipped (both pre-existing).
    assert "completed:    0" in second.output
    assert "skipped:      2" in second.output
