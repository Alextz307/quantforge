"""
Round-trip tests for :class:`FoldRecord` and :class:`ExperimentResult`.

These records are the JSONL schema for ``fold_results.jsonl`` / experiment
manifests; a broken ``from_dict`` surfaces only at report-reload time which
is the worst time. Covered here: field-by-field construction + round-trip,
``from_fold_result`` on a stubbed FoldResult, malformed-input rejection via
``src.core.json_io`` typed accessors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.engine.scenarios import SlippageScenario
from src.orchestration.manifest import Manifest
from src.orchestration.types import ExperimentResult, FoldRecord

_FOLD_INDEX = 3
_TRAIN_START_ISO = "2020-01-02T00:00:00"
_TRAIN_END_ISO = "2022-12-30T00:00:00"
_TEST_START_ISO = "2023-01-03T00:00:00"
_TEST_END_ISO = "2023-12-29T00:00:00"
_TOTAL_RETURN = 0.45
_ANNUALIZED_RETURN = 0.12
_ANNUALIZED_VOLATILITY = 0.18
_SHARPE = 0.67
_SORTINO = 0.89
_CALMAR = 0.34
_MAX_DRAWDOWN = -0.22
_WIN_RATE = 0.55
_TRADE_COUNT = 42
_EQUITY_CURVE = (10_000.0, 10_100.5, 10_050.25, 10_200.0)
_EXPERIMENT_ID = "20260422_143000_AdaptiveBollinger_abc1234"


def _make_manifest(experiment_id: str = _EXPERIMENT_ID) -> Manifest:
    return Manifest(
        experiment_id=experiment_id,
        name="types_test",
        created_at=datetime(2026, 4, 22, 14, 30),
        git_sha="abc1234",
        seed=42,
        data_hash="deadbeef",
        slippage_scenario=SlippageScenario.NORMAL,
        holdout_start=pd.Timestamp("2023-06-30T00:00:00"),
    )


def _make_record(**overrides: Any) -> FoldRecord:
    fields: dict[str, Any] = {
        "fold_index": _FOLD_INDEX,
        "train_start": pd.Timestamp(_TRAIN_START_ISO),
        "train_end": pd.Timestamp(_TRAIN_END_ISO),
        "test_start": pd.Timestamp(_TEST_START_ISO),
        "test_end": pd.Timestamp(_TEST_END_ISO),
        "total_return": _TOTAL_RETURN,
        "annualized_return": _ANNUALIZED_RETURN,
        "annualized_volatility": _ANNUALIZED_VOLATILITY,
        "sharpe_ratio": _SHARPE,
        "sortino_ratio": _SORTINO,
        "calmar_ratio": _CALMAR,
        "max_drawdown": _MAX_DRAWDOWN,
        "win_rate": _WIN_RATE,
        "trade_count": _TRADE_COUNT,
        "equity_curve": _EQUITY_CURVE,
    }
    fields.update(overrides)
    return FoldRecord(**fields)


class _StubBacktest:
    """
    Attribute-only stub for ``BacktestResult`` (C++ binding not instantiable).
    """

    def __init__(self) -> None:
        self.total_return = _TOTAL_RETURN
        self.trade_count = _TRADE_COUNT
        self.equity_curve = np.asarray(_EQUITY_CURVE, dtype=np.float64)


class _StubMetrics:
    """
    Attribute-only stub for ``PerformanceMetrics``.
    """

    def __init__(self) -> None:
        self.annualized_return = _ANNUALIZED_RETURN
        self.annualized_volatility = _ANNUALIZED_VOLATILITY
        self.sharpe_ratio = _SHARPE
        self.sortino_ratio = _SORTINO
        self.calmar_ratio = _CALMAR
        self.max_drawdown = _MAX_DRAWDOWN
        self.win_rate = _WIN_RATE


class _StubFoldResult:
    """
    Duck-typed stub matching the attribute surface ``FoldRecord.from_fold_result`` reads.

    ``FoldResult`` is a frozen dataclass whose ``backtest`` / ``metrics``
    fields are C++-bound classes not instantiable from Python. Passing a
    structural stub avoids the frozen-dataclass ``__new__`` + setattr hack
    and keeps the test readable; ``from_fold_result`` itself only does
    ``fr.backtest.<x>`` / ``fr.metrics.<y>`` attribute access, so structural
    typing is sufficient.
    """

    def __init__(self, strategy_diagnostics: dict[str, float] | None = None) -> None:
        self.fold_index = _FOLD_INDEX
        self.train_start = pd.Timestamp(_TRAIN_START_ISO)
        self.train_end = pd.Timestamp(_TRAIN_END_ISO)
        self.test_start = pd.Timestamp(_TEST_START_ISO)
        self.test_end = pd.Timestamp(_TEST_END_ISO)
        self.backtest = _StubBacktest()
        self.metrics = _StubMetrics()
        self.strategy_diagnostics = strategy_diagnostics if strategy_diagnostics is not None else {}


class TestFoldRecordFromFoldResult:
    def test_flattens_backtest_and_metrics(self) -> None:
        rec = FoldRecord.from_fold_result(_StubFoldResult())  # type: ignore[arg-type]

        assert rec.fold_index == _FOLD_INDEX
        assert rec.total_return == pytest.approx(_TOTAL_RETURN)
        assert rec.sharpe_ratio == pytest.approx(_SHARPE)
        assert rec.calmar_ratio == pytest.approx(_CALMAR)
        assert rec.trade_count == _TRADE_COUNT
        assert rec.equity_curve == _EQUITY_CURVE

    def test_equity_curve_is_tuple_of_python_floats(self) -> None:
        """
        ``.tolist()`` must yield native Python floats, not numpy scalars —
        otherwise JSON serialisation downstream produces numpy-specific
        literals that don't round-trip."""

        rec = FoldRecord.from_fold_result(_StubFoldResult())  # type: ignore[arg-type]
        assert isinstance(rec.equity_curve, tuple)
        for value in rec.equity_curve:
            assert type(value) is float


class TestFoldRecordRoundTrip:
    def test_to_dict_keys_are_exhaustive(self) -> None:
        rec = _make_record()
        expected = {
            "fold_index",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "win_rate",
            "trade_count",
            "equity_curve",
            "strategy_diagnostics",
        }
        assert set(rec.to_dict().keys()) == expected

    def test_diagnostics_round_trip_preserves_keys_and_values(self) -> None:
        original = _make_record(strategy_diagnostics={"floor_bind_fraction": 0.123})
        revived = FoldRecord.from_dict(original.to_dict())
        assert dict(revived.strategy_diagnostics) == {"floor_bind_fraction": 0.123}

    def test_diagnostics_missing_in_dict_defaults_to_empty(self) -> None:
        """
        Backwards-compat: existing JSONL files (no strategy_diagnostics
        key) deserialize cleanly to an empty mapping."""

        d = _make_record().to_dict()
        del d["strategy_diagnostics"]
        revived = FoldRecord.from_dict(d)
        assert dict(revived.strategy_diagnostics) == {}

    def test_timestamps_serialize_as_iso_strings(self) -> None:
        d = _make_record().to_dict()
        assert d["train_start"] == _TRAIN_START_ISO
        assert d["test_end"] == _TEST_END_ISO

    def test_equity_curve_serializes_as_list(self) -> None:
        d = _make_record().to_dict()
        assert d["equity_curve"] == list(_EQUITY_CURVE)

    def test_roundtrip_preserves_every_field(self) -> None:
        original = _make_record()
        revived = FoldRecord.from_dict(original.to_dict())
        assert revived == original

    def test_from_dict_rejects_non_list_equity_curve(self) -> None:
        d = _make_record().to_dict()
        d["equity_curve"] = "not_a_list"
        with pytest.raises(ValueError, match="equity_curve"):
            FoldRecord.from_dict(d)

    def test_from_dict_rejects_non_numeric_equity_curve_entry(self) -> None:
        d = _make_record().to_dict()
        d["equity_curve"] = [10_000.0, "oops", 10_200.0]
        with pytest.raises(ValueError, match="equity_curve"):
            FoldRecord.from_dict(d)

    def test_from_dict_rejects_missing_field(self) -> None:
        d = _make_record().to_dict()
        del d["sharpe_ratio"]
        with pytest.raises(KeyError, match="sharpe_ratio"):
            FoldRecord.from_dict(d)


class TestExperimentResultRoundTrip:
    def test_empty_folds_roundtrip(self) -> None:
        result = ExperimentResult(experiment_id=_EXPERIMENT_ID, folds=(), manifest=_make_manifest())
        revived = ExperimentResult.from_dict(result.to_dict())
        assert revived == result

    def test_multi_fold_roundtrip_preserves_order(self) -> None:
        folds = tuple(_make_record(fold_index=i) for i in range(3))
        result = ExperimentResult(
            experiment_id=_EXPERIMENT_ID, folds=folds, manifest=_make_manifest()
        )
        revived = ExperimentResult.from_dict(result.to_dict())

        assert revived == result
        assert tuple(f.fold_index for f in revived.folds) == (0, 1, 2)
        assert revived.manifest.slippage_scenario.value == "normal"

    def test_to_dict_folds_are_plain_dicts(self) -> None:
        """
        JSON-compatibility sanity check: every fold in ``.to_dict()`` is
        a JSON object (dict), not a dataclass instance."""

        result = ExperimentResult(
            experiment_id=_EXPERIMENT_ID,
            folds=(_make_record(),),
            manifest=_make_manifest(),
        )
        d = result.to_dict()
        assert isinstance(d["folds"], list)
        assert all(isinstance(f, dict) for f in d["folds"])

    def test_from_dict_rejects_non_list_folds(self) -> None:
        with pytest.raises(ValueError, match="folds"):
            ExperimentResult.from_dict(
                {
                    "experiment_id": _EXPERIMENT_ID,
                    "folds": "not_a_list",
                    "manifest": _make_manifest().to_dict(),
                }
            )

    def test_from_dict_rejects_non_dict_manifest(self) -> None:
        with pytest.raises(ValueError, match="manifest"):
            ExperimentResult.from_dict(
                {
                    "experiment_id": _EXPERIMENT_ID,
                    "folds": [],
                    "manifest": "not_a_dict",
                }
            )
