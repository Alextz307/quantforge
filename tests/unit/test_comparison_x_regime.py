"""Tests for the regime overlay path in :func:`run_comparison`.

Exercises the strategy x regime aggregation, the same-data validation,
the data-hash drift guard, and the reporter outputs (heatmap PNG +
LaTeX table). The bars-fetch helper is monkeypatched so tests don't
fan out to the data-source registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.analysis.metrics_aggregator import aggregate_folds
from src.core.config import ExperimentConfig
from src.core.regime_config import RegimeConfig
from src.data.fingerprint import fingerprint_bars
from src.orchestration import comparison as comparison_mod
from src.orchestration.comparison import SignificanceTest, run_comparison
from src.orchestration.experiment import RunOptions
from src.orchestration.types import (
    MIXED_REGIME_LABEL,
    ExperimentResult,
    FoldRecord,
    StrategyComparisonReport,
)
from src.visualization.comparison_reporter import ComparisonReporter
from tests.conftest import (
    comparison_curve_seed,
    make_log_return_equity_curve,
    make_stub_experiment_result,
    make_stub_fold_record,
)

_BARS_START = pd.Timestamp("2020-01-01")
_BARS_PERIODS = 200
_BARS_FREQ = "B"
_BARS_BASE_PRICE = 100.0
_BARS_RETURN_STD = 0.01
_BARS_SEED = 31

_FOLD_TEST_BARS = 30
_FOLD_HEADROOM_BARS = 30  # leaves train history to the left of the first fold
_N_FOLDS = 4
_FOLD_CURVE_LENGTH = 40

_PERIOD_BOUNDARY_FRACTION = 0.5
_STRADDLE_HALF_WIDTH = 15  # bars on each side of the period midpoint

_DRIFT_DATA_HASH = "0" * 64

_SHARPE_BY_NAME = {"Alpha": 1.5, "Bravo": 0.6}
_STRADDLE_SHARPE = 1.0


def _make_bars(periods: int = _BARS_PERIODS) -> pd.DataFrame:
    rng = np.random.default_rng(_BARS_SEED)
    idx = pd.bdate_range(start=_BARS_START, periods=periods, freq=_BARS_FREQ)
    returns = rng.normal(0.0, _BARS_RETURN_STD, periods)
    close = _BARS_BASE_PRICE * np.cumprod(1.0 + returns)
    open_ = np.empty(periods)
    open_[0] = _BARS_BASE_PRICE
    open_[1:] = close[:-1]
    band = np.abs(rng.normal(0.0, _BARS_RETURN_STD * 0.5, periods))
    high = np.maximum(open_, close) * (1.0 + band)
    low = np.minimum(open_, close) * (1.0 - band)
    volume = np.full(periods, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class _FakeSource:
    name = "csv"
    params: dict[str, object] = {}


class _FakeInterval:
    value = "daily"


class _FakeData:
    def __init__(self, *, tickers: tuple[str, ...]) -> None:
        self.source = _FakeSource()
        self.tickers = list(tickers)
        self.start = _BARS_START.to_pydatetime()
        self.end = (_BARS_START + pd.Timedelta(days=400)).to_pydatetime()
        self.interval = _FakeInterval()


class _FakeCfg:
    def __init__(self, *, name: str, data: _FakeData) -> None:
        self.name = name
        self.data = data


def _make_fake_configs(
    *names: str,
    tickers_by_name: dict[str, tuple[str, ...]] | None = None,
) -> list[ExperimentConfig]:
    """Build duck-typed configs and cast once at the boundary.

    ``run_comparison`` only reads ``cfg.name`` + ``cfg.data`` — wiring up
    the full Pydantic ``ExperimentConfig`` would force registry init for
    no test value. The cast is the single bridge between the structural
    ``_FakeCfg`` and the nominal ``ExperimentConfig`` parameter type.
    """
    cfgs = [
        _FakeCfg(
            name=name,
            data=_FakeData(
                tickers=(tickers_by_name or {}).get(name, ("SPY",)),
            ),
        )
        for name in names
    ]
    return cast(list[ExperimentConfig], cfgs)


def _make_period_regime_config(bars: pd.DataFrame) -> RegimeConfig:
    midpoint = bars.index[int(len(bars) * _PERIOD_BOUNDARY_FRACTION)]
    end = bars.index[-1] + pd.Timedelta(days=1)
    return RegimeConfig.model_validate(
        {
            "detector": {
                "name": "period",
                "params": {
                    "boundaries": [
                        {
                            "label": "early",
                            "start": str(bars.index[0]),
                            "end": str(midpoint),
                        },
                        {
                            "label": "late",
                            "start": str(midpoint),
                            "end": str(end),
                        },
                    ]
                },
            }
        }
    )


def _build_aligned_folds(bars: pd.DataFrame, sharpe: float, name: str) -> tuple[FoldRecord, ...]:
    """Folds whose test windows tile ``bars`` consecutively after a headroom prefix."""
    test_starts = bars.index[
        _FOLD_HEADROOM_BARS : _FOLD_HEADROOM_BARS + _N_FOLDS * _FOLD_TEST_BARS : _FOLD_TEST_BARS
    ]
    folds: list[FoldRecord] = []
    for i, test_start_raw in enumerate(test_starts):
        test_start_ts = pd.Timestamp(test_start_raw)
        loc = bars.index.get_loc(test_start_ts)
        assert isinstance(loc, int)
        test_end_ts = pd.Timestamp(bars.index[loc + _FOLD_TEST_BARS])
        folds.append(
            make_stub_fold_record(
                i,
                sharpe=sharpe,
                equity_curve=make_log_return_equity_curve(
                    sharpe, n=_FOLD_CURVE_LENGTH, seed=comparison_curve_seed(name, i)
                ),
                test_start=test_start_ts,
                test_end=test_end_ts,
            )
        )
    return tuple(folds)


def _stub_result(name: str, sharpe: float, bars: pd.DataFrame) -> ExperimentResult:
    return make_stub_experiment_result(
        name,
        folds=_build_aligned_folds(bars, sharpe, name),
        data_hash=fingerprint_bars(bars),
    )


class _StubExperiment:
    def __init__(self, name: str, sharpe: float, bars: pd.DataFrame) -> None:
        self._name = name
        self._sharpe = sharpe
        self._bars = bars

    def run(self, options: RunOptions | None = None) -> ExperimentResult:
        opts = options if options is not None else RunOptions()
        assert opts.write_report is False
        return _stub_result(self._name, self._sharpe, self._bars)


@pytest.fixture
def bars() -> pd.DataFrame:
    return _make_bars()


@pytest.fixture
def patched_bars(monkeypatch: pytest.MonkeyPatch, bars: pd.DataFrame) -> pd.DataFrame:
    """Monkeypatch the overlay's bar-fetcher so tests skip the registry."""
    monkeypatch.setattr(comparison_mod, "_fetch_overlay_bars", lambda _data_cfg: bars)
    return bars


@pytest.fixture
def patched_build(monkeypatch: pytest.MonkeyPatch, bars: pd.DataFrame) -> dict[str, float]:
    sharpe_by_name: dict[str, float] = {}

    def _fake_build(cfg: ExperimentConfig) -> _StubExperiment:
        return _StubExperiment(
            name=cfg.name,
            sharpe=sharpe_by_name[cfg.name],
            bars=bars,
        )

    monkeypatch.setattr(comparison_mod, "build_experiment", _fake_build)
    return sharpe_by_name


class TestRegimeOverlay:
    def test_overlay_populated_with_per_strategy_per_regime_stats(
        self,
        tmp_path: Path,
        bars: pd.DataFrame,
        patched_build: dict[str, float],
        patched_bars: pd.DataFrame,
    ) -> None:
        for name, sharpe in _SHARPE_BY_NAME.items():
            patched_build[name] = sharpe

        report, _ = run_comparison(
            _make_fake_configs("Alpha", "Bravo"),
            out_name="overlay",
            store_root=tmp_path,
            significance_test=SignificanceTest.NONE,
            regime_config=_make_period_regime_config(bars),
        )

        assert report.per_strategy_per_regime_stats is not None
        overlay = report.per_strategy_per_regime_stats
        assert set(overlay.keys()) == set(_SHARPE_BY_NAME.keys())
        for per_regime in overlay.values():
            assert {"early", "late"}.issubset(set(per_regime.keys()))

    def test_overlay_aggregates_match_per_strategy_split(
        self,
        tmp_path: Path,
        bars: pd.DataFrame,
        patched_build: dict[str, float],
        patched_bars: pd.DataFrame,
    ) -> None:
        for name, sharpe in _SHARPE_BY_NAME.items():
            patched_build[name] = sharpe

        report, folds_by_strategy = run_comparison(
            _make_fake_configs("Alpha", "Bravo"),
            out_name="agg",
            store_root=tmp_path,
            significance_test=SignificanceTest.NONE,
            regime_config=_make_period_regime_config(bars),
        )

        midpoint = bars.index[int(len(bars) * _PERIOD_BOUNDARY_FRACTION)]
        alpha_folds = folds_by_strategy["Alpha"]
        early_folds = tuple(f for f in alpha_folds if f.test_end <= midpoint)
        expected_early = aggregate_folds(early_folds)
        assert report.per_strategy_per_regime_stats is not None
        observed_early = report.per_strategy_per_regime_stats["Alpha"]["early"]
        assert observed_early.sharpe_mean == pytest.approx(expected_early.sharpe_mean)
        assert observed_early.n_folds == expected_early.n_folds


class TestUniformDataValidation:
    def test_rejects_configs_with_different_tickers(
        self,
        tmp_path: Path,
        bars: pd.DataFrame,
        patched_build: dict[str, float],
        patched_bars: pd.DataFrame,
    ) -> None:
        for name, sharpe in _SHARPE_BY_NAME.items():
            patched_build[name] = sharpe

        with pytest.raises(ValueError, match="same data block"):
            run_comparison(
                _make_fake_configs(
                    "Alpha",
                    "Bravo",
                    tickers_by_name={"Alpha": ("SPY",), "Bravo": ("QQQ",)},
                ),
                out_name="bad-data",
                store_root=tmp_path,
                significance_test=SignificanceTest.NONE,
                regime_config=_make_period_regime_config(bars),
            )


class TestDataHashGuard:
    def test_rejects_drift_between_manifest_and_refetched_bars(
        self,
        tmp_path: Path,
        bars: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        patched_bars: pd.DataFrame,
    ) -> None:
        def _fake_build_with_drift(cfg: ExperimentConfig) -> _StubExperiment:
            class _Drifted(_StubExperiment):
                def run(self, options: RunOptions | None = None) -> ExperimentResult:
                    return make_stub_experiment_result(
                        self._name,
                        folds=_build_aligned_folds(self._bars, self._sharpe, self._name),
                        data_hash=_DRIFT_DATA_HASH,
                    )

            return _Drifted(name=cfg.name, sharpe=1.0, bars=bars)

        monkeypatch.setattr(comparison_mod, "build_experiment", _fake_build_with_drift)

        with pytest.raises(ValueError, match="data_hash drift"):
            run_comparison(
                _make_fake_configs("Alpha", "Bravo"),
                out_name="drift",
                store_root=tmp_path,
                significance_test=SignificanceTest.NONE,
                regime_config=_make_period_regime_config(bars),
            )


class TestReporterOutputs:
    def test_reporter_emits_heatmap_and_latex_when_overlay_present(
        self,
        tmp_path: Path,
        bars: pd.DataFrame,
        patched_build: dict[str, float],
        patched_bars: pd.DataFrame,
    ) -> None:
        for name, sharpe in _SHARPE_BY_NAME.items():
            patched_build[name] = sharpe

        report, folds_by_strategy = run_comparison(
            _make_fake_configs("Alpha", "Bravo"),
            out_name="report",
            store_root=tmp_path,
            significance_test=SignificanceTest.NONE,
            regime_config=_make_period_regime_config(bars),
        )

        out_dir = tmp_path / "comparisons" / "report"
        ComparisonReporter().generate_full_report(
            report, out_dir, folds_by_strategy=folds_by_strategy
        )

        heatmap_png = out_dir / "plots" / "strategy_x_regime_heatmap.png"
        heatmap_svg = out_dir / "plots" / "strategy_x_regime_heatmap.svg"
        latex_table = out_dir / "tables" / "strategy_x_regime.tex"
        assert heatmap_png.is_file() and heatmap_png.stat().st_size > 0
        assert heatmap_svg.is_file() and heatmap_svg.stat().st_size > 0
        assert latex_table.is_file()
        latex_text = latex_table.read_text(encoding="utf-8")
        assert "Alpha" in latex_text and "Bravo" in latex_text
        assert "early" in latex_text and "late" in latex_text

    def test_manifest_includes_overlay_payload(
        self,
        tmp_path: Path,
        bars: pd.DataFrame,
        patched_build: dict[str, float],
        patched_bars: pd.DataFrame,
    ) -> None:
        for name, sharpe in _SHARPE_BY_NAME.items():
            patched_build[name] = sharpe

        report, _ = run_comparison(
            _make_fake_configs("Alpha", "Bravo"),
            out_name="manifest",
            store_root=tmp_path,
            significance_test=SignificanceTest.NONE,
            regime_config=_make_period_regime_config(bars),
        )
        out_dir = tmp_path / "comparisons" / "manifest"
        ComparisonReporter().generate_full_report(report, out_dir)

        import json

        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "per_strategy_per_regime_stats" in manifest
        assert set(manifest["per_strategy_per_regime_stats"]) == set(_SHARPE_BY_NAME)


class TestNoOverlayWhenRegimeConfigNone:
    def test_field_defaults_to_none_when_regime_config_omitted(
        self, tmp_path: Path, patched_build: dict[str, float]
    ) -> None:
        for name, sharpe in _SHARPE_BY_NAME.items():
            patched_build[name] = sharpe

        report, _ = run_comparison(
            _make_fake_configs("Alpha", "Bravo"),
            out_name="no-overlay",
            store_root=tmp_path,
            significance_test=SignificanceTest.NONE,
        )
        assert isinstance(report, StrategyComparisonReport)
        assert report.per_strategy_per_regime_stats is None


class TestMixedFoldRow:
    def test_mixed_fold_appears_when_majority_threshold_not_met(
        self,
        tmp_path: Path,
        bars: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        patched_bars: pd.DataFrame,
    ) -> None:
        # The straddle window is centred on the period boundary so
        # neither regime hits the splitter's 0.6 majority threshold.
        midpoint_pos = int(len(bars) * _PERIOD_BOUNDARY_FRACTION)
        straddle_start = bars.index[midpoint_pos - _STRADDLE_HALF_WIDTH]
        straddle_end = bars.index[midpoint_pos + _STRADDLE_HALF_WIDTH]
        straddle_fold = make_stub_fold_record(
            0,
            sharpe=_STRADDLE_SHARPE,
            equity_curve=make_log_return_equity_curve(
                _STRADDLE_SHARPE,
                n=_FOLD_CURVE_LENGTH,
                seed=comparison_curve_seed("straddle", 0),
            ),
            test_start=straddle_start,
            test_end=straddle_end,
        )

        def _fake_build_straddle(cfg: ExperimentConfig) -> _StubExperiment:
            class _Straddle(_StubExperiment):
                def run(self, options: RunOptions | None = None) -> ExperimentResult:
                    return make_stub_experiment_result(
                        self._name,
                        folds=(straddle_fold,),
                        data_hash=fingerprint_bars(self._bars),
                    )

            return _Straddle(name=cfg.name, sharpe=_STRADDLE_SHARPE, bars=bars)

        monkeypatch.setattr(comparison_mod, "build_experiment", _fake_build_straddle)

        report, _ = run_comparison(
            _make_fake_configs("Alpha", "Bravo"),
            out_name="mixed",
            store_root=tmp_path,
            significance_test=SignificanceTest.NONE,
            regime_config=_make_period_regime_config(bars),
        )
        assert report.per_strategy_per_regime_stats is not None
        for per_regime in report.per_strategy_per_regime_stats.values():
            assert MIXED_REGIME_LABEL in per_regime
