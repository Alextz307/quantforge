"""Tests for the multi-feature single-asset dispatch path.

Verifies the third dispatcher arm: strategies that read a wide multi-ticker
frame for features but trade exactly one asset. The C++ engine sees only
the primary asset's OHLCV — companion tickers never enter its books.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from quant_engine import SlippageConfig, SlippageModel
from src.core.config import ExperimentConfig
from src.core.constants import OHLCV_COLUMNS
from src.core.temporal import WalkForwardValidator
from src.core.types import Interval
from src.data.fingerprint import fingerprint_multi_bars
from src.data.interface import IDataSource
from src.engine.cpp_engine import CppBacktestEngine
from src.engine.walk_forward import evaluate_walk_forward, slice_primary_ohlcv
from src.orchestration.builder import _validate_strategy_data_shape
from src.orchestration.experiment import fetch_bars
from tests._strategy_stubs import MultiFeatureTestStub
from tests.conftest import GLOBAL_NUMPY_SEED, make_synthetic_ohlcv_df

_PRIMARY = "AAA"
_FEATURE = "BBB"
_THIRD = "CCC"
_N_BARS = 200
_N_SPLITS = 2
_TEST_SIZE = 50
_GAP = 2
_START = datetime(2022, 1, 3)
_END = datetime(2023, 12, 31)
_BPS_ZERO = 0.0
_VOLUME_IMPACT_ZERO = 0.0
_FAKE_SOURCE_SEED_MASK = 0xFFFF


def _wide_frame(tickers: Sequence[str], *, seed: int = GLOBAL_NUMPY_SEED) -> pd.DataFrame:
    """Build a wide ``<col>_<TICKER>`` frame for ``tickers``."""

    suffixed = [
        make_synthetic_ohlcv_df(n_rows=_N_BARS, seed=seed + offset).add_suffix(f"_{ticker}")
        for offset, ticker in enumerate(tickers)
    ]
    joined = suffixed[0]
    for other in suffixed[1:]:
        joined = joined.join(other, how="inner")
    return joined


class _FakeDataSource(IDataSource):
    """Returns a deterministic synthetic OHLCV per ticker; ignores date range.

    Overrides ``fetch`` directly so the parent's normalize / validate pipeline
    never runs on the synthetic frame (which already has canonical columns).
    """

    name: ClassVar[str] = "fake"

    def __init__(self) -> None:
        self.call_log: list[str] = []
        # Skip parent __init__: it constructs a DataNormalizer that uses
        # ``self.name`` — bypass keeps this fixture independent of the
        # normalizer registry.
        self.cache = None

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        self.call_log.append(ticker)
        return make_synthetic_ohlcv_df(n_rows=_N_BARS, seed=hash(ticker) & _FAKE_SOURCE_SEED_MASK)

    def fetch_raw(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:  # pragma: no cover — fetch() bypasses raw
        return self.fetch(ticker, start, end, interval)

    def available_tickers(self) -> list[str]:  # pragma: no cover
        return [_PRIMARY, _FEATURE, _THIRD]


class TestSlicePrimaryOhlcv:
    def test_returns_canonical_ohlcv_columns(self) -> None:
        wide = _wide_frame((_PRIMARY, _FEATURE))
        sliced = slice_primary_ohlcv(wide, _PRIMARY)

        assert list(sliced.columns) == list(OHLCV_COLUMNS)
        assert len(sliced) == len(wide)

    def test_values_match_primary_columns(self) -> None:
        wide = _wide_frame((_PRIMARY, _FEATURE))
        sliced = slice_primary_ohlcv(wide, _PRIMARY)

        for col in OHLCV_COLUMNS:
            np.testing.assert_array_equal(
                sliced[col].to_numpy(),
                wide[f"{col}_{_PRIMARY}"].to_numpy(),
            )

    def test_missing_primary_columns_raises(self) -> None:
        wide = _wide_frame((_FEATURE,))

        with pytest.raises(ValueError, match=f"primary_ticker={_PRIMARY!r}"):
            slice_primary_ohlcv(wide, _PRIMARY)


class TestFingerprintMultiBars:
    def test_invariant_to_input_ticker_order(self) -> None:
        wide = _wide_frame((_PRIMARY, _FEATURE, _THIRD))

        h_forward = fingerprint_multi_bars(wide, [_PRIMARY, _FEATURE, _THIRD])
        h_shuffled = fingerprint_multi_bars(wide, [_THIRD, _PRIMARY, _FEATURE])

        assert h_forward == h_shuffled

    def test_detects_value_drift(self) -> None:
        wide_a = _wide_frame((_PRIMARY, _FEATURE), seed=GLOBAL_NUMPY_SEED)
        wide_b = _wide_frame((_PRIMARY, _FEATURE), seed=GLOBAL_NUMPY_SEED + 1)
        tickers = [_PRIMARY, _FEATURE]

        assert fingerprint_multi_bars(wide_a, tickers) != fingerprint_multi_bars(wide_b, tickers)

    def test_missing_ticker_columns_raises(self) -> None:
        wide = _wide_frame((_PRIMARY,))

        with pytest.raises(KeyError, match="fingerprint_multi_bars"):
            fingerprint_multi_bars(wide, [_PRIMARY, _FEATURE])


def _build_cfg(
    *,
    strategy_name: str,
    tickers: Sequence[str],
    strategy_params: dict[str, object] | None = None,
    features: dict[str, object] | None = None,
) -> ExperimentConfig:
    payload: dict[str, object] = {
        "name": "multi_feature_test",
        "data": {
            "source": "yfinance",
            "tickers": list(tickers),
            "start": _START,
            "end": _END,
            "interval": "daily",
        },
        "strategy": {
            "name": strategy_name,
            "params": strategy_params or {},
        },
        "validation": {"n_splits": _N_SPLITS, "test_size": _TEST_SIZE, "gap": _GAP},
        "slippage": {"scenario": "normal"},
    }
    if features is not None:
        payload["features"] = features
    return ExperimentConfig.model_validate(payload)


def _multi_feature_strategy(
    primary: str = _PRIMARY,
    feature_tickers: Sequence[str] = (),
) -> MultiFeatureTestStub:
    return MultiFeatureTestStub(primary_ticker=primary, feature_tickers=feature_tickers)


class TestFetchBarsMultiFeature:
    def test_three_ticker_fetch_produces_suffixed_columns(self) -> None:
        source = _FakeDataSource()
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_PRIMARY, _FEATURE, _THIRD),
            strategy_params={"primary_ticker": _PRIMARY},
        )
        bars = fetch_bars(source, cfg, _multi_feature_strategy())

        for ticker in (_PRIMARY, _FEATURE, _THIRD):
            for col in OHLCV_COLUMNS:
                assert f"{col}_{ticker}" in bars.columns

    def test_single_ticker_multi_feature_path(self) -> None:
        """N=1 multi-feature is degenerate but legal — only the primary is fetched."""

        source = _FakeDataSource()
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_PRIMARY,),
            strategy_params={"primary_ticker": _PRIMARY},
        )
        bars = fetch_bars(source, cfg, _multi_feature_strategy())

        assert f"close_{_PRIMARY}" in bars.columns
        assert source.call_log == [_PRIMARY]

    def test_two_ticker_multi_feature_uses_ticker_suffix_not_pairs_suffix(self) -> None:
        """Critical: 2-ticker multi-feature ≠ pairs at the data layer."""

        source = _FakeDataSource()
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_PRIMARY, _FEATURE),
            strategy_params={"primary_ticker": _PRIMARY},
        )
        bars = fetch_bars(source, cfg, _multi_feature_strategy())

        assert f"close_{_PRIMARY}" in bars.columns
        assert f"close_{_FEATURE}" in bars.columns
        assert "close_a" not in bars.columns
        assert "close_b" not in bars.columns


class TestValidatorMultiFeature:
    def test_both_flags_true_raises_typeerror(self) -> None:
        cfg = _build_cfg(
            strategy_name="_BothFlagsStub",
            tickers=(_PRIMARY, _FEATURE),
            strategy_params={},
        )
        with pytest.raises(TypeError, match="mutually exclusive"):
            _validate_strategy_data_shape(cfg)

    def test_missing_primary_ticker_param_raises(self) -> None:
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_PRIMARY, _FEATURE),
            strategy_params={},
        )
        with pytest.raises(ValueError, match="primary_ticker"):
            _validate_strategy_data_shape(cfg)

    def test_non_string_primary_ticker_raises(self) -> None:
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_PRIMARY, _FEATURE),
            strategy_params={"primary_ticker": 42},
        )
        with pytest.raises(ValueError, match="primary_ticker"):
            _validate_strategy_data_shape(cfg)

    def test_primary_ticker_not_in_tickers_raises(self) -> None:
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_FEATURE, _THIRD),
            strategy_params={"primary_ticker": _PRIMARY},
        )
        with pytest.raises(ValueError, match="primary_ticker"):
            _validate_strategy_data_shape(cfg)

    def test_features_pipeline_with_multi_feature_raises(self) -> None:
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_PRIMARY, _FEATURE),
            strategy_params={"primary_ticker": _PRIMARY},
            features={"name": "standard", "params": {}},
        )
        with pytest.raises(ValueError, match="multi-feature"):
            _validate_strategy_data_shape(cfg)

    def test_valid_multi_feature_config_passes(self) -> None:
        cfg = _build_cfg(
            strategy_name="_MultiFeatureTestStub",
            tickers=(_PRIMARY, _FEATURE, _THIRD),
            strategy_params={"primary_ticker": _PRIMARY},
        )
        _validate_strategy_data_shape(cfg)


def _zero_slippage() -> SlippageConfig:
    return SlippageConfig(
        model=SlippageModel.NoSlippage,
        base_bps=_BPS_ZERO,
        volume_impact_coeff=_VOLUME_IMPACT_ZERO,
    )


class TestWalkForwardDispatch:
    def test_evaluate_walk_forward_routes_through_slice(self) -> None:
        """Wide-format input → engine sees sliced primary OHLCV; strategy sees wide frame.

        ``MultiFeatureTestStub.generate_signals`` indexes ``data[f"close_{primary}"]``;
        if the dispatcher had wrongly handed the strategy a sliced single-asset
        frame, that lookup would KeyError. Successful equity-curve return proves
        both halves of the dispatch — strategy got wide, engine got sliced.
        """

        bars = _wide_frame((_PRIMARY, _FEATURE))
        validator = WalkForwardValidator(n_splits=_N_SPLITS, test_size=_TEST_SIZE, gap=_GAP)
        strategy = _multi_feature_strategy(feature_tickers=(_FEATURE,))

        results = evaluate_walk_forward(
            strategy=strategy,
            bars=bars,
            validator=validator,
            engine=CppBacktestEngine(),
            slippage=_zero_slippage(),
            interval=Interval.DAILY,
        )

        assert len(results) == _N_SPLITS
        for fold in results:
            assert len(fold.backtest.equity_curve) == _TEST_SIZE
            assert np.all(np.isfinite(fold.backtest.equity_curve))
