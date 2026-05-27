"""Tests for CrossAssetMomentumStrategy (multi-feature dispatch + XGBoost gate)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.strategies.cross_asset_momentum import (
    _LOG_RETURN_WARMUP,
    CrossAssetMomentumStrategy,
    _CrossAssetMomentumConfig,
    _derive_feature_columns,
)
from tests.conftest import (
    GLOBAL_NUMPY_SEED,
    assert_params_match_constructor,
    make_synthetic_ohlcv_df,
)

_PRIMARY = "AAA"
_FEATURE_TICKERS: tuple[str, ...] = ("BBB", "CCC")
_LAGS: tuple[int, ...] = (1, 5, 21)
_THRESHOLD = 0.55
_N_BARS = 200

COMPACT_N_ESTIMATORS = 10
COMPACT_MAX_DEPTH = 2

_THRESHOLD_TOO_LOW = 0.49
_THRESHOLD_TOO_HIGH = 1.0


def _wide_frame(
    tickers: Sequence[str], *, n_rows: int = _N_BARS, seed: int = GLOBAL_NUMPY_SEED
) -> pd.DataFrame:
    """Build a wide ``<col>_<TICKER>`` frame with ``n_rows`` rows per ticker."""

    suffixed = [
        make_synthetic_ohlcv_df(n_rows=n_rows, seed=seed + offset).add_suffix(f"_{ticker}")
        for offset, ticker in enumerate(tickers)
    ]
    joined = suffixed[0]
    for other in suffixed[1:]:
        joined = joined.join(other, how="inner")
    return joined


@pytest.fixture(scope="module")
def wide_df() -> pd.DataFrame:
    return _wide_frame((_PRIMARY, *_FEATURE_TICKERS))


def _make_strategy(**overrides: object) -> CrossAssetMomentumStrategy:
    """Construct a strategy with compact XGBoost params and the test defaults."""

    kwargs: dict[str, object] = {
        "primary_ticker": _PRIMARY,
        "feature_tickers": _FEATURE_TICKERS,
        "lags": _LAGS,
        "direction_threshold": _THRESHOLD,
        "n_estimators": COMPACT_N_ESTIMATORS,
        "max_depth": COMPACT_MAX_DEPTH,
    }
    kwargs.update(overrides)
    return CrossAssetMomentumStrategy(**kwargs)  # type: ignore[arg-type]


class _FakeClassifier:
    """Duck-typed classifier returning a constant probability — bypasses XGBoost fit."""

    def __init__(self, proba: float) -> None:
        self._proba = proba
        self.training_metadata = None

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._proba, index=df.index, name="up_prob")


def _strategy_with_fake_classifier(
    proba: float, wide_df: pd.DataFrame
) -> CrossAssetMomentumStrategy:
    """Build a fitted strategy whose classifier is a constant-proba fake.

    Skips the real XGBoost fit by setting the classifier and fitted-state
    directly — the threshold tests only exercise ``generate_signals`` and
    don't care about booster weights.
    """

    s = _make_strategy()
    s._classifier = _FakeClassifier(proba)  # type: ignore[assignment]
    s._set_fitted_with_metadata(
        TrainingMetadata.from_fit(wide_df, Interval.DAILY, tuple(s._feature_columns))
    )
    return s


class TestCtorValidation:
    def test_empty_primary_ticker_raises(self) -> None:
        with pytest.raises(ValueError, match="primary_ticker"):
            CrossAssetMomentumStrategy(primary_ticker="", feature_tickers=_FEATURE_TICKERS)

    def test_empty_feature_tickers_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_tickers"):
            CrossAssetMomentumStrategy(primary_ticker=_PRIMARY, feature_tickers=())

    def test_duplicate_feature_tickers_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            CrossAssetMomentumStrategy(primary_ticker=_PRIMARY, feature_tickers=("BBB", "BBB"))

    def test_empty_lags_raises(self) -> None:
        with pytest.raises(ValueError, match="lags"):
            _make_strategy(lags=())

    def test_negative_lag_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            _make_strategy(lags=(0, 5))

    def test_duplicate_lag_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            _make_strategy(lags=(1, 1, 5))

    def test_threshold_below_lower_raises(self) -> None:
        with pytest.raises(ValueError, match="direction_threshold"):
            _make_strategy(direction_threshold=_THRESHOLD_TOO_LOW)

    def test_threshold_at_or_above_upper_raises(self) -> None:
        with pytest.raises(ValueError, match="direction_threshold"):
            _make_strategy(direction_threshold=_THRESHOLD_TOO_HIGH)


class TestFeatureDerivation:
    def test_column_order_outer_ticker_inner_lag(self) -> None:
        cols = _derive_feature_columns(("X", "Y"), (1, 3))
        assert cols == ["lag1_X", "lag3_X", "lag1_Y", "lag3_Y"]

    def test_required_warmup_includes_log_return_lead(self) -> None:
        lags = (1, 7)
        s = _make_strategy(lags=lags)
        assert s.required_warmup_bars == max(lags) + _LOG_RETURN_WARMUP


class TestTrainGenerate:
    def test_generate_signals_before_train_raises(self, wide_df: pd.DataFrame) -> None:
        s = _make_strategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.generate_signals(wide_df)

    def test_train_populates_metadata(self, wide_df: pd.DataFrame) -> None:
        s = _make_strategy()
        s.train(wide_df)
        meta = s.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(wide_df)
        assert meta.interval == Interval.DAILY
        expected = tuple(_derive_feature_columns(_FEATURE_TICKERS, _LAGS))
        assert meta.feature_columns == expected

    def test_signals_three_way_set(self, wide_df: pd.DataFrame) -> None:
        s = _make_strategy()
        s.train(wide_df)
        signals = s.generate_signals(wide_df)
        non_nan = signals.dropna()
        assert set(non_nan.unique()).issubset({-1.0, 0.0, 1.0})

    def test_high_p_up_yields_long(self, wide_df: pd.DataFrame) -> None:
        s = _strategy_with_fake_classifier(proba=0.9, wide_df=wide_df)
        signals = s.generate_signals(wide_df).dropna()
        assert (signals == 1.0).all()

    def test_low_p_up_yields_short(self, wide_df: pd.DataFrame) -> None:
        s = _strategy_with_fake_classifier(proba=0.1, wide_df=wide_df)
        signals = s.generate_signals(wide_df).dropna()
        assert (signals == -1.0).all()

    def test_mid_p_up_yields_flat(self, wide_df: pd.DataFrame) -> None:
        """``1 - threshold < proba < threshold`` lands in the dead zone (signal=0)."""

        s = _strategy_with_fake_classifier(proba=0.5, wide_df=wide_df)
        signals = s.generate_signals(wide_df).dropna()
        assert (signals == 0.0).all()

    def test_warmup_bars_are_nan(self, wide_df: pd.DataFrame) -> None:
        s = _strategy_with_fake_classifier(proba=0.55, wide_df=wide_df)
        signals = s.generate_signals(wide_df)
        warmup = max(_LAGS) + _LOG_RETURN_WARMUP
        assert signals.iloc[:warmup].isna().all()
        assert signals.iloc[warmup:].notna().any()

    def test_deterministic_signals(self, wide_df: pd.DataFrame) -> None:
        s = _strategy_with_fake_classifier(proba=0.55, wide_df=wide_df)
        s1 = s.generate_signals(wide_df)
        s2 = s.generate_signals(wide_df)
        np.testing.assert_array_equal(s1.to_numpy(), s2.to_numpy())


class TestRegistration:
    def test_registry_registration(self) -> None:
        assert "CrossAssetMomentum" in strategy_registry

    def test_capability_flag(self) -> None:
        assert CrossAssetMomentumStrategy.is_multi_feature_strategy is True
        assert CrossAssetMomentumStrategy.is_pairs_strategy is False

    def test_primary_ticker_property(self) -> None:
        s = _make_strategy()
        assert s.primary_ticker == _PRIMARY


class TestSuggestParams:
    def test_keys(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = CrossAssetMomentumStrategy.suggest_params(trial)
        assert set(params.keys()) == {
            "direction_threshold",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "subsample",
            "colsample_bytree",
        }

    def test_lags_and_feature_tickers_not_tuned(self) -> None:
        """Lag schedule + ticker basket stay fixed at YAML level — see docstring."""

        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = CrossAssetMomentumStrategy.suggest_params(trial)
        assert "lags" not in params
        assert "feature_tickers" not in params


class TestParamsDataclassDriftGuard:
    def test_fields_match_constructor(self) -> None:
        assert_params_match_constructor(
            _CrossAssetMomentumConfig,
            CrossAssetMomentumStrategy,
        )


class TestSaveLoad:
    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        s = _make_strategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.save(tmp_path / "cam")

    def test_round_trip_matches_original(self, wide_df: pd.DataFrame, tmp_path: Path) -> None:
        original = _make_strategy()
        original.train(wide_df)

        path = tmp_path / "cam"
        original.save(path)
        loaded = CrossAssetMomentumStrategy.load(path)

        assert loaded.training_metadata is not None
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.generate_signals(wide_df).to_numpy(),
            original.generate_signals(wide_df).to_numpy(),
        )
