"""Round-trip save/load tests for every strategy.

Each test trains a strategy, writes it to a fresh ``tmp_path``, reloads, and
asserts ``generate_signals()`` output is bit-identical (or within a tight
statsmodels-filter tolerance for ARMA-backed strategies). ``training_metadata``
is also compared field-by-field.

The strategies exercise every persistence shape:
 - Flat (no sub-model): PairsTrading
 - Single sub-model:    AdaptiveBollinger (GARCH), MomentumGatekeeper (classifier)
 - Nested composite:    ReturnForecast (HybridReturn), VolatilityTargeting (HybridVolatility)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.types import Interval
from src.models.hybrid_return import HybridReturnModel
from src.models.hybrid_volatility import HybridVolatilityModel
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from src.strategies.cross_asset_momentum import CrossAssetMomentumStrategy
from src.strategies.momentum_gatekeeper import MomentumGatekeeperStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.return_forecast import ReturnForecastStrategy
from src.strategies.volatility_targeting import VolatilityTargetingStrategy
from tests.conftest import (
    attach_synthetic_features,
    make_pair_close_df,
    make_synthetic_close_df,
    make_synthetic_ohlcv_df,
)

# Compact params just need a fitted model to exercise the round-trip.
COMPACT_GARCH_P_MAX = 2
COMPACT_GARCH_Q_MAX = 2
COMPACT_ARMA_P_MAX = 2
COMPACT_ARMA_Q_MAX = 2
COMPACT_LSTM_HIDDEN_DIM = 8
COMPACT_LSTM_LOOKBACK = 5
COMPACT_LSTM_EPOCHS = 1
COMPACT_XGB_N_ESTIMATORS = 5
COMPACT_XGB_MAX_DEPTH = 2

PAIRS_ENTRY_Z = 2.0
PAIRS_EXIT_Z = 0.5
PAIRS_STOP_Z = 4.0
PAIRS_LOOKBACK = 20

FIT_TORCH_SEED = 0
FIT_NUMPY_SEED = 0


@pytest.fixture
def close_df() -> pd.DataFrame:
    return make_synthetic_close_df()


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return make_synthetic_ohlcv_df()


@pytest.fixture
def pair_df() -> pd.DataFrame:
    return make_pair_close_df()


@pytest.fixture
def feature_columns() -> list[str]:
    return ["feat_a", "feat_b"]


class TestPairsTradingSaveLoad:
    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        s = PairsTradingStrategy(
            entry_zscore=PAIRS_ENTRY_Z,
            exit_zscore=PAIRS_EXIT_Z,
            stop_loss_zscore=PAIRS_STOP_Z,
            zscore_lookback=PAIRS_LOOKBACK,
        )
        with pytest.raises(RuntimeError, match="before train"):
            s.save(tmp_path / "pairs")

    def test_round_trip_matches_original(
        self,
        pair_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        original = PairsTradingStrategy(
            entry_zscore=PAIRS_ENTRY_Z,
            exit_zscore=PAIRS_EXIT_Z,
            stop_loss_zscore=PAIRS_STOP_Z,
            zscore_lookback=PAIRS_LOOKBACK,
        )
        original.train(pair_df)

        path = tmp_path / "pairs"
        original.save(path)
        loaded = PairsTradingStrategy.load(path)

        assert loaded.training_metadata is not None
        assert loaded._is_cointegrated is True
        assert loaded._hedge_ratio == original._hedge_ratio
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.generate_signals(pair_df).to_numpy(),
            original.generate_signals(pair_df).to_numpy(),
        )


class TestAdaptiveBollingerSaveLoad:
    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        s = AdaptiveBollingerStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.save(tmp_path / "ab")

    def test_round_trip_matches_original(
        self,
        close_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        original = AdaptiveBollingerStrategy(
            garch_p_max=COMPACT_GARCH_P_MAX,
            garch_q_max=COMPACT_GARCH_Q_MAX,
        )
        original.train(close_df)

        path = tmp_path / "ab"
        original.save(path)
        loaded = AdaptiveBollingerStrategy.load(path)

        assert loaded.training_metadata is not None
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.generate_signals(close_df).to_numpy(),
            original.generate_signals(close_df).to_numpy(),
        )


class TestMomentumGatekeeperSaveLoad:
    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        s = MomentumGatekeeperStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.save(tmp_path / "momentum")

    def test_round_trip_matches_original(
        self,
        close_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        original = MomentumGatekeeperStrategy(
            n_estimators=COMPACT_XGB_N_ESTIMATORS,
            max_depth=COMPACT_XGB_MAX_DEPTH,
        )
        original.train(close_df)

        path = tmp_path / "momentum"
        original.save(path)
        loaded = MomentumGatekeeperStrategy.load(path)

        assert loaded.training_metadata is not None
        assert loaded._resolved_feature_columns == original._resolved_feature_columns
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.generate_signals(close_df).to_numpy(),
            original.generate_signals(close_df).to_numpy(),
        )


class TestReturnForecastSaveLoad:
    def test_save_before_train_raises(
        self,
        feature_columns: list[str],
        tmp_path: Path,
    ) -> None:
        s = ReturnForecastStrategy(feature_columns=feature_columns)
        with pytest.raises(RuntimeError, match="before train"):
            s.save(tmp_path / "retf")

    def test_round_trip_matches_original(
        self,
        close_df: pd.DataFrame,
        feature_columns: list[str],
        tmp_path: Path,
    ) -> None:
        df = attach_synthetic_features(close_df, feature_columns)

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        original = ReturnForecastStrategy(
            feature_columns=feature_columns,
            arma_p_max=COMPACT_ARMA_P_MAX,
            arma_q_max=COMPACT_ARMA_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
        )
        original.train(df)

        path = tmp_path / "retf"
        original.save(path)
        loaded = ReturnForecastStrategy.load(path)

        assert loaded.training_metadata is not None
        assert loaded.training_metadata == original.training_metadata
        # statsmodels ``filter`` reproduces ARMA fitted values to FP noise,
        # not bit-identical, so use allclose for ARMA-backed strategies.
        np.testing.assert_allclose(
            loaded.generate_signals(df).to_numpy(),
            original.generate_signals(df).to_numpy(),
            rtol=0.0,
            atol=1e-10,
        )


class TestVolatilityTargetingSaveLoad:
    def test_save_before_train_raises(
        self,
        feature_columns: list[str],
        tmp_path: Path,
    ) -> None:
        s = VolatilityTargetingStrategy(feature_columns=feature_columns)
        with pytest.raises(RuntimeError, match="before train"):
            s.save(tmp_path / "volt")

    def test_round_trip_matches_original(
        self,
        ohlcv_df: pd.DataFrame,
        feature_columns: list[str],
        tmp_path: Path,
    ) -> None:
        df = attach_synthetic_features(ohlcv_df, feature_columns)

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        original = VolatilityTargetingStrategy(
            feature_columns=feature_columns,
            garch_p_max=COMPACT_GARCH_P_MAX,
            garch_q_max=COMPACT_GARCH_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
            interval=Interval.DAILY,
        )
        original.train(df)

        path = tmp_path / "volt"
        original.save(path)
        loaded = VolatilityTargetingStrategy.load(path)

        assert loaded.training_metadata is not None
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.generate_signals(df).to_numpy(),
            original.generate_signals(df).to_numpy(),
        )


# Drift guard: ``_ctor_kwargs_as_json()`` keys must match ``__init__`` (minus
# device prefs, which are re-resolved on load, never persisted). Fails before
# the first round-trip does when a new ctor kwarg lands without a persisted key.
_DRIFT_CASES: list[tuple[type, Callable[[], object], set[str]]] = [
    (
        HybridVolatilityModel,
        lambda: HybridVolatilityModel(feature_columns=["x"]),
        {"lstm_device"},
    ),
    (
        HybridReturnModel,
        lambda: HybridReturnModel(feature_columns=["x"]),
        {"lstm_device"},
    ),
    (PairsTradingStrategy, lambda: PairsTradingStrategy(), set()),
    (
        AdaptiveBollingerStrategy,
        lambda: AdaptiveBollingerStrategy(),
        set(),
    ),
    (
        MomentumGatekeeperStrategy,
        lambda: MomentumGatekeeperStrategy(),
        {"device"},
    ),
    (
        CrossAssetMomentumStrategy,
        lambda: CrossAssetMomentumStrategy(primary_ticker="X", feature_tickers=("Y",)),
        {"device"},
    ),
    (
        ReturnForecastStrategy,
        lambda: ReturnForecastStrategy(feature_columns=["x"]),
        {"lstm_device"},
    ),
    (
        VolatilityTargetingStrategy,
        lambda: VolatilityTargetingStrategy(feature_columns=["x"]),
        {"lstm_device"},
    ),
]


@pytest.mark.parametrize(
    "cls,factory,excluded",
    _DRIFT_CASES,
    ids=[case[0].__name__ for case in _DRIFT_CASES],
)
def test_save_config_keys_match_ctor_signature(
    cls: type,
    factory: Callable[[], object],
    excluded: set[str],
) -> None:
    """Fails loudly when a new ctor kwarg lands without a corresponding
    persisted-config key (or vice versa). Device preferences are the only
    intentional exclusion — they're re-resolved on load, not persisted."""

    instance = factory()
    config_keys = set(instance._ctor_kwargs_as_json())  # type: ignore[attr-defined]
    ctor_keys = set(inspect.signature(cls).parameters)
    expected = ctor_keys - excluded
    assert config_keys == expected, (
        f"{cls.__name__}: ctor_kwargs_as_json drifted from __init__. "
        f"symmetric diff = {config_keys ^ expected}"
    )
