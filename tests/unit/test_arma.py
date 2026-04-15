"""Tests for ARMAPredictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.arma import ARMAPredictor
from tests.conftest import make_synthetic_close_df

# Compact ARMA search space (small for fast CI)
COMPACT_P_MAX = 2
COMPACT_Q_MAX = 2

# Minimal close samples for "before fit" guards
SAMPLE_CLOSE_TWO = [100, 101]
SAMPLE_CLOSE_THREE = [100, 101, 102]

# Hourly test data
HOURLY_ROW_COUNT = 200
HOURLY_START = "2020-01-02 09:30"
HOURLY_RETURN_STD = 0.005
HOURLY_BASE_PRICE = 100.0

# Determinism seed
NUMPY_SEED = 42


@pytest.fixture
def arma_df() -> pd.DataFrame:
    return make_synthetic_close_df()


@pytest.fixture
def arma_target(arma_df: pd.DataFrame) -> pd.Series:
    """Log returns target for ARMA."""
    return compute_log_returns(arma_df["close"]).dropna()


@pytest.fixture
def fitted_arma(arma_df: pd.DataFrame, arma_target: pd.Series) -> ARMAPredictor:
    """ARMAPredictor already fitted."""
    a = ARMAPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
    a.fit(arma_df.iloc[1:], arma_target)
    return a


class TestARMAPredictor:
    def test_predict_before_fit_raises(self, arma_df: pd.DataFrame) -> None:
        a = ARMAPredictor()
        with pytest.raises(RuntimeError, match="before fit"):
            a.predict(arma_df)

    def test_predict_single_before_fit_raises(self) -> None:
        a = ARMAPredictor()
        with pytest.raises(RuntimeError, match="before fit"):
            a.predict_single(pd.DataFrame({"close": SAMPLE_CLOSE_TWO}))

    def test_fit_sets_fitted(self, fitted_arma: ARMAPredictor) -> None:
        assert fitted_arma._fitted

    def test_fitted_order_valid(self, fitted_arma: ARMAPredictor) -> None:
        p, d, q = fitted_arma._best_order
        assert p >= 0
        assert d == 0  # returns are stationary
        assert q >= 0

    def test_tune_returns_valid_order(self, arma_target: pd.Series) -> None:
        a = ARMAPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        p, q = a.tune(arma_target)
        assert 0 <= p <= COMPACT_P_MAX
        assert 0 <= q <= COMPACT_Q_MAX

    def test_predict_output_shape(self, fitted_arma: ARMAPredictor, arma_df: pd.DataFrame) -> None:
        result = fitted_arma.predict(arma_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(arma_df)

    def test_predict_single_returns_float(self, fitted_arma: ARMAPredictor) -> None:
        val = fitted_arma.predict_single(pd.DataFrame({"close": SAMPLE_CLOSE_THREE}))
        assert isinstance(val, float)

    def test_training_metadata_populated(self, fitted_arma: ARMAPredictor) -> None:
        meta = fitted_arma.training_metadata
        assert meta is not None
        assert meta.n_train_samples > 0
        assert meta.interval == Interval.DAILY
        assert meta.feature_columns == ("returns",)

    def test_training_metadata_validates_overlap(
        self, fitted_arma: ARMAPredictor, arma_df: pd.DataFrame
    ) -> None:
        meta = fitted_arma.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(arma_df)

    def test_registry_registration(self) -> None:
        from src.core.registry import model_registry

        assert "arma" in model_registry

    def test_hourly_interval(self) -> None:
        """Verify ARMA works with non-daily data."""
        np.random.seed(NUMPY_SEED)
        idx = pd.date_range(start=HOURLY_START, periods=HOURLY_ROW_COUNT, freq="h")
        returns = np.random.normal(0, HOURLY_RETURN_STD, HOURLY_ROW_COUNT)
        close = HOURLY_BASE_PRICE * np.cumprod(1 + returns)
        df = pd.DataFrame({"close": close}, index=idx)

        target = compute_log_returns(df["close"]).dropna()

        a = ARMAPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX, interval=Interval.HOUR)
        a.fit(df.iloc[1:], target)
        assert a.training_metadata is not None
        assert a.training_metadata.interval == Interval.HOUR

    def test_suggest_params(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = ARMAPredictor.suggest_params(trial)
        assert "p_max" in params
        assert "q_max" in params
        assert "information_criterion" in params
