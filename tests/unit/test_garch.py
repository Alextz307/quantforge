"""Tests for GARCHPredictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.types import Interval
from src.core.utils import compute_log_returns
from src.models.garch import GARCHPredictor
from tests.conftest import make_synthetic_close_df

COMPACT_P_MAX = 2
COMPACT_Q_MAX = 2

SAMPLE_RETURNS = [0.01, -0.02, 0.015]

OOS_ROW_COUNT = 50
OOS_START_DATE = "2021-01-02"
OOS_FIXTURE_SEED = 99

NUMPY_SEED = 42


@pytest.fixture
def garch_df() -> pd.DataFrame:
    return make_synthetic_close_df()


@pytest.fixture
def fitted_garch(garch_df: pd.DataFrame) -> GARCHPredictor:
    """GARCHPredictor already fitted on garch_df returns."""

    g = GARCHPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
    target = compute_log_returns(garch_df["close"]).dropna()
    g.fit(garch_df.iloc[1:], target)
    return g


class TestGARCHPredictor:
    def test_predict_before_fit_raises(self, garch_df: pd.DataFrame) -> None:
        g = GARCHPredictor()
        with pytest.raises(RuntimeError, match="before fit"):
            g.predict(garch_df)

    def test_generate_vol_before_fit_raises(self) -> None:
        g = GARCHPredictor()
        s = pd.Series(SAMPLE_RETURNS)
        with pytest.raises(RuntimeError, match="before fit"):
            g.generate_vol_series(s)

    def test_fit_sets_fitted(self, fitted_garch: GARCHPredictor) -> None:
        assert fitted_garch.training_metadata is not None

    def test_tune_returns_valid_order(self, garch_df: pd.DataFrame) -> None:
        g = GARCHPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        returns = compute_log_returns(garch_df["close"]).dropna()
        p, q = g.tune(returns)
        assert 1 <= p <= COMPACT_P_MAX
        assert 1 <= q <= COMPACT_Q_MAX

    def test_predict_output_shape(
        self, fitted_garch: GARCHPredictor, garch_df: pd.DataFrame
    ) -> None:
        result = fitted_garch.predict(garch_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(garch_df)

    def test_predict_values_positive(
        self, fitted_garch: GARCHPredictor, garch_df: pd.DataFrame
    ) -> None:
        result = fitted_garch.predict(garch_df)
        assert (result.dropna() > 0).all()

    def test_predict_single(self, fitted_garch: GARCHPredictor, garch_df: pd.DataFrame) -> None:
        val = fitted_garch.predict_single(garch_df)
        assert isinstance(val, float)
        assert val > 0

    def test_params_frozen_after_fit(self, fitted_garch: GARCHPredictor) -> None:
        omega = fitted_garch._omega
        alpha = fitted_garch._alpha.copy()
        beta = fitted_garch._beta.copy()

        new_df = make_synthetic_close_df(
            n_rows=OOS_ROW_COUNT, start=OOS_START_DATE, seed=OOS_FIXTURE_SEED
        )
        fitted_garch.predict(new_df)

        assert fitted_garch._omega == omega
        np.testing.assert_array_equal(fitted_garch._alpha, alpha)
        np.testing.assert_array_equal(fitted_garch._beta, beta)

    def test_generate_vol_series(
        self, fitted_garch: GARCHPredictor, garch_df: pd.DataFrame
    ) -> None:
        returns = compute_log_returns(garch_df["close"]).dropna()
        vol = fitted_garch.generate_vol_series(returns)
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(returns)
        assert (vol > 0).all()

    def test_training_metadata_populated(self, fitted_garch: GARCHPredictor) -> None:
        meta = fitted_garch.training_metadata
        assert meta is not None
        assert meta.n_train_samples > 0
        assert meta.interval == Interval.DAILY
        assert meta.feature_columns == ("returns",)

    def test_training_metadata_validates_overlap(
        self, fitted_garch: GARCHPredictor, garch_df: pd.DataFrame
    ) -> None:
        meta = fitted_garch.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(garch_df)

    def test_registry_registration(self) -> None:
        from src.core.registry import model_registry

        assert "garch" in model_registry

    def test_deterministic_with_seed(self, garch_df: pd.DataFrame) -> None:
        np.random.seed(NUMPY_SEED)
        g1 = GARCHPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        target = compute_log_returns(garch_df["close"]).dropna()
        g1.fit(garch_df.iloc[1:], target)
        r1 = g1.predict(garch_df)

        np.random.seed(NUMPY_SEED)
        g2 = GARCHPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        g2.fit(garch_df.iloc[1:], target)
        r2 = g2.predict(garch_df)

        pd.testing.assert_series_equal(r1, r2)

    def test_suggest_params(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = GARCHPredictor.suggest_params(trial)
        assert "p_max" in params
        assert "q_max" in params

    def test_predict_returns_kwarg_matches_default_path(
        self, fitted_garch: GARCHPredictor, garch_df: pd.DataFrame
    ) -> None:
        # Bit-identical to the implicit-returns path; catches alignment-
        # logic regressions (positional slice-assign vs reindex).
        default_vol = fitted_garch.predict(garch_df)
        returns = compute_log_returns(garch_df["close"]).dropna()
        explicit_vol = fitted_garch.predict(garch_df, returns=returns)
        pd.testing.assert_series_equal(default_vol, explicit_vol)
