"""Tests for DirectionalClassifier (XGBoost)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.types import Interval
from src.models.xgboost_classifier import DirectionalClassifier

# Synthetic fixture data
SYNTH_ROW_COUNT = 200
SYNTH_START_DATE = "2020-01-02"
SYNTH_BASE_PRICE = 100.0
SYNTH_RETURN_MEAN = 0.0003
SYNTH_RETURN_STD = 0.01
RETURN_1D_STD = 0.01
RETURN_5D_STD = 0.02
VOL_MEAN = 0.15
VOL_STD = 0.03
RSI_LOW = 20.0
RSI_HIGH = 80.0
SYNTH_FIXTURE_SEED = 42

# Compact XGBoost parameters (small for fast CI)
COMPACT_N_ESTIMATORS = 10
EARLY_STOP_N_ESTIMATORS = 100
EARLY_STOP_ROUNDS = 5


@pytest.fixture
def xgb_data() -> tuple[pd.DataFrame, pd.Series]:
    """Feature DataFrame and binary target for XGBoost testing."""
    np.random.seed(SYNTH_FIXTURE_SEED)
    idx = pd.bdate_range(start=SYNTH_START_DATE, periods=SYNTH_ROW_COUNT, freq="B")
    close = SYNTH_BASE_PRICE * np.cumprod(
        1 + np.random.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, SYNTH_ROW_COUNT)
    )

    features = pd.DataFrame(
        {
            "return_1d": np.random.normal(0, RETURN_1D_STD, SYNTH_ROW_COUNT),
            "return_5d": np.random.normal(0, RETURN_5D_STD, SYNTH_ROW_COUNT),
            "vol_20": np.abs(np.random.normal(VOL_MEAN, VOL_STD, SYNTH_ROW_COUNT)),
            "rsi_14": np.random.uniform(RSI_LOW, RSI_HIGH, SYNTH_ROW_COUNT),
        },
        index=idx,
    )

    # Target: next-day direction (exclude last row — no future)
    direction = (np.diff(close) > 0).astype(int)
    target = pd.Series(direction, index=idx[:-1], name="direction")

    return features.iloc[:-1], target


@pytest.fixture
def xgb_features() -> list[str]:
    """Feature column list matching xgb_data."""
    return ["return_1d", "return_5d", "vol_20", "rsi_14"]


class TestDirectionalClassifier:
    def test_predict_before_fit_raises(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        c = DirectionalClassifier(xgb_features)
        with pytest.raises(RuntimeError, match="before fit"):
            c.predict(xgb_data[0])

    def test_predict_proba_before_fit_raises(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        c = DirectionalClassifier(xgb_features)
        with pytest.raises(RuntimeError, match="before fit"):
            c.predict_proba(xgb_data[0])

    def test_fit_sets_fitted(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
        c.fit(features, target)
        assert c._fitted

    def test_predict_proba_in_range(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
        c.fit(features, target)

        proba = c.predict_proba(features)
        assert isinstance(proba, pd.Series)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_binary(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
        c.fit(features, target)

        preds = c.predict(features)
        assert isinstance(preds, pd.Series)
        assert set(preds.unique()).issubset({0, 1})

    def test_output_length(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
        c.fit(features, target)

        proba = c.predict_proba(features)
        preds = c.predict(features)
        assert len(proba) == len(features)
        assert len(preds) == len(features)

    def test_early_stopping_records(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(
            xgb_features,
            n_estimators=EARLY_STOP_N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
        )
        c.fit(features, target)
        # eval_results should be populated from the validation split
        assert len(c._eval_results) > 0

    def test_training_metadata_populated(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
        c.fit(features, target)

        meta = c.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(features)
        assert meta.interval == Interval.DAILY
        assert set(meta.feature_columns) == set(xgb_features)

    def test_training_metadata_validates_overlap(
        self, xgb_data: tuple[pd.DataFrame, pd.Series], xgb_features: list[str]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
        c.fit(features, target)

        meta = c.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(features)

    def test_empty_feature_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_columns"):
            DirectionalClassifier([])

    def test_feature_columns_honored(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Explicit feature_columns restricts which columns XGBoost trains on."""
        features, target = xgb_data
        subset = ["return_1d", "vol_20"]
        c = DirectionalClassifier(subset, n_estimators=COMPACT_N_ESTIMATORS)
        c.fit(features, target)
        assert c._feature_columns == subset
        # rsi_14 and return_5d are ignored
        proba = c.predict_proba(features)
        assert isinstance(proba, pd.Series)

    def test_registry_registration(self) -> None:
        from src.core.registry import classifier_registry

        assert "xgboost_directional" in classifier_registry

    def test_suggest_params(self) -> None:
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = DirectionalClassifier.suggest_params(trial)
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "max_depth" in params
