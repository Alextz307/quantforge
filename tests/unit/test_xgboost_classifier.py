"""Tests for DirectionalClassifier (XGBoost)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import LeakageError
from src.core.types import Interval
from src.models.xgboost_classifier import DirectionalClassifier


@pytest.fixture
def xgb_data() -> tuple[pd.DataFrame, pd.Series]:
    """Feature DataFrame and binary target for XGBoost testing."""
    np.random.seed(42)
    n = 200
    idx = pd.bdate_range(start="2020-01-02", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))

    features = pd.DataFrame(
        {
            "return_1d": np.random.normal(0, 0.01, n),
            "return_5d": np.random.normal(0, 0.02, n),
            "vol_20": np.abs(np.random.normal(0.15, 0.03, n)),
            "rsi_14": np.random.uniform(20, 80, n),
        },
        index=idx,
    )

    # Target: next-day direction (exclude last row — no future)
    direction = (np.diff(close) > 0).astype(int)
    target = pd.Series(direction, index=idx[:-1], name="direction")

    return features.iloc[:-1], target


class TestDirectionalClassifier:
    def test_predict_before_fit_raises(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        c = DirectionalClassifier()
        with pytest.raises(RuntimeError, match="before fit"):
            c.predict(xgb_data[0])

    def test_predict_proba_before_fit_raises(
        self, xgb_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        c = DirectionalClassifier()
        with pytest.raises(RuntimeError, match="before fit"):
            c.predict_proba(xgb_data[0])

    def test_fit_sets_fitted(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(n_estimators=10)
        c.fit(features, target)
        assert c._fitted

    def test_predict_proba_in_range(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(n_estimators=10)
        c.fit(features, target)

        proba = c.predict_proba(features)
        assert isinstance(proba, pd.Series)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_binary(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(n_estimators=10)
        c.fit(features, target)

        preds = c.predict(features)
        assert isinstance(preds, pd.Series)
        assert set(preds.unique()).issubset({0, 1})

    def test_output_length(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(n_estimators=10)
        c.fit(features, target)

        proba = c.predict_proba(features)
        preds = c.predict(features)
        assert len(proba) == len(features)
        assert len(preds) == len(features)

    def test_early_stopping_records(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(n_estimators=100, early_stopping_rounds=5)
        c.fit(features, target)
        # eval_results should be populated from the validation split
        assert len(c._eval_results) > 0

    def test_training_metadata_populated(self, xgb_data: tuple[pd.DataFrame, pd.Series]) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(n_estimators=10)
        c.fit(features, target)

        meta = c.training_metadata
        assert meta is not None
        assert meta.n_train_samples == len(features)
        assert meta.interval == Interval.DAILY
        assert set(meta.feature_columns) == set(features.columns)

    def test_training_metadata_validates_overlap(
        self, xgb_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, target = xgb_data
        c = DirectionalClassifier(n_estimators=10)
        c.fit(features, target)

        meta = c.training_metadata
        assert meta is not None
        with pytest.raises(LeakageError):
            meta.validate_no_overlap(features)

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
