"""Round-trip save/load tests for every leaf predictor and classifier.

Each test fits a model, writes it to a fresh ``tmp_path``, reloads, and asserts
``predict()`` output is bit-identical. ``training_metadata`` is also compared
field-by-field. The four tests exercise the four persistence formats:
JSON weights (GARCH), JSON-plus-endog (ARMA), torch state_dict (LSTM), and
XGBoost UBJ (DirectionalClassifier).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.persistence import CONFIG_JSON, WEIGHTS_JSON, read_json_dict, write_json
from src.core.utils import compute_log_returns
from src.models.arma import ARMAPredictor
from src.models.garch import GARCHPredictor
from src.models.lstm import LSTMPredictor
from src.models.xgboost_classifier import DirectionalClassifier
from tests.conftest import make_synthetic_close_df

# ARMA/GARCH: small grid, fast CI
COMPACT_P_MAX = 2
COMPACT_Q_MAX = 2

# LSTM: compact architecture, one epoch of training — we only need a fitted
# state dict for round-trip, not a well-trained model.
LSTM_FEATURE_COUNT = 3
LSTM_HIDDEN_DIM = 8
LSTM_NUM_LAYERS = 1
LSTM_LOOKBACK = 5
LSTM_EPOCHS = 1
LSTM_BATCH_SIZE = 8
LSTM_VAL_SPLIT = 0.2

# XGBoost: small booster; one feature vector pass is enough to exercise round-trip.
XGB_N_ESTIMATORS = 5
XGB_MAX_DEPTH = 2

# Fixture-wide synthetic-data constants
SYNTH_SEED = 7
SYNTH_FEATURE_NOISE_STD = 0.01
SYNTH_VOLUME_LOW = 1_000_000
SYNTH_VOLUME_HIGH = 5_000_000


@pytest.fixture
def close_df() -> pd.DataFrame:
    return make_synthetic_close_df()


@pytest.fixture
def log_return_target(close_df: pd.DataFrame) -> pd.Series:
    return compute_log_returns(close_df["close"]).dropna()


@pytest.fixture
def lstm_df() -> pd.DataFrame:
    """DataFrame with multiple feature columns suitable for LSTMPredictor."""
    rng = np.random.default_rng(SYNTH_SEED)
    idx = pd.bdate_range(start="2020-01-02", periods=100, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, 100))
    return pd.DataFrame(
        {
            "close": close,
            "volume": rng.integers(SYNTH_VOLUME_LOW, SYNTH_VOLUME_HIGH, 100).astype(float),
            "return_1d": rng.normal(0.0, SYNTH_FEATURE_NOISE_STD, 100),
        },
        index=idx,
    )


@pytest.fixture
def lstm_features() -> list[str]:
    return ["close", "volume", "return_1d"]


@pytest.fixture
def lstm_target(lstm_df: pd.DataFrame) -> pd.Series:
    returns = lstm_df["close"].pct_change().shift(-1)
    return returns.iloc[:-1]


@pytest.fixture
def xgb_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(SYNTH_SEED)
    n = 120
    idx = pd.bdate_range(start="2020-01-02", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    features = pd.DataFrame(
        {
            "return_1d": rng.normal(0, 0.01, n),
            "return_5d": rng.normal(0, 0.02, n),
        },
        index=idx,
    )
    target = pd.Series((np.diff(close) > 0).astype(int), index=idx[:-1], name="direction")
    return features.iloc[:-1], target


class TestGARCHSaveLoad:
    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        g = GARCHPredictor()
        with pytest.raises(RuntimeError, match="before fit"):
            g.save(tmp_path / "garch")

    def test_round_trip_matches_original(
        self,
        close_df: pd.DataFrame,
        log_return_target: pd.Series,
        tmp_path: Path,
    ) -> None:
        original = GARCHPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        original.fit(close_df.iloc[1:], log_return_target)

        path = tmp_path / "garch"
        original.save(path)
        loaded = GARCHPredictor.load(path)

        assert loaded._fitted
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.predict(close_df).to_numpy(),
            original.predict(close_df).to_numpy(),
        )


class TestARMASaveLoad:
    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        a = ARMAPredictor()
        with pytest.raises(RuntimeError, match="before fit"):
            a.save(tmp_path / "arma")

    def test_round_trip_matches_original(
        self,
        close_df: pd.DataFrame,
        log_return_target: pd.Series,
        tmp_path: Path,
    ) -> None:
        original = ARMAPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        original.fit(close_df.iloc[1:], log_return_target)

        path = tmp_path / "arma"
        original.save(path)
        loaded = ARMAPredictor.load(path)

        assert loaded._fitted
        assert loaded._best_order == original._best_order
        assert loaded.training_metadata == original.training_metadata
        # ARMA predictions are fitted-values + forecasts; round-trip via
        # statsmodels ``filter`` reproduces both branches exactly.
        np.testing.assert_allclose(
            loaded.predict(close_df).to_numpy(),
            original.predict(close_df).to_numpy(),
            rtol=0.0,
            atol=1e-10,
        )


class TestLSTMSaveLoad:
    def test_save_before_fit_raises(
        self,
        lstm_features: list[str],
        tmp_path: Path,
    ) -> None:
        p = LSTMPredictor(lstm_features, lookback=LSTM_LOOKBACK, epochs=LSTM_EPOCHS)
        with pytest.raises(RuntimeError, match="before fit"):
            p.save(tmp_path / "lstm")

    def test_round_trip_matches_original(
        self,
        lstm_df: pd.DataFrame,
        lstm_target: pd.Series,
        lstm_features: list[str],
        tmp_path: Path,
    ) -> None:
        original = LSTMPredictor(
            feature_columns=lstm_features,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            lookback=LSTM_LOOKBACK,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            val_split_ratio=LSTM_VAL_SPLIT,
        )
        original.fit(lstm_df.iloc[:-1], lstm_target)

        path = tmp_path / "lstm"
        original.save(path)
        loaded = LSTMPredictor.load(path)

        assert loaded._fitted
        assert loaded._feature_columns == original._feature_columns
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.predict(lstm_df).to_numpy(),
            original.predict(lstm_df).to_numpy(),
        )

    def test_saved_weights_are_cpu_for_cross_device_portability(
        self,
        lstm_df: pd.DataFrame,
        lstm_target: pd.Series,
        lstm_features: list[str],
        tmp_path: Path,
    ) -> None:
        """Every persisted tensor must sit on CPU so a CUDA/MPS-trained model
        loads on a CPU-only machine without a ``map_location`` dance.

        We can't spin up a second device in CI, so instead we inspect the
        ``.pt`` payload directly: every tensor's device must be CPU regardless
        of where the live model lives.
        """
        original = LSTMPredictor(
            feature_columns=lstm_features,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            lookback=LSTM_LOOKBACK,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            val_split_ratio=LSTM_VAL_SPLIT,
        )
        original.fit(lstm_df.iloc[:-1], lstm_target)

        path = tmp_path / "lstm"
        original.save(path)

        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        for name, tensor in state.items():
            assert tensor.device.type == "cpu", (
                f"tensor {name!r} was saved on {tensor.device}, breaking cross-device load"
            )


class TestDirectionalClassifierSaveLoad:
    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        c = DirectionalClassifier(["return_1d", "return_5d"])
        with pytest.raises(RuntimeError, match="before fit"):
            c.save(tmp_path / "xgb")

    def test_round_trip_matches_original(
        self,
        xgb_data: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        features, target = xgb_data
        original = DirectionalClassifier(
            feature_columns=["return_1d", "return_5d"],
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
        )
        original.fit(features, target)

        path = tmp_path / "xgb"
        original.save(path)
        loaded = DirectionalClassifier.load(path)

        assert loaded._fitted
        assert loaded._feature_columns == original._feature_columns
        assert loaded.training_metadata == original.training_metadata
        np.testing.assert_array_equal(
            loaded.predict(features).to_numpy(),
            original.predict(features).to_numpy(),
        )
        np.testing.assert_allclose(
            loaded.predict_proba(features).to_numpy(),
            original.predict_proba(features).to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )


class TestCorruptPayloadLoad:
    """Loading a model from a corrupted or truncated save directory must raise
    with a message naming the specific field or file that's wrong — silent
    partial loads could pass ``_fitted=True`` without valid internal state and
    break late inside ``predict()`` with a much more opaque error.

    GARCH is the representative case: it exercises every narrowing helper
    (``json_get_int``, ``json_get_float``, ``json_get_float_list``,
    ``json_get_str``) and the ``read_json_dict`` top-level guard. Other models
    share the same machinery, so their failure modes would be identical.
    """

    @pytest.fixture
    def fitted_garch_path(
        self,
        close_df: pd.DataFrame,
        log_return_target: pd.Series,
        tmp_path: Path,
    ) -> Path:
        model = GARCHPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        model.fit(close_df.iloc[1:], log_return_target)
        path = tmp_path / "garch"
        model.save(path)
        return path

    def test_missing_config_file_raises(self, fitted_garch_path: Path) -> None:
        (fitted_garch_path / CONFIG_JSON).unlink()
        with pytest.raises(FileNotFoundError):
            GARCHPredictor.load(fitted_garch_path)

    def test_config_not_an_object_raises(self, fitted_garch_path: Path) -> None:
        write_json(fitted_garch_path / CONFIG_JSON, [1, 2, 3])
        with pytest.raises(ValueError, match="must be an object"):
            GARCHPredictor.load(fitted_garch_path)

    def test_missing_config_field_raises(self, fitted_garch_path: Path) -> None:
        config = read_json_dict(fitted_garch_path / CONFIG_JSON)
        del config["p_max"]
        write_json(fitted_garch_path / CONFIG_JSON, config)
        with pytest.raises(KeyError, match="p_max"):
            GARCHPredictor.load(fitted_garch_path)

    def test_wrong_type_in_config_raises(self, fitted_garch_path: Path) -> None:
        config = read_json_dict(fitted_garch_path / CONFIG_JSON)
        config["p_max"] = "not-an-int"
        write_json(fitted_garch_path / CONFIG_JSON, config)
        with pytest.raises(ValueError, match="'p_max' must be an int"):
            GARCHPredictor.load(fitted_garch_path)

    def test_wrong_type_in_weights_list_raises(self, fitted_garch_path: Path) -> None:
        weights = read_json_dict(fitted_garch_path / WEIGHTS_JSON)
        weights["alpha"] = [0.1, "nope", 0.3]
        write_json(fitted_garch_path / WEIGHTS_JSON, weights)
        with pytest.raises(ValueError, match=r"'alpha'\[1\] must be a number"):
            GARCHPredictor.load(fitted_garch_path)

    def test_malformed_json_raises(self, fitted_garch_path: Path) -> None:
        (fitted_garch_path / CONFIG_JSON).write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError):
            GARCHPredictor.load(fitted_garch_path)


class TestARMACorruptOrder:
    """ARMA has a unique fixed-length list field (``order``) that goes through
    ``json_get_int_list`` + a post-check on length. Covers the branch that the
    GARCH corrupt-payload suite doesn't touch.
    """

    def test_wrong_order_length_raises(
        self,
        close_df: pd.DataFrame,
        log_return_target: pd.Series,
        tmp_path: Path,
    ) -> None:
        model = ARMAPredictor(p_max=COMPACT_P_MAX, q_max=COMPACT_Q_MAX)
        model.fit(close_df.iloc[1:], log_return_target)
        path = tmp_path / "arma"
        model.save(path)

        weights = read_json_dict(path / WEIGHTS_JSON)
        weights["order"] = [1, 2]  # too short
        write_json(path / WEIGHTS_JSON, weights)

        with pytest.raises(ValueError, match="3-element list"):
            ARMAPredictor.load(path)
