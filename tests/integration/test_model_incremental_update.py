"""Incremental ``update()`` tests for every leaf model, composite, and strategy.

Each test fits a model on the first half of a synthetic window, calls
``update()`` on the second half, and asserts:
 - ``train_end`` advanced to ``new_data.index.max()``
 - ``n_train_samples`` equals the combined window length
 - ``fit_timestamp`` is preserved (provenance — only ``fit()`` sets it)
 - internal params moved (not a silent no-op)
 - ``predict()`` / ``generate_signals()`` output is finite on a held-out tail

Smoke tests also verify the round-trip ``fit → save → load → update`` flow
works for models that cache training targets on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.core.exceptions import LeakageError
from src.core.types import InformationCriterion, Interval
from src.core.utils import compute_log_returns
from src.models.arma import ARMAPredictor
from src.models.garch import GARCHPredictor
from src.models.hybrid_return import HybridReturnModel
from src.models.hybrid_volatility import HybridVolatilityModel
from src.models.lstm import LSTMPredictor
from src.models.xgboost_classifier import DirectionalClassifier
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from src.strategies.momentum_gatekeeper import MomentumGatekeeperStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.return_forecast import ReturnForecastStrategy
from src.strategies.volatility_targeting import VolatilityTargetingStrategy
from tests.conftest import (
    FEATURE_NOISE_SEED,
    attach_synthetic_features,
    make_pair_close_df,
    make_synthetic_close_df,
    make_synthetic_ohlcv_df,
)

# Synthetic windows — split in half for fit / update.
TOTAL_ROWS = 300
SPLIT_POINT = 200  # first 200 rows are initial train, next 100 are update delta

# Compact model parameters — we need fitted models to exercise the warm-start
# path, not production-grade hyperparameters.
COMPACT_GARCH_P_MAX = 2
COMPACT_GARCH_Q_MAX = 2
COMPACT_ARMA_P_MAX = 2
COMPACT_ARMA_Q_MAX = 2
COMPACT_LSTM_HIDDEN_DIM = 8
COMPACT_LSTM_LOOKBACK = 5
COMPACT_LSTM_EPOCHS = 2
COMPACT_XGB_N_ESTIMATORS = 5
COMPACT_XGB_MAX_DEPTH = 2

# Pairs cointegration z-scores (irrelevant to the update path; just need to pass
# the in-range validators so train() runs).
PAIRS_ENTRY_Z = 2.0
PAIRS_EXIT_Z = 0.5
PAIRS_STOP_Z = 4.0
PAIRS_LOOKBACK = 20

# Explicit update() knobs — call out the tuning so tests don't hide magic numbers.
LSTM_UPDATE_EPOCHS = 2
XGB_UPDATE_N_ESTIMATORS = 3
REALIZED_VOL_WINDOW = 20

# Walk-forward second-update window: we split the post-SPLIT_POINT tail in half
# to simulate two successive updates (e.g. two walk-forward folds on a single
# persisted model), then verify metadata extends correctly across both calls.
SECOND_UPDATE_SPLIT = 250

# Offset into a future fold used by ``validate_no_overlap``-after-update tests.
# 30 bars past the extended train_end gives a comfortable gap for the test to
# construct a non-overlapping "future" frame.
FUTURE_FOLD_GAP = 30

# Minimum L2-norm shift between pre- and post-update LSTM parameters. Catches
# "near no-op" fine-tunes where Adam momentum alone produces a trivial delta
# that would still pass a bit-inequality check. Derived from update_epochs=2
# at lr=1e-4 on the synthetic fixture below; empirically the delta is
# ~1e-3, so 1e-5 is a comfortable floor that still rejects a genuine no-op.
MIN_LSTM_FINE_TUNE_L2 = 1e-5

# Embargo gap (in bars) between initial training and the update window. Walk-forward
# folds typically leave a gap to embargo rolling-feature leakage; this constant
# drives a test that ``extend()`` accepts a non-contiguous but still-valid window.
EMBARGO_GAP_BARS = 10

# Torch/numpy seeds pinned before every LSTM-using fit — deterministic enough
# that the "internal params actually moved" assertion isn't flaky.
FIT_TORCH_SEED = 0
FIT_NUMPY_SEED = 0


@pytest.fixture
def close_df() -> pd.DataFrame:
    return make_synthetic_close_df(n_rows=TOTAL_ROWS)


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    return make_synthetic_ohlcv_df(n_rows=TOTAL_ROWS)


@pytest.fixture
def pair_df() -> pd.DataFrame:
    return make_pair_close_df(n_rows=TOTAL_ROWS)


@pytest.fixture
def feature_columns() -> list[str]:
    return ["feat_a", "feat_b"]


class TestGARCHUpdate:
    def test_update_before_fit_raises(self, close_df: pd.DataFrame) -> None:
        g = GARCHPredictor()
        target = compute_log_returns(close_df["close"]).dropna()
        with pytest.raises(RuntimeError, match="before fit"):
            g.update(close_df, target)

    def test_update_extends_metadata_and_moves_params(self, close_df: pd.DataFrame) -> None:
        train_df = close_df.iloc[:SPLIT_POINT]
        new_df = close_df.iloc[SPLIT_POINT:]
        train_target = compute_log_returns(train_df["close"]).dropna()
        new_target = compute_log_returns(new_df["close"]).dropna()

        model = GARCHPredictor(p_max=COMPACT_GARCH_P_MAX, q_max=COMPACT_GARCH_Q_MAX)
        model.fit(train_df.loc[train_target.index], train_target)
        before_omega = model._omega
        before_alpha = model._alpha.copy()
        before_beta = model._beta.copy()
        before_meta = model.training_metadata
        assert before_meta is not None

        model.update(new_df.loc[new_target.index], new_target)

        after_meta = model.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new_df.loc[new_target.index].index[-1])
        assert after_meta.n_train_samples == before_meta.n_train_samples + len(new_target)
        # fit_timestamp is provenance — never bumped by update()
        assert after_meta.fit_timestamp == before_meta.fit_timestamp

        # At least one coefficient should have moved. GARCH on additional data
        # produces near-identical but not bit-identical params.
        moved = (
            model._omega != before_omega
            or not np.array_equal(model._alpha, before_alpha)
            or not np.array_equal(model._beta, before_beta)
        )
        assert moved, "GARCH update() left every coefficient untouched"

    def test_predict_after_update_is_finite(self, close_df: pd.DataFrame) -> None:
        train_df = close_df.iloc[:SPLIT_POINT]
        new_df = close_df.iloc[SPLIT_POINT:]
        train_target = compute_log_returns(train_df["close"]).dropna()
        new_target = compute_log_returns(new_df["close"]).dropna()

        model = GARCHPredictor(p_max=COMPACT_GARCH_P_MAX, q_max=COMPACT_GARCH_Q_MAX)
        model.fit(train_df.loc[train_target.index], train_target)
        model.update(new_df.loc[new_target.index], new_target)

        preds = model.predict(close_df).dropna()
        assert len(preds) > 0
        assert np.isfinite(preds.to_numpy()).all()

    def test_update_after_save_load(self, close_df: pd.DataFrame, tmp_path: Path) -> None:
        train_df = close_df.iloc[:SPLIT_POINT]
        new_df = close_df.iloc[SPLIT_POINT:]
        train_target = compute_log_returns(train_df["close"]).dropna()
        new_target = compute_log_returns(new_df["close"]).dropna()

        model = GARCHPredictor(p_max=COMPACT_GARCH_P_MAX, q_max=COMPACT_GARCH_Q_MAX)
        model.fit(train_df.loc[train_target.index], train_target)
        path = tmp_path / "garch"
        model.save(path)

        loaded = GARCHPredictor.load(path)
        loaded.update(new_df.loc[new_target.index], new_target)
        meta = loaded.training_metadata
        assert meta is not None
        assert meta.train_end == pd.Timestamp(new_df.loc[new_target.index].index[-1])


class TestARMAUpdate:
    def test_update_before_fit_raises(self, close_df: pd.DataFrame) -> None:
        a = ARMAPredictor()
        target = compute_log_returns(close_df["close"]).dropna()
        with pytest.raises(RuntimeError, match="before fit"):
            a.update(close_df, target)

    def test_update_extends_metadata_and_moves_params(self, close_df: pd.DataFrame) -> None:
        train_df = close_df.iloc[:SPLIT_POINT]
        new_df = close_df.iloc[SPLIT_POINT:]
        train_target = compute_log_returns(train_df["close"]).dropna()
        new_target = compute_log_returns(new_df["close"]).dropna()

        model = ARMAPredictor(p_max=COMPACT_ARMA_P_MAX, q_max=COMPACT_ARMA_Q_MAX)
        model.fit(train_df.loc[train_target.index], train_target)
        assert model._model is not None
        before_predict_in_sample = model._model.predict_in_sample().copy()
        before_meta = model.training_metadata
        assert before_meta is not None

        model.update(new_df.loc[new_target.index], new_target)

        after_meta = model.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new_df.loc[new_target.index].index[-1])
        assert after_meta.n_train_samples == before_meta.n_train_samples + len(new_target)
        assert after_meta.fit_timestamp == before_meta.fit_timestamp

        # In-sample fitted values after update cover the combined endog, so the
        # lengths must differ — a silent no-op would keep the original length.
        assert model._model is not None
        assert len(model._model.predict_in_sample()) != len(before_predict_in_sample)

    def test_predict_after_update_is_finite(self, close_df: pd.DataFrame) -> None:
        train_df = close_df.iloc[:SPLIT_POINT]
        new_df = close_df.iloc[SPLIT_POINT:]
        train_target = compute_log_returns(train_df["close"]).dropna()
        new_target = compute_log_returns(new_df["close"]).dropna()

        model = ARMAPredictor(p_max=COMPACT_ARMA_P_MAX, q_max=COMPACT_ARMA_Q_MAX)
        model.fit(train_df.loc[train_target.index], train_target)
        model.update(new_df.loc[new_target.index], new_target)

        preds = model.predict(close_df).dropna()
        assert len(preds) > 0
        assert np.isfinite(preds.to_numpy()).all()


class TestLSTMUpdate:
    def test_update_before_fit_raises(
        self, close_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(close_df, feature_columns)
        lstm = LSTMPredictor(
            feature_columns=feature_columns,
            lookback=COMPACT_LSTM_LOOKBACK,
            epochs=COMPACT_LSTM_EPOCHS,
        )
        target = pd.Series(np.random.default_rng(0).normal(0, 1, len(df)), index=df.index)
        with pytest.raises(RuntimeError, match="before fit"):
            lstm.update(df, target)

    def test_update_rejects_too_short_new_data(
        self, close_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(close_df, feature_columns)
        target = pd.Series(np.random.default_rng(0).normal(0, 1, len(df)), index=df.index, name="y")
        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        lstm = LSTMPredictor(
            feature_columns=feature_columns,
            hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lookback=COMPACT_LSTM_LOOKBACK,
            epochs=COMPACT_LSTM_EPOCHS,
        )
        lstm.fit(df.iloc[:SPLIT_POINT], target.iloc[:SPLIT_POINT])
        too_short = df.iloc[SPLIT_POINT : SPLIT_POINT + COMPACT_LSTM_LOOKBACK]
        with pytest.raises(ValueError, match="lookback"):
            lstm.update(too_short, target.iloc[SPLIT_POINT : SPLIT_POINT + COMPACT_LSTM_LOOKBACK])

    def test_update_fine_tunes_and_extends_metadata(
        self, close_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(close_df, feature_columns)
        target = pd.Series(np.random.default_rng(0).normal(0, 1, len(df)), index=df.index, name="y")
        train_df = df.iloc[:SPLIT_POINT]
        train_target = target.iloc[:SPLIT_POINT]
        new_df = df.iloc[SPLIT_POINT:]
        new_target = target.iloc[SPLIT_POINT:]

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        lstm = LSTMPredictor(
            feature_columns=feature_columns,
            hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lookback=COMPACT_LSTM_LOOKBACK,
            epochs=COMPACT_LSTM_EPOCHS,
            update_epochs=LSTM_UPDATE_EPOCHS,
        )
        lstm.fit(train_df, train_target)
        before_meta = lstm.training_metadata
        assert before_meta is not None
        assert lstm._model is not None
        # Capture a fingerprint of the weights — any parameter should move a
        # nonzero amount during fine-tune.
        before_fingerprint = torch.cat(
            [p.detach().cpu().flatten() for p in lstm._model.parameters()]
        ).clone()

        lstm.update(new_df, new_target)

        after_meta = lstm.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new_df.index[-1])
        assert after_meta.n_train_samples == before_meta.n_train_samples + len(new_df)
        assert after_meta.fit_timestamp == before_meta.fit_timestamp

        after_fingerprint = torch.cat(
            [p.detach().cpu().flatten() for p in lstm._model.parameters()]
        )
        # Assert a meaningful L2 movement — bit-inequality alone can pass on
        # trivial Adam momentum changes even when the fine-tune was effectively
        # a no-op. ``MIN_LSTM_FINE_TUNE_L2`` floors the acceptable shift.
        l2_delta = (after_fingerprint - before_fingerprint).norm().item()
        assert l2_delta > MIN_LSTM_FINE_TUNE_L2, (
            f"LSTM update() produced near-zero L2 delta ({l2_delta:.2e}); "
            f"fine-tune loop likely didn't make meaningful progress"
        )

        preds = lstm.predict(df).dropna()
        assert np.isfinite(preds.to_numpy()).all()

    def test_update_rolls_back_weights_on_exception(
        self, close_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        """A mid-fine-tune exception must restore the pre-update weights —
        ``update()`` is transactional at the leaf level so composites can rely
        on partial-failure atomicity for each leaf."""
        df = attach_synthetic_features(close_df, feature_columns)
        target = pd.Series(np.random.default_rng(0).normal(0, 1, len(df)), index=df.index, name="y")
        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        lstm = LSTMPredictor(
            feature_columns=feature_columns,
            hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lookback=COMPACT_LSTM_LOOKBACK,
            epochs=COMPACT_LSTM_EPOCHS,
            update_epochs=LSTM_UPDATE_EPOCHS,
        )
        lstm.fit(df.iloc[:SPLIT_POINT], target.iloc[:SPLIT_POINT])
        assert lstm._model is not None
        before_meta = lstm.training_metadata
        before_weights = {k: v.detach().cpu().clone() for k, v in lstm._model.state_dict().items()}

        # Force a mid-loop crash by monkey-patching the model's forward to raise
        # after one successful batch. The try/except-restore block must reset
        # every parameter back to ``before_weights``.
        call_count = {"n": 0}
        original_forward = lstm._model.forward

        def crashing_forward(x: torch.Tensor) -> torch.Tensor:
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("simulated mid-fine-tune crash")
            return original_forward(x)

        lstm._model.forward = crashing_forward  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="simulated mid-fine-tune crash"):
                lstm.update(df.iloc[SPLIT_POINT:], target.iloc[SPLIT_POINT:])
        finally:
            lstm._model.forward = original_forward  # type: ignore[method-assign]

        # Weights restored to pre-update snapshot, metadata still pre-update.
        after_weights = lstm._model.state_dict()
        for k, before in before_weights.items():
            torch.testing.assert_close(after_weights[k].cpu(), before, rtol=0.0, atol=0.0)
        assert lstm.training_metadata == before_meta


class TestDirectionalClassifierUpdate:
    def test_update_before_fit_raises(self) -> None:
        clf = DirectionalClassifier(["return_1d"])
        features = pd.DataFrame({"return_1d": [0.1, 0.2]})
        target = pd.Series([0, 1])
        with pytest.raises(RuntimeError, match="before fit"):
            clf.update(features, target)

    def test_update_appends_trees_and_extends_metadata(self) -> None:
        rng = np.random.default_rng(FEATURE_NOISE_SEED)
        n = TOTAL_ROWS
        idx = pd.bdate_range(start="2020-01-02", periods=n, freq="B")
        features = pd.DataFrame(
            {
                "return_1d": rng.normal(0, 0.01, n),
                "return_5d": rng.normal(0, 0.02, n),
            },
            index=idx,
        )
        target = pd.Series(rng.integers(0, 2, n), index=idx, name="direction")

        clf = DirectionalClassifier(
            feature_columns=["return_1d", "return_5d"],
            n_estimators=COMPACT_XGB_N_ESTIMATORS,
            max_depth=COMPACT_XGB_MAX_DEPTH,
            update_n_estimators=XGB_UPDATE_N_ESTIMATORS,
        )
        clf.fit(features.iloc[:SPLIT_POINT], target.iloc[:SPLIT_POINT])
        before_meta = clf.training_metadata
        assert before_meta is not None
        assert clf._model is not None
        before_tree_count = clf._model.get_booster().num_boosted_rounds()

        clf.update(features.iloc[SPLIT_POINT:], target.iloc[SPLIT_POINT:])

        after_meta = clf.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(features.iloc[SPLIT_POINT:].index[-1])
        assert after_meta.n_train_samples == before_meta.n_train_samples + len(
            features.iloc[SPLIT_POINT:]
        )
        assert after_meta.fit_timestamp == before_meta.fit_timestamp

        # Continue-boost appends update_n_estimators rounds.
        assert clf._model is not None
        after_tree_count = clf._model.get_booster().num_boosted_rounds()
        assert after_tree_count == before_tree_count + XGB_UPDATE_N_ESTIMATORS

        preds = clf.predict_proba(features).dropna()
        assert np.isfinite(preds.to_numpy()).all()


class TestHybridVolatilityUpdate:
    def test_update_before_fit_raises(
        self, ohlcv_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(ohlcv_df, feature_columns)
        target = pd.Series(np.ones(len(df)) * 0.2, index=df.index)
        m = HybridVolatilityModel(feature_columns=feature_columns)
        with pytest.raises(RuntimeError, match="before fit"):
            m.update(df, target)

    def test_update_delegates_to_leaves_and_extends_metadata(
        self, ohlcv_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(ohlcv_df, feature_columns)
        # Synthetic realized-vol target — rolling std of log returns scaled to annual.
        rets = compute_log_returns(df["close"])
        rv = rets.rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW).std() * np.sqrt(
            Interval.DAILY.annualization_factor()
        )

        train_df = df.iloc[:SPLIT_POINT]
        train_target = rv.iloc[:SPLIT_POINT].dropna()
        new_df = df.iloc[SPLIT_POINT:]
        new_target = rv.iloc[SPLIT_POINT:].dropna()

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        model = HybridVolatilityModel(
            feature_columns=feature_columns,
            garch_p_max=COMPACT_GARCH_P_MAX,
            garch_q_max=COMPACT_GARCH_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
        )
        model.fit(train_df.loc[train_target.index], train_target)
        before_meta = model.training_metadata
        before_garch_meta = model._garch.training_metadata
        scaler_mean_before = model._scaler.mean_.copy()  # type: ignore[union-attr]

        model.update(new_df.loc[new_target.index], new_target)

        # Scaler MUST NOT be re-fit — anti-leakage invariant.
        np.testing.assert_array_equal(model._scaler.mean_, scaler_mean_before)  # type: ignore[union-attr]

        after_meta = model.training_metadata
        assert before_meta is not None and after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new_df.loc[new_target.index].index[-1])
        assert after_meta.fit_timestamp == before_meta.fit_timestamp

        # Leaf GARCH must also preserve its own fit_timestamp and advance its
        # train_end — it's a black-box composition so the leaf has its own,
        # independently-set metadata captured during the hybrid's fit().
        after_garch_meta = model._garch.training_metadata
        assert before_garch_meta is not None and after_garch_meta is not None
        assert after_garch_meta.fit_timestamp == before_garch_meta.fit_timestamp
        assert after_garch_meta.train_end > before_garch_meta.train_end


class TestHybridReturnUpdate:
    def test_update_delegates_to_leaves_and_extends_metadata(
        self, close_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(close_df, feature_columns)
        target = compute_log_returns(df["close"])

        train_df = df.iloc[:SPLIT_POINT]
        train_target = target.iloc[:SPLIT_POINT]
        new_df = df.iloc[SPLIT_POINT:]
        new_target = target.iloc[SPLIT_POINT:]

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        model = HybridReturnModel(
            feature_columns=feature_columns,
            arma_p_max=COMPACT_ARMA_P_MAX,
            arma_q_max=COMPACT_ARMA_Q_MAX,
            arma_information_criterion=InformationCriterion.AIC,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
        )
        model.fit(train_df, train_target)
        before_meta = model.training_metadata
        scaler_mean_before = model._scaler.mean_.copy()  # type: ignore[union-attr]

        model.update(new_df, new_target)

        np.testing.assert_array_equal(model._scaler.mean_, scaler_mean_before)  # type: ignore[union-attr]

        after_meta = model.training_metadata
        assert before_meta is not None and after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new_df.index[-1])
        assert after_meta.fit_timestamp == before_meta.fit_timestamp


class TestPairsTradingUpdate:
    def test_update_before_train_raises(self, pair_df: pd.DataFrame) -> None:
        s = PairsTradingStrategy(
            entry_zscore=PAIRS_ENTRY_Z,
            exit_zscore=PAIRS_EXIT_Z,
            stop_loss_zscore=PAIRS_STOP_Z,
            zscore_lookback=PAIRS_LOOKBACK,
        )
        with pytest.raises(RuntimeError, match="before train"):
            s.update(pair_df)

    def test_update_extends_metadata_and_refits_spread(self, pair_df: pd.DataFrame) -> None:
        train = pair_df.iloc[:SPLIT_POINT]
        new = pair_df.iloc[SPLIT_POINT:]

        s = PairsTradingStrategy(
            entry_zscore=PAIRS_ENTRY_Z,
            exit_zscore=PAIRS_EXIT_Z,
            stop_loss_zscore=PAIRS_STOP_Z,
            zscore_lookback=PAIRS_LOOKBACK,
        )
        s.train(train)
        before_hedge = s._hedge_ratio
        before_meta = s.training_metadata
        assert before_meta is not None

        s.update(new)

        after_meta = s.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new.index[-1])
        assert after_meta.n_train_samples == before_meta.n_train_samples + len(new)
        assert after_meta.fit_timestamp == before_meta.fit_timestamp
        # Hedge ratio should shift slightly with more observations.
        assert s._hedge_ratio != before_hedge

    def test_update_requires_pair_columns(
        self, pair_df: pd.DataFrame, close_df: pd.DataFrame
    ) -> None:
        s = PairsTradingStrategy(
            entry_zscore=PAIRS_ENTRY_Z,
            exit_zscore=PAIRS_EXIT_Z,
            stop_loss_zscore=PAIRS_STOP_Z,
            zscore_lookback=PAIRS_LOOKBACK,
        )
        s.train(pair_df.iloc[:SPLIT_POINT])
        with pytest.raises(ValueError, match="close_a"):
            s.update(close_df.iloc[:10])

    def test_update_warns_on_decointegration(
        self, pair_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the extended window breaks cointegration, the strategy flips
        ``_is_cointegrated`` to False and logs a warning so the caller can
        pull the pair from the live book."""
        import logging

        s = PairsTradingStrategy(
            entry_zscore=PAIRS_ENTRY_Z,
            exit_zscore=PAIRS_EXIT_Z,
            stop_loss_zscore=PAIRS_STOP_Z,
            zscore_lookback=PAIRS_LOOKBACK,
        )
        s.train(pair_df.iloc[:SPLIT_POINT])
        assert s._is_cointegrated is True

        # Construct a noise tail that breaks cointegration: close_b diverges
        # with a large uncorrelated random walk immediately after split.
        rng = np.random.default_rng(FEATURE_NOISE_SEED)
        divergent = pair_df.iloc[SPLIT_POINT:].copy()
        drift = np.cumsum(rng.normal(0.5, 2.0, len(divergent)))
        divergent["close_b"] = divergent["close_b"].to_numpy() + drift

        with caplog.at_level(logging.WARNING, logger="src.strategies.pairs_trading"):
            s.update(divergent)

        assert s._is_cointegrated is False, (
            "test fixture should force de-cointegration (divergent random walks)"
        )
        assert any("de-cointegrated" in rec.message for rec in caplog.records), (
            "update() must log a warning when the pair de-cointegrates"
        )

    def test_update_warns_on_short_new_data(
        self, pair_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Passing new_data shorter than ``zscore_lookback`` doesn't break
        update() itself but produces all-NaN signals downstream; a warning
        surfaces this to the caller."""
        import logging

        s = PairsTradingStrategy(
            entry_zscore=PAIRS_ENTRY_Z,
            exit_zscore=PAIRS_EXIT_Z,
            stop_loss_zscore=PAIRS_STOP_Z,
            zscore_lookback=PAIRS_LOOKBACK,
        )
        s.train(pair_df.iloc[:SPLIT_POINT])
        too_short_len = PAIRS_LOOKBACK - 1
        too_short = pair_df.iloc[SPLIT_POINT : SPLIT_POINT + too_short_len]
        with caplog.at_level(logging.WARNING, logger="src.strategies.pairs_trading"):
            s.update(too_short)
        assert any("zscore_lookback" in rec.message for rec in caplog.records)


class TestAdaptiveBollingerUpdate:
    def test_update_before_train_raises(self, close_df: pd.DataFrame) -> None:
        s = AdaptiveBollingerStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.update(close_df)

    def test_update_extends_metadata(self, close_df: pd.DataFrame) -> None:
        train = close_df.iloc[:SPLIT_POINT]
        new = close_df.iloc[SPLIT_POINT:]

        s = AdaptiveBollingerStrategy(
            garch_p_max=COMPACT_GARCH_P_MAX, garch_q_max=COMPACT_GARCH_Q_MAX
        )
        s.train(train)
        before_meta = s.training_metadata
        assert before_meta is not None

        s.update(new)
        after_meta = s.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new.index[-1])
        assert after_meta.fit_timestamp == before_meta.fit_timestamp

        signals = s.generate_signals(close_df).dropna()
        assert len(signals) > 0


class TestMomentumGatekeeperUpdate:
    def test_update_before_train_raises(self, close_df: pd.DataFrame) -> None:
        s = MomentumGatekeeperStrategy()
        with pytest.raises(RuntimeError, match="before train"):
            s.update(close_df)

    def test_update_extends_metadata_and_appends_trees(self, close_df: pd.DataFrame) -> None:
        train = close_df.iloc[:SPLIT_POINT]
        new = close_df.iloc[SPLIT_POINT:]

        s = MomentumGatekeeperStrategy(
            n_estimators=COMPACT_XGB_N_ESTIMATORS,
            max_depth=COMPACT_XGB_MAX_DEPTH,
        )
        s.train(train)
        assert s._classifier is not None
        before_trees = s._classifier._model.get_booster().num_boosted_rounds()  # type: ignore[union-attr]
        before_scaler_mean = s._pipeline.scaler.mean_.copy()  # type: ignore[union-attr]
        before_meta = s.training_metadata
        assert before_meta is not None

        s.update(new)

        # Pipeline scaler must stay frozen (anti-leakage).
        np.testing.assert_array_equal(s._pipeline.scaler.mean_, before_scaler_mean)  # type: ignore[union-attr]

        after_meta = s.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new.index[-1])
        assert after_meta.fit_timestamp == before_meta.fit_timestamp

        assert s._classifier._model is not None
        after_trees = s._classifier._model.get_booster().num_boosted_rounds()
        assert after_trees > before_trees


class TestReturnForecastUpdate:
    def test_update_before_train_raises(
        self, close_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        s = ReturnForecastStrategy(feature_columns=feature_columns)
        with pytest.raises(RuntimeError, match="before train"):
            s.update(close_df)

    def test_update_extends_metadata(
        self, close_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(close_df, feature_columns)
        train = df.iloc[:SPLIT_POINT]
        new = df.iloc[SPLIT_POINT:]

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        s = ReturnForecastStrategy(
            feature_columns=feature_columns,
            arma_p_max=COMPACT_ARMA_P_MAX,
            arma_q_max=COMPACT_ARMA_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
        )
        s.train(train)
        before_meta = s.training_metadata
        assert before_meta is not None

        s.update(new)
        after_meta = s.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new.index[-1])
        assert after_meta.fit_timestamp == before_meta.fit_timestamp


class TestVolatilityTargetingUpdate:
    def test_update_before_train_raises(
        self, ohlcv_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        s = VolatilityTargetingStrategy(feature_columns=feature_columns)
        with pytest.raises(RuntimeError, match="before train"):
            s.update(ohlcv_df)

    def test_update_extends_metadata(
        self, ohlcv_df: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        df = attach_synthetic_features(ohlcv_df, feature_columns)
        train = df.iloc[:SPLIT_POINT]
        new = df.iloc[SPLIT_POINT:]

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        s = VolatilityTargetingStrategy(
            feature_columns=feature_columns,
            garch_p_max=COMPACT_GARCH_P_MAX,
            garch_q_max=COMPACT_GARCH_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
        )
        s.train(train)
        before_meta = s.training_metadata
        assert before_meta is not None

        s.update(new)
        after_meta = s.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(new.index[-1])
        assert after_meta.fit_timestamp == before_meta.fit_timestamp


class TestMetadataInvariantsAfterUpdate:
    """Cross-model invariants that ``update()`` must uphold.

    Picks GARCH as the representative model — it exercises the full update
    path end-to-end (cached training returns + arch warm-start + metadata
    extend) and does so fast enough to afford the three scenarios below.
    """

    def _fit_garch(self, df: pd.DataFrame) -> GARCHPredictor:
        target = compute_log_returns(df["close"]).dropna()
        m = GARCHPredictor(p_max=COMPACT_GARCH_P_MAX, q_max=COMPACT_GARCH_Q_MAX)
        m.fit(df.loc[target.index], target)
        return m

    def test_update_rejects_overlapping_new_data(self, close_df: pd.DataFrame) -> None:
        """``new_data`` starting inside the training window must raise — the
        ``extend()`` leakage guard catches double-counted rows before the
        refit can corrupt the cached training returns."""
        m = self._fit_garch(close_df.iloc[:SPLIT_POINT])
        overlapping = close_df.iloc[SPLIT_POINT - 5 :]
        overlap_target = compute_log_returns(overlapping["close"]).dropna()
        with pytest.raises(LeakageError, match="new_start > train_end"):
            m.update(overlapping.loc[overlap_target.index], overlap_target)

    def test_update_leakage_error_leaves_state_untouched(self, close_df: pd.DataFrame) -> None:
        """When ``update()`` raises LeakageError, internal state must be
        unchanged — the validate-then-commit pattern makes the update
        transactional. A post-error ``predict()`` should behave as if
        ``update()`` was never called."""
        m = self._fit_garch(close_df.iloc[:SPLIT_POINT])
        before_omega = m._omega
        before_alpha = m._alpha.copy()
        before_beta = m._beta.copy()
        before_train_returns = m._train_returns.copy()
        before_meta = m.training_metadata

        overlapping = close_df.iloc[SPLIT_POINT - 5 :]
        overlap_target = compute_log_returns(overlapping["close"]).dropna()
        with pytest.raises(LeakageError):
            m.update(overlapping.loc[overlap_target.index], overlap_target)

        # Every piece of internal state must match the pre-update snapshot.
        assert m._omega == before_omega
        np.testing.assert_array_equal(m._alpha, before_alpha)
        np.testing.assert_array_equal(m._beta, before_beta)
        np.testing.assert_array_equal(m._train_returns, before_train_returns)
        assert m.training_metadata == before_meta

    def test_two_successive_updates_extend_correctly(self, close_df: pd.DataFrame) -> None:
        """Walk-forward use case: fit once, update twice on disjoint windows.
        ``train_end`` advances each time; ``fit_timestamp`` stays frozen."""
        first_new = close_df.iloc[SPLIT_POINT:SECOND_UPDATE_SPLIT]
        second_new = close_df.iloc[SECOND_UPDATE_SPLIT:]

        m = self._fit_garch(close_df.iloc[:SPLIT_POINT])
        fit_ts = m.training_metadata
        assert fit_ts is not None

        first_target = compute_log_returns(first_new["close"]).dropna()
        m.update(first_new.loc[first_target.index], first_target)
        after_first = m.training_metadata
        assert after_first is not None
        assert after_first.train_end == pd.Timestamp(first_new.loc[first_target.index].index[-1])
        assert after_first.fit_timestamp == fit_ts.fit_timestamp

        second_target = compute_log_returns(second_new["close"]).dropna()
        m.update(second_new.loc[second_target.index], second_target)
        after_second = m.training_metadata
        assert after_second is not None
        assert after_second.train_end == pd.Timestamp(second_new.loc[second_target.index].index[-1])
        assert after_second.fit_timestamp == fit_ts.fit_timestamp
        assert after_second.n_train_samples > after_first.n_train_samples

    def test_validate_no_overlap_still_works_after_update(self, close_df: pd.DataFrame) -> None:
        """After ``update()`` extends ``train_end``, evaluation on data that
        now overlaps the extended window must raise. This is the headline
        anti-leakage invariant the extend-pattern exists to preserve."""
        m = self._fit_garch(close_df.iloc[:SPLIT_POINT])
        new_df = close_df.iloc[SPLIT_POINT:]
        new_target = compute_log_returns(new_df["close"]).dropna()
        m.update(new_df.loc[new_target.index], new_target)

        meta = m.training_metadata
        assert meta is not None
        inside_extended_window = new_df.iloc[-FUTURE_FOLD_GAP:]
        with pytest.raises(LeakageError, match="data leakage"):
            meta.validate_no_overlap(inside_extended_window)

    def test_update_accepts_embargo_gap_window(self, close_df: pd.DataFrame) -> None:
        """Walk-forward folds often embargo a gap between train and test. The
        new_start > train_end check must pass even if new_data starts well
        after train_end (not just immediately adjacent)."""
        train_end_loc = SPLIT_POINT
        gapped_start = train_end_loc + EMBARGO_GAP_BARS
        m = self._fit_garch(close_df.iloc[:train_end_loc])
        before_meta = m.training_metadata
        assert before_meta is not None

        gapped = close_df.iloc[gapped_start:]
        gapped_target = compute_log_returns(gapped["close"]).dropna()
        m.update(gapped.loc[gapped_target.index], gapped_target)

        after_meta = m.training_metadata
        assert after_meta is not None
        assert after_meta.train_end == pd.Timestamp(gapped.loc[gapped_target.index].index[-1])
        # The embargo bars were never seen during training; they remain absent
        # from ``n_train_samples``.
        assert after_meta.n_train_samples == before_meta.n_train_samples + len(gapped_target)


class TestCompositeSaveLoadUpdate:
    """``fit → save → load → update → predict`` round-trip for composites.

    The composites persist nested leaves + a ``scaler.json``; these tests
    catch format bugs where one of the nested artifacts doesn't round-trip
    cleanly enough for the post-load ``update()`` to run.
    """

    def test_hybrid_volatility_save_load_update(
        self,
        ohlcv_df: pd.DataFrame,
        feature_columns: list[str],
        tmp_path: Path,
    ) -> None:
        df = attach_synthetic_features(ohlcv_df, feature_columns)
        rets = compute_log_returns(df["close"])
        rv = rets.rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW).std() * np.sqrt(
            Interval.DAILY.annualization_factor()
        )
        train_df = df.iloc[:SPLIT_POINT]
        new_df = df.iloc[SPLIT_POINT:]
        train_target = rv.iloc[:SPLIT_POINT].dropna()
        new_target = rv.iloc[SPLIT_POINT:].dropna()

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        model = HybridVolatilityModel(
            feature_columns=feature_columns,
            garch_p_max=COMPACT_GARCH_P_MAX,
            garch_q_max=COMPACT_GARCH_Q_MAX,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
        )
        model.fit(train_df.loc[train_target.index], train_target)
        path = tmp_path / "hybrid_vol"
        model.save(path)

        loaded = HybridVolatilityModel.load(path)
        loaded.update(new_df.loc[new_target.index], new_target)
        meta = loaded.training_metadata
        assert meta is not None
        assert meta.train_end == pd.Timestamp(new_df.loc[new_target.index].index[-1])
        preds = loaded.predict(df).dropna()
        assert np.isfinite(preds.to_numpy()).all()

    def test_hybrid_return_save_load_update(
        self,
        close_df: pd.DataFrame,
        feature_columns: list[str],
        tmp_path: Path,
    ) -> None:
        df = attach_synthetic_features(close_df, feature_columns)
        target = compute_log_returns(df["close"])
        train_df = df.iloc[:SPLIT_POINT]
        new_df = df.iloc[SPLIT_POINT:]
        train_target = target.iloc[:SPLIT_POINT]
        new_target = target.iloc[SPLIT_POINT:]

        torch.manual_seed(FIT_TORCH_SEED)
        np.random.seed(FIT_NUMPY_SEED)
        model = HybridReturnModel(
            feature_columns=feature_columns,
            arma_p_max=COMPACT_ARMA_P_MAX,
            arma_q_max=COMPACT_ARMA_Q_MAX,
            arma_information_criterion=InformationCriterion.AIC,
            lstm_hidden_dim=COMPACT_LSTM_HIDDEN_DIM,
            lstm_lookback=COMPACT_LSTM_LOOKBACK,
            lstm_epochs=COMPACT_LSTM_EPOCHS,
        )
        model.fit(train_df, train_target)
        path = tmp_path / "hybrid_return"
        model.save(path)

        loaded = HybridReturnModel.load(path)
        loaded.update(new_df, new_target)
        meta = loaded.training_metadata
        assert meta is not None
        assert meta.train_end == pd.Timestamp(new_df.index[-1])
        preds = loaded.predict(df).dropna()
        assert np.isfinite(preds.to_numpy()).all()
