"""Per-strategy pretrained-leaf injection behaviour.

Each IN-scope strategy gets:

* A happy-path test — a fake leaf with matching interval + feature_columns
  + lookback passes ``validate_pretrained_leaf`` and ctor stores it.
* A ``train()`` invariant — when the leaf is pretrained, ``train()``
  updates ``_training_metadata`` but does NOT call ``fit()`` on the leaf
  (verified via a fake that counts calls).
* A ``get_all_training_metadata`` invariant — every metadata entry
  inherited from the leaf has ``is_pretrained=True``.

Non-IN-scope strategies (AdaptiveBollinger, PairsTrading) verify that
non-empty ``pretrained_leaves`` raises with the actionable reason.

Fake leaves duck-type the public HybridReturnModel / HybridVolatilityModel
surface that the strategy's ``train()`` / ``generate_signals()`` would
touch. No real ML training runs here — that's covered by
``test_standalone_training.py`` and the gated smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.core.temporal import TrackedMetadata, TrainingMetadata
from src.core.types import Interval
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from src.strategies.momentum_gatekeeper import MomentumGatekeeperStrategy
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.return_forecast import ReturnForecastStrategy
from src.strategies.volatility_targeting import VolatilityTargetingStrategy
from tests.conftest import make_synthetic_ohlcv_df

if TYPE_CHECKING:
    pass

_FEATURES: tuple[str, ...] = ("sma_20", "rsi_14", "volume_z")
# MomentumGatekeeper owns a real FeatureEngineeringPipeline whose output
# columns are fixed by the pipeline's formulas (not configurable). The
# pretrained classifier's ``training_metadata.feature_columns`` must be
# a subset of those produced columns; pick three with default periods so
# the pipeline produces them at the ctor defaults.
_MOMENTUM_FEATURES: tuple[str, ...] = ("rsi_14", "macd", "macd_signal")
_INTERVAL = Interval.DAILY
_LSTM_LOOKBACK = 10
_TRAIN_ROWS = 80
_OHLCV_SEED = 7
_LEAF_N_TRAIN_SAMPLES = 250
_TRAIN_START = pd.Timestamp("2019-01-02")
_TRAIN_END = pd.Timestamp("2019-12-31")
_LEAF_FIT_TIMESTAMP = pd.Timestamp("2020-01-05")


def _leaf_metadata(
    feature_columns: tuple[str, ...] = _FEATURES,
    interval: Interval = _INTERVAL,
) -> TrainingMetadata:
    return TrainingMetadata(
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        n_train_samples=_LEAF_N_TRAIN_SAMPLES,
        fit_timestamp=_LEAF_FIT_TIMESTAMP,
        interval=interval,
        feature_columns=feature_columns,
    )


@dataclass
class _FakeLstm:
    _lookback: int = _LSTM_LOOKBACK


@dataclass
class _FakeHybridLeaf:
    """Duck-types the HybridReturnModel / HybridVolatilityModel surface
    the strategies touch when a leaf is pretrained-injected.

    Tracks ``fit_calls`` to let tests assert the strategy respects the
    frozen invariant (no leaf refit across ``train()`` calls).
    """

    training_metadata: TrainingMetadata | None
    _lstm: _FakeLstm
    fit_calls: int = 0
    training_metadata_entries: tuple[TrackedMetadata, ...] = field(
        default_factory=lambda: (
            TrackedMetadata(origin="arma", metadata=_leaf_metadata()),
            TrackedMetadata(origin="lstm", metadata=_leaf_metadata()),
        )
    )

    def fit(self, df: pd.DataFrame, target: pd.Series, **_: object) -> None:
        self.fit_calls += 1

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index)

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        return self.training_metadata_entries


@dataclass
class _FakeClassifier:
    """Duck-types the DirectionalClassifier surface MomentumGatekeeperStrategy
    touches when the classifier is pretrained-injected. Different public API
    than the hybrid leaves: ``predict_proba`` (not ``predict``) and no
    recursive ``get_all_training_metadata`` — the strategy reads
    ``training_metadata`` directly.
    """

    training_metadata: TrainingMetadata | None
    fit_calls: int = 0

    def fit(self, df: pd.DataFrame, target: pd.Series, **_: object) -> None:
        self.fit_calls += 1

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([0.5] * len(df), index=df.index, name="up_prob")


def _ohlcv_df() -> pd.DataFrame:
    df = make_synthetic_ohlcv_df(n_rows=_TRAIN_ROWS, seed=_OHLCV_SEED)
    for col in _FEATURES:
        df[col] = 0.0
    return df


@pytest.fixture
def train_df() -> pd.DataFrame:
    return _ohlcv_df()


@pytest.fixture
def fake_leaf() -> _FakeHybridLeaf:
    return _FakeHybridLeaf(
        training_metadata=_leaf_metadata(),
        _lstm=_FakeLstm(_lookback=_LSTM_LOOKBACK),
    )


@pytest.fixture
def fake_classifier() -> _FakeClassifier:
    return _FakeClassifier(
        training_metadata=_leaf_metadata(feature_columns=_MOMENTUM_FEATURES),
    )


class TestReturnForecastPretrainedInjection:
    def test_ctor_stores_leaf_and_skips_build(self, fake_leaf: _FakeHybridLeaf) -> None:
        s = ReturnForecastStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"return_model": fake_leaf},
        )
        # Typed as HybridReturnModel but the runtime object is our fake —
        # static identity check would fail, so compare by id().
        assert id(s._hybrid_return) == id(fake_leaf)
        assert "return_model" in s._pretrained_leaves

    def test_train_does_not_refit_pretrained_leaf(
        self, train_df: pd.DataFrame, fake_leaf: _FakeHybridLeaf
    ) -> None:
        s = ReturnForecastStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"return_model": fake_leaf},
        )
        s.train(train_df)
        assert fake_leaf.fit_calls == 0
        assert s._fitted is True
        assert s.training_metadata is not None
        assert s.training_metadata.train_end == pd.Timestamp(train_df.index[-1])

    def test_get_all_training_metadata_marks_leaf_entries_pretrained(
        self, train_df: pd.DataFrame, fake_leaf: _FakeHybridLeaf
    ) -> None:
        s = ReturnForecastStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"return_model": fake_leaf},
        )
        s.train(train_df)
        tracked = s.get_all_training_metadata()
        # First entry is the strategy's own — not pretrained.
        assert tracked[0].origin == "strategy"
        assert tracked[0].is_pretrained is False
        # Subsequent entries come from the leaf — all marked pretrained.
        leaf_entries = tracked[1:]
        assert len(leaf_entries) > 0
        assert all(t.is_pretrained for t in leaf_entries)

    def test_interval_mismatch_rejected_at_ctor(self, fake_leaf: _FakeHybridLeaf) -> None:
        fake_leaf.training_metadata = TrainingMetadata(
            train_start=_TRAIN_START,
            train_end=_TRAIN_END,
            n_train_samples=_LEAF_N_TRAIN_SAMPLES,
            fit_timestamp=_LEAF_FIT_TIMESTAMP,
            interval=Interval.HOUR,
            feature_columns=_FEATURES,
        )
        with pytest.raises(ValueError, match="interval mismatch"):
            ReturnForecastStrategy(
                feature_columns=list(_FEATURES),
                lstm_lookback=_LSTM_LOOKBACK,
                pretrained_leaves={"return_model": fake_leaf},
            )


class TestVolatilityTargetingPretrainedInjection:
    def test_ctor_stores_leaf_and_skips_build(self, fake_leaf: _FakeHybridLeaf) -> None:
        s = VolatilityTargetingStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"vol_model": fake_leaf},
        )
        assert id(s._hybrid_vol) == id(fake_leaf)

    def test_train_does_not_refit_pretrained_leaf(
        self, train_df: pd.DataFrame, fake_leaf: _FakeHybridLeaf
    ) -> None:
        s = VolatilityTargetingStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"vol_model": fake_leaf},
        )
        s.train(train_df)
        assert fake_leaf.fit_calls == 0
        assert s._fitted is True

    def test_get_all_training_metadata_marks_leaf_entries_pretrained(
        self, train_df: pd.DataFrame, fake_leaf: _FakeHybridLeaf
    ) -> None:
        s = VolatilityTargetingStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"vol_model": fake_leaf},
        )
        s.train(train_df)
        tracked = s.get_all_training_metadata()
        leaf_entries = tracked[1:]
        assert len(leaf_entries) > 0
        assert all(t.is_pretrained for t in leaf_entries)


class TestMomentumGatekeeperPretrainedInjection:
    def test_ctor_stores_leaf_and_skips_build(self, fake_classifier: _FakeClassifier) -> None:
        s = MomentumGatekeeperStrategy(
            feature_columns=list(_MOMENTUM_FEATURES),
            pretrained_leaves={"directional_classifier": fake_classifier},
        )
        # Typed as DirectionalClassifier but the runtime object is our fake —
        # compare by id().
        assert id(s._classifier) == id(fake_classifier)
        assert "directional_classifier" in s._pretrained_leaves

    def test_ctor_auto_adopts_feature_columns_from_leaf(
        self, fake_classifier: _FakeClassifier
    ) -> None:
        """Default ``feature_columns=None`` + injected leaf → ctor copies
        the leaf's ``training_metadata.feature_columns`` so the user isn't
        forced to duplicate the list in both configs."""
        s = MomentumGatekeeperStrategy(
            pretrained_leaves={"directional_classifier": fake_classifier},
        )
        assert s._resolved_feature_columns == list(_MOMENTUM_FEATURES)

    def test_train_does_not_refit_pretrained_leaf(
        self, train_df: pd.DataFrame, fake_classifier: _FakeClassifier
    ) -> None:
        s = MomentumGatekeeperStrategy(
            feature_columns=list(_MOMENTUM_FEATURES),
            pretrained_leaves={"directional_classifier": fake_classifier},
        )
        s.train(train_df)
        assert fake_classifier.fit_calls == 0
        assert s._fitted is True
        assert s.training_metadata is not None
        assert s.training_metadata.train_end == pd.Timestamp(train_df.index[-1])

    def test_get_all_training_metadata_marks_leaf_entries_pretrained(
        self, train_df: pd.DataFrame, fake_classifier: _FakeClassifier
    ) -> None:
        s = MomentumGatekeeperStrategy(
            feature_columns=list(_MOMENTUM_FEATURES),
            pretrained_leaves={"directional_classifier": fake_classifier},
        )
        s.train(train_df)
        tracked = s.get_all_training_metadata()
        assert tracked[0].origin == "strategy"
        assert tracked[0].is_pretrained is False
        leaf_entries = tracked[1:]
        assert len(leaf_entries) == 1
        assert leaf_entries[0].origin == "classifier"
        assert leaf_entries[0].is_pretrained is True

    def test_interval_mismatch_rejected_at_ctor(self, fake_classifier: _FakeClassifier) -> None:
        fake_classifier.training_metadata = _leaf_metadata(
            feature_columns=_MOMENTUM_FEATURES, interval=Interval.HOUR
        )
        with pytest.raises(ValueError, match="interval mismatch"):
            MomentumGatekeeperStrategy(
                feature_columns=list(_MOMENTUM_FEATURES),
                pretrained_leaves={"directional_classifier": fake_classifier},
            )


class TestNonMLStrategiesRejectInjection:
    def test_adaptive_bollinger_rejects_any_key(self) -> None:
        with pytest.raises(ValueError, match="owns no ML leaves"):
            AdaptiveBollingerStrategy(pretrained_leaves={"garch": object()})

    def test_pairs_trading_rejects_any_key(self) -> None:
        with pytest.raises(ValueError, match="owns no ML leaves"):
            PairsTradingStrategy(pretrained_leaves={"anything": object()})

    def test_adaptive_bollinger_accepts_empty_dict(self) -> None:
        s = AdaptiveBollingerStrategy(pretrained_leaves={})
        assert s._pretrained_leaves == {}

    def test_pairs_trading_accepts_none(self) -> None:
        s = PairsTradingStrategy(pretrained_leaves=None)
        assert s._pretrained_leaves == {}


class TestFreshLeafIsNotMarkedPretrained:
    """Regression guard: a strategy built WITHOUT pretrained_leaves must
    never report ``is_pretrained=True`` on any leaf entry. Otherwise the
    walk-forward strict-overlap check would trip against the fold's own
    train window (which the fresh leaf just legitimately trained on)."""

    def test_return_forecast_fresh_leaves_not_pretrained(self, train_df: pd.DataFrame) -> None:
        s = ReturnForecastStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            arma_p_max=1,
            arma_q_max=1,
            lstm_hidden_dim=4,
            lstm_num_layers=1,
            lstm_epochs=1,
            lstm_batch_size=8,
        )
        s.train(train_df)
        tracked = s.get_all_training_metadata()
        assert all(not t.is_pretrained for t in tracked)

    def test_momentum_gatekeeper_fresh_leaves_not_pretrained(self, train_df: pd.DataFrame) -> None:
        s = MomentumGatekeeperStrategy(
            n_estimators=5,
            max_depth=2,
        )
        s.train(train_df)
        tracked = s.get_all_training_metadata()
        assert all(not t.is_pretrained for t in tracked)
