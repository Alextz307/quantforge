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
from src.strategies.pairs_trading import PairsTradingStrategy
from src.strategies.return_forecast import ReturnForecastStrategy
from src.strategies.volatility_targeting import VolatilityTargetingStrategy
from tests.conftest import make_synthetic_ohlcv_df

if TYPE_CHECKING:
    pass

_FEATURES: tuple[str, ...] = ("sma_20", "rsi_14", "volume_z")
_INTERVAL = Interval.DAILY
_LSTM_LOOKBACK = 10
_TRAIN_ROWS = 80
_OHLCV_SEED = 7
_LEAF_N_TRAIN_SAMPLES = 250
_TRAIN_START = pd.Timestamp("2019-01-02")
_TRAIN_END = pd.Timestamp("2019-12-31")
_LEAF_FIT_TIMESTAMP = pd.Timestamp("2020-01-05")


def _leaf_metadata() -> TrainingMetadata:
    return TrainingMetadata(
        train_start=_TRAIN_START,
        train_end=_TRAIN_END,
        n_train_samples=_LEAF_N_TRAIN_SAMPLES,
        fit_timestamp=_LEAF_FIT_TIMESTAMP,
        interval=_INTERVAL,
        feature_columns=_FEATURES,
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
    update_calls: int = 0
    training_metadata_entries: tuple[TrackedMetadata, ...] = field(
        default_factory=lambda: (
            TrackedMetadata(origin="arma", metadata=_leaf_metadata()),
            TrackedMetadata(origin="lstm", metadata=_leaf_metadata()),
        )
    )

    def fit(self, df: pd.DataFrame, target: pd.Series, **_: object) -> None:
        self.fit_calls += 1

    def update(self, df: pd.DataFrame, target: pd.Series, **_: object) -> None:
        self.update_calls += 1

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0] * len(df), index=df.index)

    def get_all_training_metadata(self) -> tuple[TrackedMetadata, ...]:
        return self.training_metadata_entries


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

    def test_update_does_not_refit_pretrained_leaf(
        self, train_df: pd.DataFrame, fake_leaf: _FakeHybridLeaf
    ) -> None:
        """Frozen-leaf contract: ``strategy.update()`` must NOT call
        ``leaf.update()`` when the leaf was pretrained-injected. Otherwise
        a forward-run loop silently mutates GARCH/ARMA/LSTM params and
        advances the leaf's own ``training_metadata`` fold by fold.
        """
        s = ReturnForecastStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"return_model": fake_leaf},
        )
        s.train(train_df)
        assert fake_leaf.update_calls == 0
        # Disjoint update window strictly after train_end (extend() rejects overlap)
        update_df = _ohlcv_df()
        update_df.index = pd.date_range(
            train_df.index[-1] + pd.Timedelta(days=1),
            periods=len(update_df),
            freq="B",
        )
        s.update(update_df)
        assert fake_leaf.update_calls == 0
        assert s.training_metadata is not None
        assert s.training_metadata.train_end == pd.Timestamp(update_df.index[-1])


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

    def test_update_does_not_refit_pretrained_leaf(
        self, train_df: pd.DataFrame, fake_leaf: _FakeHybridLeaf
    ) -> None:
        """Frozen-leaf contract: ``strategy.update()`` must NOT call
        ``leaf.update()`` when the leaf was pretrained-injected.
        """
        s = VolatilityTargetingStrategy(
            feature_columns=list(_FEATURES),
            lstm_lookback=_LSTM_LOOKBACK,
            pretrained_leaves={"vol_model": fake_leaf},
        )
        s.train(train_df)
        assert fake_leaf.update_calls == 0
        update_df = _ohlcv_df()
        update_df.index = pd.date_range(
            train_df.index[-1] + pd.Timedelta(days=1),
            periods=len(update_df),
            freq="B",
        )
        s.update(update_df)
        assert fake_leaf.update_calls == 0
        assert s.training_metadata is not None
        assert s.training_metadata.train_end == pd.Timestamp(update_df.index[-1])


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
