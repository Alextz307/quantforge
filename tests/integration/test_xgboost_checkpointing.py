"""Tests for DirectionalClassifier best-iteration checkpointing.

Two invariants:

1. With ``checkpoint_path`` set, ``fit()`` writes ``BEST_ITERATION_UBJ``
   whenever the **validation** metric improves; after a clean fit the file
   exists and reloads into a usable XGBoost booster.
2. A mid-fit interrupt leaves the best-so-far snapshot recoverable — the
   booster on disk is loadable even though the wrapping ``XGBClassifier``
   never finished ``fit()``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import xgboost as xgb

from src.core.persistence import BEST_ITERATION_UBJ
from src.core.utils import next_bar_direction
from src.models.xgboost_classifier import DirectionalClassifier
from tests.conftest import make_synthetic_close_df, seed_globally

# 200 bars × 30 estimators leaves room for several val-metric improvements.
COMPACT_N_ESTIMATORS = 30
SAVES_BEFORE_INTERRUPT = 1
# Matches the realised-vol window used by VolatilityTargetingStrategy.
VOL_WINDOW = 20


@pytest.fixture
def xgb_data() -> tuple[pd.DataFrame, pd.Series]:
    """Features + binary direction target with weak but nonzero signal.

    Pure-noise features collapse the val-improvement count to ~1 (only
    round 0 beats the ``None`` baseline), which would make the
    interrupt-survives test flaky once checkpoint saves are gated on val
    improvement only. Computing the features from the close series itself
    plus a one-bar lagged direction provides enough autocorrelation that
    val log-loss improves several times across 30 boosting rounds.
    """

    seed_globally()
    base = make_synthetic_close_df()
    close = base["close"]
    target = next_bar_direction(close)
    return_1d = close.pct_change()
    momentum_5 = close.pct_change(5)
    rolling_vol = return_1d.rolling(VOL_WINDOW).std()
    # shift(1) gives yesterday's direction — weakly autocorrelated with today's.
    direction_lag1 = target.shift(1).reindex(base.index).astype(float)
    features = pd.DataFrame(
        {
            "return_1d": return_1d,
            "momentum_5": momentum_5,
            "rolling_vol": rolling_vol,
            "direction_lag1": direction_lag1,
        }
    ).dropna()
    return features, target.reindex(features.index)


@pytest.fixture
def xgb_features() -> list[str]:
    return ["return_1d", "momentum_5", "rolling_vol", "direction_lag1"]


def test_checkpoint_path_creates_best_iteration_file(
    tmp_path: Path,
    xgb_data: tuple[pd.DataFrame, pd.Series],
    xgb_features: list[str],
) -> None:
    features, target = xgb_data
    c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
    ckpt_dir = tmp_path / "xgb_ckpt"

    c.fit(features, target, checkpoint_path=ckpt_dir)

    assert (ckpt_dir / BEST_ITERATION_UBJ).exists()


def test_checkpoint_reloads_into_usable_booster(
    tmp_path: Path,
    xgb_data: tuple[pd.DataFrame, pd.Series],
    xgb_features: list[str],
) -> None:
    features, target = xgb_data
    c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
    ckpt_dir = tmp_path / "xgb_ckpt"

    c.fit(features, target, checkpoint_path=ckpt_dir)
    booster = xgb.Booster()
    booster.load_model(str(ckpt_dir / BEST_ITERATION_UBJ))

    dmat = xgb.DMatrix(features)
    preds = booster.predict(dmat)
    assert preds.shape == (len(features),)


def test_checkpoint_survives_mid_fit_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    xgb_data: tuple[pd.DataFrame, pd.Series],
    xgb_features: list[str],
) -> None:
    """Force the second improvement-save to raise.

    The first save lands on disk; the second raises ``KeyboardInterrupt``
    from inside the XGBoost training callback, simulating a Ctrl+C between
    rounds. The on-disk booster from the first improvement should still be
    loadable.
    """

    features, target = xgb_data
    c = DirectionalClassifier(xgb_features, n_estimators=COMPACT_N_ESTIMATORS)
    ckpt_dir = tmp_path / "xgb_ckpt"
    ckpt_file = ckpt_dir / BEST_ITERATION_UBJ

    real_save_model = xgb.Booster.save_model
    save_count = {"n": 0}

    def raising_save_model(self: xgb.Booster, fname: str) -> None:
        save_count["n"] += 1
        if save_count["n"] > SAVES_BEFORE_INTERRUPT:
            raise KeyboardInterrupt("simulated mid-fit interrupt")
        real_save_model(self, fname)

    monkeypatch.setattr(xgb.Booster, "save_model", raising_save_model)

    with pytest.raises(KeyboardInterrupt):
        c.fit(features, target, checkpoint_path=ckpt_dir)

    assert ckpt_file.exists()
    booster = xgb.Booster()
    booster.load_model(str(ckpt_file))
    dmat = xgb.DMatrix(features)
    preds = booster.predict(dmat)
    assert preds.shape == (len(features),)
