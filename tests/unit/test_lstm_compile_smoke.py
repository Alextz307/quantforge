"""Smoke tests for LSTM ``torch.compile`` + opt-in AMP integration.

The existing ``test_lstm.py`` suite covers full fit/predict semantics; it
runs against the *compiled* model on every fit (compile is unconditional
on the training path), so a passing existing suite already proves
``torch.compile`` doesn't break correctness. These tests add the new
contract surface that's specific to Batch 8.0f:

* ``amp=False`` is the default — thesis runs preserve FP32 numerics
  unless explicitly opted in.
* The ``amp`` flag round-trips through save/load.
* A fit + predict cycle with the compile path active produces finite
  predictions on a tiny synthetic frame (regression guard against a
  future torch upgrade where compile fails on this model shape).

No CUDA-only AMP test ships here. AMP only meaningfully quantises on
CUDA, and the runner doesn't carry one — exercising it would either
silently no-op (giving false confidence) or skip (adding no signal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm import LSTMPredictor

# Mirror the compact-fixture conventions from ``test_lstm.py``.
SYNTH_ROWS = 80
SYNTH_START = "2020-01-02"
SYNTH_BASE = 100.0
SYNTH_RETURN_MEAN = 0.0003
SYNTH_RETURN_STD = 0.01
SYNTH_FEATURE_NOISE = 0.01
SYNTH_SEED = 42
TORCH_SEED = 42

COMPACT_HIDDEN = 16
COMPACT_LAYERS = 1
COMPACT_LOOKBACK = 10
SHORT_EPOCHS = 3


@pytest.fixture
def lstm_df() -> pd.DataFrame:
    np.random.seed(SYNTH_SEED)
    idx = pd.bdate_range(start=SYNTH_START, periods=SYNTH_ROWS, freq="B")
    close = SYNTH_BASE * np.cumprod(
        1 + np.random.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, SYNTH_ROWS)
    )
    return pd.DataFrame(
        {
            "close": close,
            "return_1d": np.random.normal(0.0, SYNTH_FEATURE_NOISE, SYNTH_ROWS),
        },
        index=idx,
    )


@pytest.fixture
def lstm_target(lstm_df: pd.DataFrame) -> pd.Series:
    returns = lstm_df["close"].pct_change().shift(-1)
    return returns.iloc[:-1]


def _build(amp: bool = False) -> LSTMPredictor:
    return LSTMPredictor(
        feature_columns=["close", "return_1d"],
        hidden_dim=COMPACT_HIDDEN,
        num_layers=COMPACT_LAYERS,
        lookback=COMPACT_LOOKBACK,
        epochs=SHORT_EPOCHS,
        amp=amp,
    )


def test_amp_default_is_false() -> None:
    """No-arg ctor → ``amp`` is False; thesis runs land on FP32 by default."""
    p = LSTMPredictor(feature_columns=["close"])
    assert p._amp is False


def test_amp_kwarg_round_trips_via_save_load(
    tmp_path: Path, lstm_df: pd.DataFrame, lstm_target: pd.Series
) -> None:
    """A predictor saved with ``amp=True`` reloads with the same flag set,
    so a downstream ``update()`` or fresh fit honours the persisted choice
    rather than silently dropping back to FP32.
    """
    torch.manual_seed(TORCH_SEED)
    p = _build(amp=True)
    p.fit(lstm_df.iloc[:-1], lstm_target)

    save_dir = tmp_path / "lstm_amp_on"
    p.save(save_dir)
    loaded = LSTMPredictor.load(save_dir)
    assert loaded._amp is True

    # And the False path round-trips cleanly too.
    q = _build(amp=False)
    q.fit(lstm_df.iloc[:-1], lstm_target)
    save_dir_off = tmp_path / "lstm_amp_off"
    q.save(save_dir_off)
    loaded_off = LSTMPredictor.load(save_dir_off)
    assert loaded_off._amp is False


def test_compile_path_produces_finite_predictions(
    lstm_df: pd.DataFrame, lstm_target: pd.Series
) -> None:
    """Fit (which exercises ``torch.compile`` on the training module) +
    predict on the same frame must yield finite values for every
    non-warmup row. A future torch upgrade where compile fails on this
    model shape would surface as NaNs / inf here, not as a silent failure.
    """
    torch.manual_seed(TORCH_SEED)
    p = _build(amp=False)
    p.fit(lstm_df.iloc[:-1], lstm_target)
    preds = p.predict(lstm_df)

    non_warmup = preds.iloc[COMPACT_LOOKBACK:]
    assert non_warmup.notna().all()
    assert np.isfinite(non_warmup.to_numpy(dtype=np.float64)).all()
