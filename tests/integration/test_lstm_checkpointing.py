"""
Tests for LSTMPredictor best-state checkpointing.

Two invariants:

1. With ``checkpoint_path`` set, ``fit()`` writes ``BEST_STATE_PT`` whenever
   validation loss improves; after a clean fit the file exists and its
   tensors match the model's restored best-state.
2. A mid-fit interrupt leaves the best-so-far snapshot recoverable via
   ``torch.load`` — the file is intact even though the in-memory model is
   half-mutated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from src.core.persistence import BEST_STATE_PT
from src.models.lstm import LSTMPredictor
from tests.conftest import (
    attach_synthetic_features,
    make_synthetic_close_df,
    seed_globally,
)

COMPACT_HIDDEN_DIM = 16
COMPACT_NUM_LAYERS = 1
COMPACT_LOOKBACK = 10
SHORT_EPOCHS = 5
LONGER_EPOCHS = 10
HIGH_PATIENCE = 50  # disables early stopping for the duration of these tests
SAVE_CALLS_BEFORE_INTERRUPT = 2


@pytest.fixture
def df() -> pd.DataFrame:
    seed_globally()
    base = make_synthetic_close_df()
    return attach_synthetic_features(base, ["return_1d"])


@pytest.fixture
def target(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change().shift(-1)
    return returns.iloc[:-1]


@pytest.fixture
def features() -> list[str]:
    return ["close", "return_1d"]


def _build_predictor(features: list[str], epochs: int) -> LSTMPredictor:
    return LSTMPredictor(
        features,
        hidden_dim=COMPACT_HIDDEN_DIM,
        num_layers=COMPACT_NUM_LAYERS,
        lookback=COMPACT_LOOKBACK,
        epochs=epochs,
        patience=HIGH_PATIENCE,
    )


def test_checkpoint_path_creates_best_state_file(
    tmp_path: Path, df: pd.DataFrame, target: pd.Series, features: list[str]
) -> None:
    seed_globally()
    train = df.iloc[:-1]
    p = _build_predictor(features, epochs=SHORT_EPOCHS)
    ckpt_dir = tmp_path / "lstm_ckpt"

    p.fit(train, target, checkpoint_path=ckpt_dir)

    assert (ckpt_dir / BEST_STATE_PT).exists()


def test_checkpoint_state_matches_restored_best_after_clean_fit(
    tmp_path: Path, df: pd.DataFrame, target: pd.Series, features: list[str]
) -> None:
    seed_globally()
    train = df.iloc[:-1]
    p = _build_predictor(features, epochs=SHORT_EPOCHS)
    ckpt_dir = tmp_path / "lstm_ckpt"

    p.fit(train, target, checkpoint_path=ckpt_dir)
    saved = torch.load(ckpt_dir / BEST_STATE_PT, weights_only=True)
    assert p._model is not None
    current = p._model.state_dict()

    assert set(saved.keys()) == set(current.keys())
    for key, current_tensor in current.items():
        torch.testing.assert_close(saved[key], current_tensor.cpu())


def test_checkpoint_survives_mid_fit_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    df: pd.DataFrame,
    target: pd.Series,
    features: list[str],
) -> None:
    """
    Force ``torch.save`` to raise on the third call.

    The first two best-state writes land on disk. The third call raises
    ``KeyboardInterrupt`` from inside ``fit()``, simulating a Ctrl+C between
    epochs. After unwinding the exception we verify the on-disk checkpoint
    is loadable and shape-compatible with the model — i.e. the most recent
    successful write was atomic from the consumer's point of view.
    """

    seed_globally()
    train = df.iloc[:-1]
    p = _build_predictor(features, epochs=LONGER_EPOCHS)
    ckpt_dir = tmp_path / "lstm_ckpt"

    real_save = torch.save
    save_count = {"n": 0}

    # ``torch.save`` is overloaded across many destination types (Path, IO,
    # callable writer); this passthrough mirrors the upstream signature.
    def raising_save(*args: Any, **kwargs: Any) -> None:
        save_count["n"] += 1
        if save_count["n"] > SAVE_CALLS_BEFORE_INTERRUPT:
            raise KeyboardInterrupt("simulated mid-fit interrupt")
        real_save(*args, **kwargs)

    monkeypatch.setattr("src.models.lstm.torch.save", raising_save)

    with pytest.raises(KeyboardInterrupt):
        p.fit(train, target, checkpoint_path=ckpt_dir)

    ckpt_file = ckpt_dir / BEST_STATE_PT
    assert ckpt_file.exists()
    saved = torch.load(ckpt_file, weights_only=True)
    # Build a fresh predictor + train one epoch to materialise a state_dict
    # we can compare keys against — interrupted ``p`` has a model object
    # whose state_dict is partially mutated.
    fresh = _build_predictor(features, epochs=1)
    seed_globally()
    fresh.fit(train, target)
    assert fresh._model is not None
    assert set(saved.keys()) == set(fresh._model.state_dict().keys())
