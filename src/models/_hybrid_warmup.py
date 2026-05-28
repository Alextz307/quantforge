"""
Shared warmup-row drop for hybrid models.

The standard feature pipeline emits engineered columns (RSI, MACD, etc.)
with multi-bar warmup; the hybrid leaf's residual target may have a much
shorter warmup, so reindexing features against residuals leaves NaN rows
exposed. ``StandardScaler.transform`` propagates NaN under sklearn 1.x,
which corrupts the LSTM's first forward pass with non-finite gradients.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def drop_feature_warmup(
    feature_frame: pd.DataFrame,
    residuals: pd.Series,
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    valid_mask = feature_frame.notna().all(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped == 0:
        return feature_frame, residuals
    logger.info(
        "%s: dropped %d feature-NaN warmup rows before LSTM (kept %d)",
        label,
        n_dropped,
        int(valid_mask.sum()),
    )
    return feature_frame.loc[valid_mask], residuals.loc[valid_mask]
