"""Model-persistence layout + skeletons + sklearn-scaler round-trip.

No pickle, no joblib. Every persisted artifact is JSON (metadata + configs +
small numeric weights) or the model's own native binary format (``.pt`` for
torch, ``.ubj`` for XGBoost). Generic JSON read/write + typed field extraction
live in ``src.core.json_io``; this module only knows how our models arrange
themselves on disk.

The canonical directory layout under a model's save path:

    <path>/
      metadata.json      TrainingMetadata.to_dict()
      config.json        ctor hyperparams
      weights.json       GARCH/ARMA: params as plain JSON
      weights.pt         LSTM: torch state_dict
      model.ubj          XGBoost: native save_model
      scaler.json        StandardScaler: mean_, scale_, var_, n_features_in_
      <subdir>/          composite: nested leaf model dirs
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.core import json_io

if TYPE_CHECKING:
    from src.core.temporal import TrainingMetadata

# Canonical filenames used by every model's save() / load(). Keeping them
# together avoids silent drift when a format change touches multiple models.
CONFIG_JSON = "config.json"
WEIGHTS_JSON = "weights.json"
METADATA_JSON = "metadata.json"
SCALER_JSON = "scaler.json"
WEIGHTS_PT = "weights.pt"
MODEL_UBJ = "model.ubj"
ENDOG_NPY = "endog.npy"

# Canonical subdirectory names for composite save layouts. Centralized so a
# strategy that persists a GARCH subdir and the GARCHPredictor it delegates to
# can't disagree on the string. Adding a new leaf type? Add its subdir here.
GARCH_SUBDIR = "garch"
ARMA_SUBDIR = "arma"
LSTM_SUBDIR = "lstm"
CLASSIFIER_SUBDIR = "classifier"
HYBRID_VOL_SUBDIR = "hybrid_vol"
HYBRID_RETURN_SUBDIR = "hybrid_return"
# Momentum strategy's feature-pipeline scaler sits at the strategy root.
PIPELINE_SCALER_JSON = "pipeline_scaler.json"


def ensure_model_dir(path: str | Path) -> Path:
    """Create ``path`` as an empty directory and return the Path.

    Raises ``FileExistsError`` if ``path`` exists and is non-empty — prevents
    silent overwrite of an existing save. If ``path`` exists and is empty it is
    reused as-is.
    """
    p = Path(path)
    # Try to create atomically. If ``path`` already exists, ``mkdir`` raises
    # ``FileExistsError`` and we disambiguate (empty dir = reuse, non-empty =
    # re-raise, file-at-path = NotADirectoryError). This collapses the old
    # ``exists()/is_dir()/iterdir()/mkdir()`` check-then-act into one syscall
    # for the fresh-path fast path and removes the TOCTOU window.
    try:
        p.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not p.is_dir():
            raise NotADirectoryError(f"save path {p} exists and is not a directory") from None
        if any(p.iterdir()):
            raise FileExistsError(
                f"save path {p} already exists and is non-empty; choose a fresh path"
            ) from None
    return p


def save_model_skeleton(
    path: str | Path,
    *,
    config: dict[str, object],
    training_metadata: TrainingMetadata,
    write_weights: Callable[[Path], None],
) -> Path:
    """Canonical 4-step save-directory skeleton for every model + strategy.

    Steps: ``ensure_model_dir`` → write ``config.json`` → user-provided
    weights write → write ``metadata.json``. The caller validates preconditions
    (fitted, internal handles non-None) before invoking.
    """
    root = ensure_model_dir(path)
    json_io.write(root / CONFIG_JSON, config)
    write_weights(root)
    json_io.write(root / METADATA_JSON, training_metadata.to_dict())
    return root


# --- Scaler round-trip ------------------------------------------------------


def save_standard_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Serialize a fitted ``StandardScaler`` to JSON.

    Captures the public fitted attributes (``mean_``, ``scale_``, ``var_``,
    ``n_features_in_``, ``n_samples_seen_``). Sklearn's private ``__getstate__``
    surface has drifted across versions — manual attribute capture is safer.

    ``feature_names_in_`` is persisted whenever present (scaler was fit on a
    DataFrame) so post-load ``.transform()`` on a named DataFrame doesn't
    trip sklearn's "fit without feature names" warning.

    Raises ``RuntimeError`` if the scaler hasn't been fitted.
    """
    if not hasattr(scaler, "mean_"):
        raise RuntimeError("cannot save an unfitted StandardScaler")
    payload: dict[str, object] = {
        "mean_": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scale_": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "var_": np.asarray(scaler.var_, dtype=np.float64).tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
    }
    if hasattr(scaler, "feature_names_in_"):
        payload["feature_names_in_"] = [str(n) for n in scaler.feature_names_in_]
    json_io.write(path, payload)


def load_standard_scaler(path: str | Path) -> StandardScaler:
    """Reconstruct a ``StandardScaler`` from the JSON emitted by ``save_standard_scaler``.

    The loaded scaler is marked fitted — ``transform()`` works immediately,
    ``fit()`` would re-enter sklearn's normal flow. ``feature_names_in_`` is
    restored as the ``dtype=object`` numpy array sklearn expects when
    present in the payload.
    """
    raw = json_io.read_dict(path)
    scaler = StandardScaler(
        with_mean=json_io.get_bool(raw, "with_mean"),
        with_std=json_io.get_bool(raw, "with_std"),
    )
    scaler.mean_ = np.asarray(json_io.get_float_list(raw, "mean_"), dtype=np.float64)
    scaler.scale_ = np.asarray(json_io.get_float_list(raw, "scale_"), dtype=np.float64)
    scaler.var_ = np.asarray(json_io.get_float_list(raw, "var_"), dtype=np.float64)
    scaler.n_features_in_ = json_io.get_int(raw, "n_features_in_")
    scaler.n_samples_seen_ = json_io.get_int(raw, "n_samples_seen_")
    if "feature_names_in_" in raw:
        scaler.feature_names_in_ = np.asarray(
            json_io.get_str_list(raw, "feature_names_in_"), dtype=object
        )
    return scaler
