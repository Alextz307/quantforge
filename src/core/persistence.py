"""Shared save/load helpers for model persistence.

No pickle, no joblib. Every persisted artifact is JSON (metadata + configs +
small numeric weights) or the model's own native binary format (``.pt`` for
torch, ``.ubj`` for XGBoost).

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

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Canonical filenames used by every model's save() / load(). Keeping them
# together avoids silent drift when a format change touches multiple models.
CONFIG_JSON = "config.json"
WEIGHTS_JSON = "weights.json"
METADATA_JSON = "metadata.json"
SCALER_JSON = "scaler.json"
WEIGHTS_PT = "weights.pt"
MODEL_UBJ = "model.ubj"
ENDOG_NPY = "endog.npy"


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


def write_json(path: str | Path, obj: object) -> None:
    """Write ``obj`` as UTF-8 JSON at ``path`` with sorted keys and 2-space indent.

    Accepts ``object`` rather than a narrow union to match ``json.dump``'s own
    duck-typed contract — callers pass arbitrarily-nested dict/list/scalar
    payloads and invariance on ``dict[str, X]`` would otherwise force casts at
    every call site.
    """
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> object:
    """Load JSON from ``path``. Caller narrows the return type via ``isinstance``."""
    parsed: object = json.loads(Path(path).read_text(encoding="utf-8"))
    return parsed


# --- Typed JSON field accessors --------------------------------------------
# Narrowing the values out of ``read_json_dict(...)`` is ceremonial (``int(str
# (raw[key]))``) when repeated at every load site. These helpers centralize the
# "read then narrow" pattern with uniform error messages, so a load() body
# reads as a flat list of field extractions rather than nested casts.


def read_json_dict(path: str | Path) -> dict[str, object]:
    """Load JSON from ``path`` and require the top level to be an object."""
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"JSON at {path} must be an object, got {type(raw).__name__}")
    return raw


def json_get_int(d: dict[str, object], key: str) -> int:
    """Pull ``key`` out of ``d`` and narrow to ``int`` with a named error."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, int):
        # bool is an int subclass — reject to avoid ``True``/``False`` leaking in
        raise ValueError(f"JSON field {key!r} must be an int, got {type(value).__name__}")
    return value


def json_get_float(d: dict[str, object], key: str) -> float:
    """Pull ``key`` out of ``d`` and narrow to ``float`` (accepting ``int``)."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"JSON field {key!r} must be a number, got {type(value).__name__}")
    return float(value)


def json_get_str(d: dict[str, object], key: str) -> str:
    """Pull ``key`` out of ``d`` and require a ``str``."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if not isinstance(value, str):
        raise ValueError(f"JSON field {key!r} must be a string, got {type(value).__name__}")
    return value


def _json_get_list(d: dict[str, object], key: str) -> list[object]:
    """Module-private: pull ``key`` and require a ``list``. Callers should use
    a typed variant (``json_get_int_list``, ``json_get_float_list``,
    ``json_get_str_list``) — untyped element access leaves mypy unhappy."""
    if key not in d:
        raise KeyError(f"missing required JSON field {key!r}")
    value = d[key]
    if not isinstance(value, list):
        raise ValueError(f"JSON field {key!r} must be a list, got {type(value).__name__}")
    return value


def json_get_float_list(d: dict[str, object], key: str) -> list[float]:
    """Pull ``key`` out of ``d`` and require a list of numbers."""
    raw = _json_get_list(d, key)
    out: list[float] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"JSON field {key!r}[{i}] must be a number, got {type(item).__name__}")
        out.append(float(item))
    return out


def json_get_int_list(d: dict[str, object], key: str) -> list[int]:
    """Pull ``key`` out of ``d`` and require a list of integers."""
    raw = _json_get_list(d, key)
    out: list[int] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"JSON field {key!r}[{i}] must be an int, got {type(item).__name__}")
        out.append(item)
    return out


def json_get_str_list(d: dict[str, object], key: str) -> list[str]:
    """Pull ``key`` out of ``d`` and require a list of strings."""
    raw = _json_get_list(d, key)
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"JSON field {key!r}[{i}] must be a string, got {type(item).__name__}")
    return [str(item) for item in raw]


# --- Scaler round-trip ------------------------------------------------------


def save_standard_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Serialize a fitted ``StandardScaler`` to JSON.

    Captures the public fitted attributes (``mean_``, ``scale_``, ``var_``,
    ``n_features_in_``, ``n_samples_seen_``). Sklearn's private ``__getstate__``
    surface has drifted across versions — manual attribute capture is safer.

    Raises ``RuntimeError`` if the scaler hasn't been fitted.
    """
    if not hasattr(scaler, "mean_"):
        raise RuntimeError("cannot save an unfitted StandardScaler")
    payload = {
        "mean_": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scale_": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "var_": np.asarray(scaler.var_, dtype=np.float64).tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
    }
    write_json(path, payload)


def load_standard_scaler(path: str | Path) -> StandardScaler:
    """Reconstruct a ``StandardScaler`` from the JSON emitted by ``save_standard_scaler``.

    The loaded scaler is marked fitted — ``transform()`` works immediately,
    ``fit()`` would re-enter sklearn's normal flow.
    """
    raw = read_json_dict(path)
    with_mean = raw["with_mean"]
    with_std = raw["with_std"]
    if not isinstance(with_mean, bool) or not isinstance(with_std, bool):
        raise ValueError("scaler JSON fields with_mean/with_std must be booleans")
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    scaler.mean_ = np.asarray(json_get_float_list(raw, "mean_"), dtype=np.float64)
    scaler.scale_ = np.asarray(json_get_float_list(raw, "scale_"), dtype=np.float64)
    scaler.var_ = np.asarray(json_get_float_list(raw, "var_"), dtype=np.float64)
    scaler.n_features_in_ = json_get_int(raw, "n_features_in_")
    scaler.n_samples_seen_ = json_get_int(raw, "n_samples_seen_")
    return scaler
