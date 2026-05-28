"""
Model-persistence layout + skeletons + sklearn-scaler round-trip.

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

from collections.abc import Callable, Iterable
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.core import json_io
from src.core.fs import atomic_write_path

if TYPE_CHECKING:
    from src.core.temporal import TrainingMetadata
    from src.orchestration.manifest import Manifest

CONFIG_JSON = "config.json"
WEIGHTS_JSON = "weights.json"
METADATA_JSON = "metadata.json"
SCALER_JSON = "scaler.json"
WEIGHTS_PT = "weights.pt"
MODEL_UBJ = "model.ubj"
ENDOG_NPY = "endog.npy"
# Mid-fit checkpoints written so a Ctrl+C between epochs / boosting rounds
# leaves best-so-far weights recoverable. Distinct filenames from the
# canonical save outputs so a checkpoint dir and a save dir can sit
# side-by-side without collision.
BEST_STATE_PT = "best_state.pt"
BEST_ITERATION_UBJ = "best_iteration.ubj"

GARCH_SUBDIR = "garch"
ARMA_SUBDIR = "arma"
LSTM_SUBDIR = "lstm"
CLASSIFIER_SUBDIR = "classifier"
HYBRID_VOL_SUBDIR = "hybrid_vol"
HYBRID_RETURN_SUBDIR = "hybrid_return"
PIPELINE_SCALER_JSON = "pipeline_scaler.json"

EXPERIMENT_MANIFEST_JSON = "manifest.json"
FOLD_RESULTS_JSONL = "fold_results.jsonl"
EXPERIMENT_CONFIG_YAML = "config.yaml"
EXPERIMENT_METRICS_JSON = "metrics.json"
EXPERIMENT_STRATEGY_SUBDIR = "strategy_state"
EXPERIMENT_CHECKPOINTS_SUBDIR = "checkpoints"
EXPERIMENT_RUN_LOG = "run.log"
FOLD_DIR_PREFIX = "fold_"

RUNS_SUBDIR = "runs"
HPO_SUBDIR = "hpo"
HPO_TRIALS_RUNS_SUBDIR = "runs"
COMPARISONS_SUBDIR = "comparisons"
HOLDOUT_EVALS_SUBDIR = "holdout_evals"
CLI_LOGS_SUBDIR = "cli_logs"

# Live deployment layout — each deployment is a directory under
# ``<store_root>/deployments/<deployment_id>/`` holding the typed manifest
# (round-trippable provenance) and an append-only signal log.
DEPLOYMENTS_SUBDIR = "deployments"
DEPLOYMENT_MANIFEST_JSON = "manifest.json"
DEPLOYMENT_SIGNALS_JSONL = "signals.jsonl"

# Holdout-eval bundles deliberately skip the typed ``Manifest`` — only
# commands that CREATE an experiment write one (run / tune). Holdout-eval
# REFERENCES the source run's manifest, and its own provenance lives in
# this payload (source id, kind, boundary, data hash, slippage, metrics,
# and ``is_holdout_eval: true`` so automated tooling can't confuse this
# artifact for a normal run).
HOLDOUT_EVAL_JSON = "holdout_eval.json"
DSR_JSON_FILENAME = "dsr.json"

# Marker file written as the FINAL step of every model/strategy save(); its
# presence is the load-time invariant that all sibling files (config.json,
# weights.*, metadata.json, scaler.json, sub-leaf dirs) are on disk. The
# marker is dot-prefixed so it sorts at the top of ``ls -la`` and does not
# collide with any real artifact name.
SAVE_COMPLETE_MARKER = ".save_complete"


def ensure_model_dir(path: str | Path) -> Path:
    """
    Create ``path`` as an empty directory and return the Path.

    Raises ``FileExistsError`` if ``path`` exists and is non-empty — prevents
    silent overwrite of an existing save. If ``path`` exists and is empty it is
    reused as-is.
    """

    p = Path(path)
    # Create atomically; on collision disambiguate. Collapses the old
    # exists()/is_dir()/iterdir()/mkdir() check-then-act into one syscall
    # on the fresh-path fast path and removes the TOCTOU window.
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


def frozen_params_to_json(
    params: object,
    *,
    omit: Iterable[str] = (),
) -> dict[str, object]:
    """
    Convert a frozen ctor-params dataclass to a JSON-safe dict.

    Centralises three conversions that every composite's
    ``_ctor_kwargs_as_json()`` would otherwise reinvent:

    * ``tuple`` → ``list`` (JSON has no tuple; ``feature_columns`` is the
      canonical victim).
    * ``Enum`` / ``StrEnum`` → ``.value`` (so the saved config JSON is human-
      readable and the load path can reconstruct via ``EnumClass(value)``).
    * Fields in ``omit`` are dropped — used for non-persisted preferences
      like ``device`` / ``lstm_device`` that re-resolve on load.

    Intentionally narrow:
    * Does NOT recurse into nested dataclasses (none of our ctor params
      nest; adding recursion later is backwards-compatible).
    * Accepts any dataclass instance by duck-typing (``is_dataclass``);
      raises ``TypeError`` on anything else so misuse surfaces loudly.
    """

    if not is_dataclass(params) or isinstance(params, type):
        raise TypeError(
            f"frozen_params_to_json requires a dataclass INSTANCE, "
            f"got {type(params).__name__}; fix by passing self._params, not the class."
        )
    raw: dict[str, object] = asdict(params)
    for key in omit:
        raw.pop(key, None)
    for key, value in raw.items():
        if isinstance(value, tuple):
            raw[key] = list(value)
        elif isinstance(value, Enum):
            raw[key] = value.value
    return raw


def write_experiment_manifest(path: str | Path, manifest: Manifest) -> None:
    """
    Write ``manifest.json`` under the experiment run directory.

    The directory MUST already exist (the experiment runner creates it as
    part of its own save skeleton). Centralised here so a future manifest
    format change touches one function, not every caller.
    """

    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(
            f"experiment directory {p} does not exist; "
            f"create it via ensure_model_dir() before writing the manifest."
        )
    json_io.write(p / EXPERIMENT_MANIFEST_JSON, manifest.to_dict())


def read_experiment_manifest(path: str | Path) -> Manifest:
    """
    Read ``<path>/manifest.json`` and reconstruct a typed :class:`Manifest`.

    Companion to :func:`write_experiment_manifest`. ``path`` is the run
    directory (not the JSON file itself), mirroring the writer's contract.
    Raises :class:`FileNotFoundError` if the manifest is missing — partial
    run dirs (mid-crash) shouldn't be silently treated as analysable.
    """

    from src.orchestration.manifest import Manifest as _Manifest

    p = Path(path)
    manifest_path = p / EXPERIMENT_MANIFEST_JSON
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}; the source directory "
            f"may be incomplete — re-run the source experiment or pass a "
            f"different path."
        )
    return _Manifest.from_dict(json_io.read_dict(manifest_path))


def mark_save_complete(root: str | Path) -> None:
    """
    Write the ``.save_complete`` marker as the FINAL step of a save().

    Every other artifact in the save dir must already be on disk when this
    is called; presence of the marker is the load-time invariant that the
    dir is consistent.
    """

    with atomic_write_path(Path(root) / SAVE_COMPLETE_MARKER) as tmp:
        tmp.write_text("", encoding="utf-8")


def assert_save_complete(root: str | Path) -> Path:
    """
    Verify ``<root>/.save_complete`` exists; return the resolved ``Path``.

    Called at the top of every model/strategy ``load()`` before any sibling
    file is read. A missing marker means the source save was interrupted
    (SIGKILL, OOM, Ctrl+C between the per-file writes) — loading from such
    a directory would silently return an inconsistent model (e.g. fresh
    config + stale weights). Raise loudly instead.
    """

    p = Path(root)
    marker = p / SAVE_COMPLETE_MARKER
    if not marker.is_file():
        raise FileNotFoundError(
            f"save at {p} is incomplete: missing {SAVE_COMPLETE_MARKER!r} marker. "
            f"The producing save() either crashed mid-write or never finished — "
            f"refit and re-save rather than loading a half-written directory."
        )
    return p


def save_model_skeleton(
    path: str | Path,
    *,
    config: dict[str, object],
    training_metadata: TrainingMetadata,
    write_weights: Callable[[Path], None],
) -> Path:
    """
    Canonical 5-step save-directory skeleton for every model + strategy.

    Steps: ``ensure_model_dir`` → write ``config.json`` → user-provided
    weights write → write ``metadata.json`` → write ``.save_complete`` marker.
    The caller validates preconditions (fitted, internal handles non-None)
    before invoking; the marker (always the last write) is the load-time
    invariant that every prior step landed on disk.
    """

    root = ensure_model_dir(path)
    json_io.write(root / CONFIG_JSON, config)
    write_weights(root)
    json_io.write(root / METADATA_JSON, training_metadata.to_dict())
    mark_save_complete(root)
    return root


def save_standard_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """
    Serialize a fitted ``StandardScaler`` to JSON.

    Captures the public fitted attributes (``mean_``, ``scale_``, ``var_``,
    ``n_features_in_``, ``n_samples_seen_``). Sklearn's private ``__getstate__``
    surface has drifted across versions — manual attribute capture is safer.

    ``feature_names_in_`` is persisted whenever present (scaler was fit on a
    DataFrame) so post-load ``.transform()`` on a named DataFrame doesn't
    trip sklearn's "fit without feature names" warning.

    Raises ``RuntimeError`` if the scaler hasn't been fitted.
    """

    if not hasattr(scaler, "mean_"):
        raise RuntimeError(
            "cannot save an unfitted StandardScaler; fix by calling scaler.fit() "
            "(or scaler.fit_transform()) on training data before save."
        )
    # ``n_samples_seen_`` is a scalar int when fit input has no NaNs, a
    # per-feature ndarray otherwise (the live path: feature pipeline emits
    # leading warmup NaNs). Serialising as a list round-trips both shapes.
    payload: dict[str, object] = {
        "mean_": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scale_": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "var_": np.asarray(scaler.var_, dtype=np.float64).tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": [int(v) for v in np.atleast_1d(scaler.n_samples_seen_)],
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
    }
    if hasattr(scaler, "feature_names_in_"):
        payload["feature_names_in_"] = [str(n) for n in scaler.feature_names_in_]
    json_io.write(path, payload)


def load_standard_scaler(path: str | Path) -> StandardScaler:
    """
    Reconstruct a ``StandardScaler`` from the JSON emitted by ``save_standard_scaler``.

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
    samples_seen = json_io.get_int_list(raw, "n_samples_seen_")
    if len(samples_seen) == 1:
        scaler.n_samples_seen_ = samples_seen[0]
    else:
        scaler.n_samples_seen_ = np.asarray(samples_seen, dtype=np.int64)
    if "feature_names_in_" in raw:
        scaler.feature_names_in_ = np.asarray(
            json_io.get_str_list(raw, "feature_names_in_"), dtype=object
        )
    return scaler
