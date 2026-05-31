"""
Reconstruct in-memory experiment artifacts from a persisted run directory.

Readers anchored on the canonical persistence layout
(``<run_dir>/manifest.json`` + ``fold_results.jsonl`` + ``config.yaml`` +
``strategy_state/``):

* :func:`load_experiment_result` - used by ``experiment compare
  --reuse-runs`` so the comparison reporter can rank + bootstrap-test
  prior runs without retraining the underlying experiments.
* :func:`load_experiment_config_from_run` - reads the frozen
  ``config.yaml`` back into a typed :class:`ExperimentConfig`.
* :func:`load_strategy_from_run_dir` - reconstructs the trained
  :class:`IStrategy` instance by resolving its registered class via
  ``strategy_registry`` and dispatching to ``cls.load(strategy_state/)``.
  Used by the deployment layer to predict from a previously trained run.
* :func:`resolve_run_dir` - resolve a run dir by id, matching both the
  flat ``<store_root>/runs/<id>`` and study-nested
  ``<store_root>/studies/<x>/runs/<id>`` layouts.
"""

from __future__ import annotations

from pathlib import Path

import src.strategies  # noqa: F401 - fires @strategy_registry.register decorators
from src.core import json_io
from src.core.config import ExperimentConfig, load_experiment_config
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_STRATEGY_SUBDIR,
    FOLD_RESULTS_JSONL,
    RUNS_SUBDIR,
)
from src.core.registry import strategy_registry
from src.orchestration.manifest import Manifest
from src.orchestration.types import ExperimentResult, FoldRecord
from src.strategies.interface import IStrategy


def load_experiment_result(run_dir: Path) -> ExperimentResult:
    """
    Reconstruct an :class:`ExperimentResult` from a persisted run directory.

    Reads ``manifest.json`` + ``fold_results.jsonl`` only; the frozen
    ``config.yaml`` is intentionally NOT required here so the comparison
    reuse path doesn't pay the YAML+pydantic cost when it only needs
    fold-level data.

    Raises:
        FileNotFoundError: ``run_dir`` is not a directory, or either of
            the required artifacts is missing.
    """

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"experiment run directory not found: {run_dir}; "
            f"check the --reuse-runs path resolves to a previous run."
        )
    manifest_path = run_dir / EXPERIMENT_MANIFEST_JSON
    folds_path = run_dir / FOLD_RESULTS_JSONL
    for required in (manifest_path, folds_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"missing artifact {required.name} under {run_dir}; the "
                f"source run may be incomplete - re-run the experiment "
                f"or pass a different path."
            )
    manifest = Manifest.from_dict(json_io.read_dict(manifest_path))
    folds = tuple(FoldRecord.from_dict(d) for d in json_io.read_jsonl(folds_path))
    return ExperimentResult(
        experiment_id=manifest.experiment_id,
        folds=folds,
        manifest=manifest,
    )


def load_experiment_config_from_run(run_dir: Path) -> ExperimentConfig:
    """
    Read the frozen ``config.yaml`` from a persisted run directory.
    """

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"experiment run directory not found: {run_dir}; "
            f"check the --reuse-runs path resolves to a previous run."
        )
    config_path = run_dir / EXPERIMENT_CONFIG_YAML
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing {EXPERIMENT_CONFIG_YAML} under {run_dir}; the run "
            f"may be incomplete - re-run the experiment or pass a "
            f"different path."
        )
    return load_experiment_config(config_path)


def resolve_run_dir(store_root: Path, experiment_id: str) -> Path:
    """
    Resolve a run's directory under ``store_root`` by experiment id.

    Returns the flat ``store_root / runs / <experiment_id>`` when it exists.
    Runs produced inside a study live at
    ``store_root / studies / <study> / runs / <experiment_id>``; when the flat
    path is absent, fall back to a recursive search so a study-internal run
    still resolves from the top-level store root (the layout a deployment
    points at). The flat path is returned unchanged when nothing matches, so
    callers raise their own pointed error against a concrete path.
    """

    flat = store_root / RUNS_SUBDIR / experiment_id
    if flat.is_dir():
        return flat
    for candidate in store_root.glob(f"**/{RUNS_SUBDIR}/{experiment_id}"):
        if candidate.is_dir():
            return candidate
    return flat


def strategy_supports_feature_importance(strategy_name: str) -> bool:
    """
    Whether a registered strategy can produce feature importance.

    A strategy supports importance iff it overrides ``feature_columns`` to
    declare engineered columns (the rule-based strategies inherit the base
    empty-tuple method and the importance subsystem skips them). Derived from
    the registered class so it can't drift from the strategies themselves;
    returns ``False`` for an unregistered name rather than raising, since both
    callers (the read endpoint's ``computable`` flag and the job handler's
    pre-launch guard) treat an unknown strategy as "nothing to compute".
    """

    try:
        cls = strategy_registry.get(strategy_name)
    except KeyError:
        return False
    return cls.feature_columns is not IStrategy.feature_columns


def load_strategy_from_run_dir(run_dir: Path) -> IStrategy:
    """
    Reconstruct the trained strategy persisted under a run directory.

    Reads ``config.yaml`` to discover the strategy's registered name
    (``strategy.name``), resolves the concrete class via the project-wide
    :data:`~src.core.registry.strategy_registry`, and dispatches to
    ``cls.load(run_dir / strategy_state)``. The returned instance is
    fully trained: ``generate_signals()`` works immediately and
    ``training_metadata`` is populated from the saved ``metadata.json``.

    The registry is the single source of truth for ``YAML name -> class``
    - no per-strategy switch lives here. Adding a new strategy file
    under ``src/strategies/`` makes it loadable through this entry
    point automatically.

    Raises:
        FileNotFoundError: ``run_dir`` is missing, its ``config.yaml`` is
            missing, or its ``strategy_state/`` directory is missing.
        KeyError: ``config.yaml`` names a strategy that is not registered.
    """

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"experiment run directory not found: {run_dir}; "
            f"fix by passing the path to a completed run."
        )
    state_dir = run_dir / EXPERIMENT_STRATEGY_SUBDIR
    if not state_dir.is_dir():
        raise FileNotFoundError(
            f"strategy state directory not found: {state_dir}; the source "
            f"run may be incomplete - re-run the experiment or pass a "
            f"different path."
        )
    cfg = load_experiment_config_from_run(run_dir)
    cls = strategy_registry.get(cfg.strategy.name)
    return cls.load(state_dir)
