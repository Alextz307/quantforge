"""End-to-end driver for ``experiment regime``.

Loads a persisted experiment run, instantiates the user-supplied
detector, splits its folds, aggregates per regime, and assembles a
:class:`RegimeReport`. The CLI layer in ``scripts/experiment.py`` is a
thin click wrapper around :func:`run_regime_report`.

Why a separate module
---------------------
Mirrors the ``run_comparison`` / ``train_model_standalone`` split: the
CLI does YAML loading + click-friendly error wrapping, the orchestration
layer does the actual coordination work and is unit-testable without
spinning up the click runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.analysis.regime_split import aggregate_split, split_folds_by_regime
from src.core import json_io
from src.core.config import ExperimentConfig, load_experiment_config
from src.core.logging import get_logger
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    FOLD_RESULTS_JSONL,
    REGIME_REPORTS_SUBDIR,
    RUNS_SUBDIR,
)
from src.core.regime_config import RegimeConfig
from src.core.registry import data_source_registry
from src.data.fingerprint import fingerprint_bars
from src.data.interface import IDataSource
from src.orchestration.git_info import read_git_sha
from src.orchestration.manifest import Manifest
from src.orchestration.regime import regime_registry
from src.orchestration.types import (
    MIXED_REGIME_LABEL,
    FoldRecord,
    RegimeReport,
    RegimeSlice,
)

_logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedRun:
    """A minimal view of a persisted run on disk.

    Just enough to drive a regime split: the experiment config (so we
    can re-fetch bars), the manifest (for ``data_hash`` cross-check and
    ``experiment_id``), and the fold records.
    """

    experiment_id: str
    config: ExperimentConfig
    manifest: Manifest
    folds: tuple[FoldRecord, ...]


def load_run_from_disk(run_dir: Path) -> LoadedRun:
    """Read ``config.yaml`` + ``manifest.json`` + ``fold_results.jsonl``.

    Raises :class:`FileNotFoundError` with a pointed message if any of
    the three files is missing — partial run dirs (e.g. mid-crash) are
    not silently treated as analysable.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"experiment run directory not found: {run_dir}; check the --exp-id "
            f"value matches a directory under experiment_results/runs/."
        )

    config_path = run_dir / EXPERIMENT_CONFIG_YAML
    manifest_path = run_dir / EXPERIMENT_MANIFEST_JSON
    folds_path = run_dir / FOLD_RESULTS_JSONL
    for required in (config_path, manifest_path, folds_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"missing artifact {required.name} under {run_dir}; run may be "
                f"incomplete — re-run the experiment or pass a different --exp-id."
            )

    config = load_experiment_config(config_path)
    manifest = Manifest.from_dict(json_io.read_dict(manifest_path))
    folds = _read_fold_jsonl(folds_path)
    return LoadedRun(
        experiment_id=manifest.experiment_id,
        config=config,
        manifest=manifest,
        folds=folds,
    )


def run_regime_report(
    *,
    run_dir: Path,
    regime_cfg: RegimeConfig,
    out_name: str,
    store_root: Path,
) -> tuple[RegimeReport, Path]:
    """Drive the full regime-analysis pipeline for one persisted run.

    Pipeline:
    1. Load the run's config / manifest / fold records from disk.
    2. Re-fetch bars via the run's saved data source — needed for the
       detector's bar-level tagging.
    3. Sanity-check the re-fetched ``data_hash`` against the manifest;
       a mismatch means data-vendor drift since the run was saved and
       any regime split would be unfaithful to the original walk.
    4. Build the detector via ``regime_registry``, tag bars, split folds.
    5. Aggregate per regime, assemble :class:`RegimeReport`, persist
       under ``<store_root>/regime_reports/<out_name>/``.

    Returns ``(report, out_dir)`` so the CLI can echo the artifact path
    without recomputing it.
    """
    run = load_run_from_disk(run_dir)
    data_source: IDataSource = data_source_registry.create_from_config(run.config.data.source)

    if len(run.config.data.tickers) != 1:
        raise ValueError(
            f"regime analysis currently supports single-ticker runs only, got "
            f"tickers={run.config.data.tickers}; fix by re-running the analysis "
            f"on a single-ticker experiment."
        )
    bars = data_source.fetch(
        run.config.data.tickers[0],
        run.config.data.start,
        run.config.data.end,
        run.config.data.interval,
    )

    refetched_hash = fingerprint_bars(bars)
    if refetched_hash != run.manifest.data_hash:
        raise ValueError(
            f"data_hash drift detected: manifest recorded "
            f"{run.manifest.data_hash[:12]}..., re-fetched "
            f"{refetched_hash[:12]}...; a regime split on drifted data "
            f"would not match the original walk-forward windows. Fix by "
            f"using the same data source / cache as the original run, "
            f"or re-run the experiment so the manifest reflects the new bars."
        )

    detector = regime_registry.create_from_config(regime_cfg.detector)
    split = split_folds_by_regime(run.folds, detector, bars)
    per_regime_stats = aggregate_split(split)

    per_regime_fold_indices: dict[str, tuple[int, ...]] = {
        label: tuple(f.fold_index for f in folds) for label, folds in split.per_regime.items()
    }
    if split.mixed:
        per_regime_fold_indices[MIXED_REGIME_LABEL] = tuple(f.fold_index for f in split.mixed)

    slices: tuple[RegimeSlice, ...] = tuple(detector.slices(bars))
    report = RegimeReport(
        out_name=out_name,
        experiment_id=run.experiment_id,
        kind=detector.kind,
        detector_name=regime_cfg.detector.name,
        created_at=datetime.now(UTC),
        git_sha=read_git_sha(),
        per_regime_stats=per_regime_stats,
        per_regime_fold_indices=per_regime_fold_indices,
        mixed_fold_indices=tuple(f.fold_index for f in split.mixed),
        slices=slices,
    )

    out_dir = store_root / REGIME_REPORTS_SUBDIR / out_name
    _logger.info(
        "regime split: %d regime(s) + %d mixed fold(s) for experiment %s",
        len(per_regime_fold_indices) - (1 if split.mixed else 0),
        len(split.mixed),
        run.experiment_id,
    )
    return report, out_dir


def _read_fold_jsonl(path: Path) -> tuple[FoldRecord, ...]:
    """Read ``fold_results.jsonl`` (one JSON object per line) into records."""
    records: list[FoldRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(FoldRecord.from_dict(json.loads(stripped)))
    return tuple(records)


def resolve_run_dir(store_root: Path, experiment_id: str) -> Path:
    """Resolve ``store_root / runs / <experiment_id>``.

    A trivial helper, but the CLI calls it twice (the resolved path is
    echoed and used) — and centralising the join here means a future
    runs-layout change touches one function.
    """
    return store_root / RUNS_SUBDIR / experiment_id


__all__ = [
    "LoadedRun",
    "load_run_from_disk",
    "resolve_run_dir",
    "run_regime_report",
]
