"""Read-only services for the persisted regime-reports tree."""

from __future__ import annotations

from pathlib import Path

from src.core import json_io
from src.core.persistence import EXPERIMENT_MANIFEST_JSON
from src.orchestration.types import RegimeKind, RegimeSlice
from webapp.backend.app.infrastructure.store import (
    RegimeReportNotFoundError,
    find_regime_report_dir,
    iter_regime_report_dirs,
    store_label,
)
from webapp.backend.app.schemas.regime import (
    PerRegimeStatsRow,
    RegimeReportDetail,
    RegimeReportSummary,
    RegimeSliceDTO,
)
from webapp.backend.app.services.plots import (
    PlotNotFoundError,
    list_plots,
    resolve_plot_path,
)

__all__ = [
    "PlotNotFoundError",
    "RegimeReportNotFoundError",
    "get_regime_report",
    "list_regime_reports",
    "resolve_plot",
]


def list_regime_reports(root: Path) -> list[RegimeReportSummary]:
    """List every regime report under ``root``, newest first."""
    summaries: list[RegimeReportSummary] = []
    for report_dir in iter_regime_report_dirs(root):
        manifest = json_io.read_dict(report_dir / EXPERIMENT_MANIFEST_JSON)
        per_regime = json_io.get_dict(manifest, "per_regime_stats")
        summaries.append(
            RegimeReportSummary(
                name=json_io.get_str(manifest, "out_name"),
                store=store_label(report_dir, root),
                created_at=json_io.get_timestamp(manifest, "created_at"),
                experiment_id=json_io.get_str(manifest, "experiment_id"),
                kind=RegimeKind(json_io.get_str(manifest, "kind")),
                detector_name=json_io.get_str(manifest, "detector_name"),
                regime_labels=sorted(per_regime.keys()),
            )
        )
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_regime_report(root: Path, name: str) -> RegimeReportDetail:
    """Read the full detail payload for one regime report."""
    report_dir = find_regime_report_dir(root, name)
    manifest = json_io.read_dict(report_dir / EXPERIMENT_MANIFEST_JSON)
    per_regime_stats = json_io.get_dict(manifest, "per_regime_stats")
    rows = sorted(
        (
            PerRegimeStatsRow.model_validate(
                {**json_io.get_dict(per_regime_stats, label), "regime_label": label}
            )
            for label in per_regime_stats
        ),
        key=lambda r: r.regime_label,
    )
    fold_indices_raw = json_io.get_dict(manifest, "per_regime_fold_indices")
    fold_indices = {
        label: json_io.get_int_list(fold_indices_raw, label) for label in fold_indices_raw
    }
    slices = [_build_slice(s) for s in json_io.get_list_of_dicts(manifest, "slices")]
    return RegimeReportDetail(
        name=json_io.get_str(manifest, "out_name"),
        store=store_label(report_dir, root),
        created_at=json_io.get_timestamp(manifest, "created_at"),
        git_sha=json_io.get_str(manifest, "git_sha"),
        experiment_id=json_io.get_str(manifest, "experiment_id"),
        kind=RegimeKind(json_io.get_str(manifest, "kind")),
        detector_name=json_io.get_str(manifest, "detector_name"),
        per_regime_stats=rows,
        per_regime_fold_indices=fold_indices,
        mixed_fold_indices=json_io.get_int_list(manifest, "mixed_fold_indices"),
        slices=slices,
        plots=list_plots(report_dir),
    )


def resolve_plot(root: Path, name: str, plot_name: str) -> Path:
    """Resolve a regime-report plot filename to an absolute path, blocking traversal."""
    return resolve_plot_path(find_regime_report_dir(root, name), plot_name)


def _build_slice(d: dict[str, object]) -> RegimeSliceDTO:
    parsed = RegimeSlice.from_dict(d)
    return RegimeSliceDTO(label=parsed.label, start=parsed.start, end=parsed.end)
