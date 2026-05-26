"""TTL-cached artifact-directory listings shared by the read services.

Recursive `**/<subdir>/*/<manifest>` globs over the experiment-results tree
are the dominant cost on warm list-endpoint calls. Each service kind (runs,
holdouts, comparisons, hpo, studies) has its own glob pattern, so the cache
is keyed by ``(root, kind)``. A short TTL keeps successive pagination /
sort / filter requests instant without making freshly written artifacts
invisible for long.

Each cache entry carries both the path tuple and a ``{dir.name: dir}``
id-index so callers needing O(1) name→path lookup (e.g. ``find_run_dir``
on the run-detail hot path) share the same snapshot as the listing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

_TTL_SECONDS = 5.0
_CACHE: dict[tuple[str, str], tuple[float, tuple[Path, ...], dict[str, Path]]] = {}


def cached_artifact_dirs(
    root: Path, kind: str, walker: Callable[[Path], Iterator[Path]]
) -> tuple[Path, ...]:
    """Return the artifact directories under ``root`` for ``kind``, TTL-cached."""
    paths, _ = _get_or_refresh(root, kind, walker)
    return paths


def cached_artifact_index(
    root: Path, kind: str, walker: Callable[[Path], Iterator[Path]]
) -> tuple[tuple[Path, ...], dict[str, Path]]:
    """Return paths plus a ``{dir.name: dir}`` index, TTL-cached together."""
    return _get_or_refresh(root, kind, walker)


def warm_index(root: Path, kind: str, name: str, path: Path) -> None:
    """Insert ``name → path`` into the cached id-index without bumping the TTL.

    Used when a single artifact is resolved out-of-band (glob fallback for
    items written after the last snapshot) so successive lookups in the
    same TTL window skip the glob.
    """
    hit = _CACHE.get((str(root), kind))
    if hit is not None:
        hit[2][name] = path


def _get_or_refresh(
    root: Path, kind: str, walker: Callable[[Path], Iterator[Path]]
) -> tuple[tuple[Path, ...], dict[str, Path]]:
    key = (str(root), kind)
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and now - hit[0] < _TTL_SECONDS:
        return hit[1], hit[2]
    paths = tuple(walker(root))
    id_index = {p.name: p for p in paths}
    _CACHE[key] = (now, paths, id_index)
    return paths, id_index


def clear() -> None:
    """Drop every cached entry. Test fixture hook."""
    _CACHE.clear()
