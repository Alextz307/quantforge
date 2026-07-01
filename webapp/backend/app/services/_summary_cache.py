"""
Mtime-invalidated + thread-pooled artifact summarization for the list services.

The paginated list endpoints re-derive a one-line summary for every visible
artifact on every request (each page/sort/filter click is a fresh round-trip).
Without caching that reparses the whole tree each time; this helper reuses a
summary while its source files' mtimes are unchanged and overlaps the remaining
blocking reads across a small pool - the same pattern every list service shares.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# Per-artifact summarization is dominated by blocking file reads (manifest /
# state file / trials feed), which release the GIL, so a small pool overlaps
# them. Past ~4 the JSON/YAML parsing (CPU-bound, GIL-contended) stops scaling.
_LIST_WORKER_COUNT = 4


def cached_summaries[T](
    dirs: Iterable[Path],
    *,
    mtime_sources: Sequence[str],
    summarize: Callable[[Path], T],
    cache: dict[str, tuple[int, T | None]],
) -> list[T]:
    """
    Summarize each dir once per artifact revision, across a small thread pool.

    ``mtime_sources`` are the filenames within each dir whose mtimes together
    define the artifact's revision - the cache invalidates when the MAX mtime
    across them advances. Passing more than one covers summaries derived from a
    companion file that lands *after* the primary write: an HPO study's
    ``best_config.yaml`` is written after its final ``trials.jsonl`` append, and
    a run's ``metrics.json`` after its ``manifest.json``. Keying on the primary
    file alone would pin the pre-companion summary until the next primary write,
    which for a finished artifact never comes.

    Robust to two races against the TTL-cached directory walk that feeds
    ``dirs``: a dir whose sources have all vanished (deleted or atomically
    swapped in the window between walk and stat) is skipped, and a dir whose
    ``summarize`` raises (a truncated/half-written artifact) is skipped too - one
    bad artifact must not 500 the whole listing. Both are logged. A skip caches
    ``None`` keyed on the current revision so the failure isn't reparsed until
    the artifact is rewritten. The returned list drops every skip, so callers
    receive only ``T`` values, in ``dirs`` order (they sort downstream).

    Cache keys absent from ``dirs`` this pass are pruned, so a deleted artifact
    doesn't leak its entry forever.
    """

    dir_list = list(dirs)
    live_keys = {str(d) for d in dir_list}
    for stale in [key for key in cache if key not in live_keys]:
        del cache[stale]

    def revision(artifact_dir: Path) -> int | None:
        mtimes: list[int] = []
        for name in mtime_sources:
            try:
                mtimes.append((artifact_dir / name).stat().st_mtime_ns)
            except OSError:
                continue
        return max(mtimes) if mtimes else None

    def summarize_cached(artifact_dir: Path) -> T | None:
        key = str(artifact_dir)
        rev = revision(artifact_dir)
        if rev is None:
            logger.warning("skipping artifact with no readable source at %s", artifact_dir)
            return None
        hit = cache.get(key)
        if hit is not None and hit[0] == rev:
            return hit[1]
        try:
            summary: T | None = summarize(artifact_dir)
        except Exception as exc:  # noqa: BLE001 - one bad artifact must not 500 the listing
            logger.warning("skipping unreadable artifact at %s: %s", artifact_dir, exc)
            summary = None
        cache[key] = (rev, summary)
        return summary

    with ThreadPoolExecutor(max_workers=_LIST_WORKER_COUNT) as pool:
        return [s for s in pool.map(summarize_cached, dir_list) if s is not None]
