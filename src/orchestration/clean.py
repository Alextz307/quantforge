"""
Tidy up the experiment-results store before a fresh study run.

The CLI command ``experiment clean`` wraps :func:`plan_clean` (dry-run)
and :func:`apply_clean` (wipe) so the destructive step is explicit and
the same logic is exercised in tests without going through click.

What gets wiped:

* The *contents* of every immediate child *directory* under
  ``<store_root>/`` that is NOT explicitly listed in ``--keep``. The
  directory itself is left in place (empty) so consumers that assume
  the canonical store layout exists (``<root>/hpo/``, ``<root>/runs/``,
  ...) keep finding it after a wipe.
* Any directory containing a git-tracked file is refused with a clear
  error pointing the user at ``git rm`` or ``--keep``. The check goes
  through ``git ls-files`` so .gitignored ephemera can be removed
  freely while a directory the user committed by accident gets a
  loud failure rather than a silent wipe.

What is NOT touched:

* Files at the top level of ``<store_root>/`` (e.g., a stray README).
  Only directory contents are candidates.
* Directories whose names are listed in ``--keep``.
* Anything outside ``<store_root>/``: parent path traversal is
  defensively blocked.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.core.logging import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class CleanCandidate:
    """
    One directory the cleaner would (or would not) delete.

    ``size_bytes`` is the recursive on-disk size - used to print a tally
    in the dry-run output so the user can spot oversized leftover study
    runs at a glance. ``tracked_files`` is the list of git-tracked paths
    found under this directory; non-empty means the dir is refused.
    """

    path: Path
    size_bytes: int
    tracked_files: tuple[Path, ...]

    @property
    def is_safe_to_delete(self) -> bool:
        return not self.tracked_files


@dataclass(frozen=True)
class CleanPlan:
    """
    Outcome of :func:`plan_clean` - every candidate plus the kept set.
    """

    store_root: Path
    candidates: tuple[CleanCandidate, ...]
    preserved: tuple[str, ...]

    @property
    def deletable(self) -> tuple[CleanCandidate, ...]:
        return tuple(c for c in self.candidates if c.is_safe_to_delete)

    @property
    def refused(self) -> tuple[CleanCandidate, ...]:
        return tuple(c for c in self.candidates if not c.is_safe_to_delete)


def plan_clean(
    store_root: Path, *, keep: Iterable[str] = (), repo_root: Path | None = None
) -> CleanPlan:
    """
    Walk ``store_root`` and classify every immediate-child directory.

    ``keep`` is the set of child-directory names to preserve. ``repo_root``
    is the directory used as the cwd for ``git ls-files``; defaults to
    ``store_root``'s parent (so a typical ``experiment_results/`` under
    a repo root works without an explicit override).
    """

    if not store_root.is_dir():
        return CleanPlan(
            store_root=store_root,
            candidates=(),
            preserved=tuple(sorted(set(keep))),
        )

    preserved = set(keep)
    git_root = repo_root if repo_root is not None else store_root.parent
    tracked_by_child = _git_tracked_by_child(store_root, git_root=git_root)

    candidates: list[CleanCandidate] = []
    for child in sorted(store_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in preserved:
            continue
        candidates.append(
            CleanCandidate(
                path=child,
                size_bytes=_dir_size_bytes(child),
                tracked_files=tracked_by_child.get(child.name, ()),
            )
        )
    return CleanPlan(
        store_root=store_root,
        candidates=tuple(candidates),
        preserved=tuple(sorted(preserved)),
    )


def apply_clean(plan: CleanPlan) -> tuple[Path, ...]:
    """
    Wipe the contents of every safe candidate in ``plan`` and return the wiped paths.

    The candidate directories themselves are kept as empty placeholders
    so the canonical store layout (``runs/``, ``hpo/``, ``studies/``, ...)
    survives a wipe.

    Refuses to apply if any candidate has tracked files: the caller is
    expected to surface :attr:`CleanPlan.refused` to the user and let
    them re-run with ``--keep <name>`` after they've inspected the
    refused entries.
    """

    refused = plan.refused
    if refused:
        names = ", ".join(c.path.name for c in refused)
        raise ValueError(
            f"refusing to apply clean: directories contain git-tracked files: "
            f"{names}. Either `git rm` the tracked files first, or pass "
            f"`--keep <name>` for each to exclude them from the plan."
        )
    wiped: list[Path] = []
    for candidate in plan.deletable:
        _logger.info("wiping %s (%.1f MB)", candidate.path, candidate.size_bytes / (1024 * 1024))
        _empty_directory(candidate.path)
        wiped.append(candidate.path)
    return tuple(wiped)


def _empty_directory(directory: Path) -> None:
    """
    Remove every entry inside ``directory`` while keeping ``directory`` itself.
    """

    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def format_plan(plan: CleanPlan) -> str:
    """
    Human-readable rendering of a :class:`CleanPlan` for the dry-run output.
    """

    if not plan.candidates:
        return (
            f"experiment clean: nothing to wipe under {plan.store_root} "
            f"(preserved: {', '.join(plan.preserved) or '<none>'})"
        )

    lines = [f"experiment clean: candidates under {plan.store_root}"]
    wipeable_total_bytes = 0
    for c in plan.candidates:
        size_mb = c.size_bytes / (1024 * 1024)
        if c.is_safe_to_delete:
            lines.append(f"  WIPE    {c.path.name:40s}  {size_mb:8.1f} MB")
            wipeable_total_bytes += c.size_bytes
        else:
            tracked_count = len(c.tracked_files)
            lines.append(
                f"  REFUSE  {c.path.name:40s}  {size_mb:8.1f} MB  "
                f"({tracked_count} tracked file{'s' if tracked_count != 1 else ''})"
            )

    lines.append(f"preserved: {', '.join(plan.preserved) or '<none>'}")
    lines.append(
        f"would free {wipeable_total_bytes / (1024 * 1024):.1f} MB across "
        f"{len(plan.deletable)} directory(ies); pass --apply to wipe."
    )
    return "\n".join(lines)


def _git_tracked_by_child(store_root: Path, *, git_root: Path) -> dict[str, tuple[Path, ...]]:
    """
    Map every immediate-child directory name to its git-tracked files.

    Returns an empty dict when ``git_root`` isn't a git repository or
    ``store_root`` lives outside it - both cases mean "safe to delete".
    Implemented as a single ``git ls-files`` call against ``store_root``
    so a dirty results store with N child dirs costs one subprocess
    instead of N.
    """

    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", str(store_root)],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}
    if result.returncode != 0:
        return {}
    by_child: dict[str, list[Path]] = {}
    store_root_abs = store_root.resolve()
    for raw in result.stdout.split("\0"):
        if not raw:
            continue
        absolute = (git_root / raw).resolve()
        try:
            relative = absolute.relative_to(store_root_abs)
        except ValueError:
            continue
        if not relative.parts:
            continue
        by_child.setdefault(relative.parts[0], []).append(Path(raw))
    return {name: tuple(paths) for name, paths in by_child.items()}


def _dir_size_bytes(directory: Path) -> int:
    """
    Recursive sum of file sizes under ``directory`` (best-effort).

    Symlinks aren't followed (``rglob`` already obeys this); broken
    symlinks and permission errors are skipped silently - a wrong size
    in the dry-run output is preferable to crashing the planner.
    """

    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


__all__ = [
    "CleanCandidate",
    "CleanPlan",
    "apply_clean",
    "format_plan",
    "plan_clean",
]
