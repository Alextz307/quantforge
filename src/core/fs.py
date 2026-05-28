"""
Generic filesystem helpers.

Intentionally free of domain imports (no pandas, numpy, sklearn, torch,
TrainingMetadata, etc.) so this module is reusable anywhere a file I/O
primitive is needed without dragging the ML stack along. Pairs with
``src.core.json_io`` (generic JSON) and stays out of ``persistence.py``
(domain-specific model/run layout) per the module-cohesion convention.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Create ``path``'s parent directory tree if missing and return ``Path(path)``.

    Idempotent (no-op when the parent already exists). Used by every file-
    writer that can't assume the target's parent was pre-created — benchmark
    plots, experiment reports, JSON manifests, LaTeX tables.

    Returns the resolved ``Path`` for chaining:

        p = ensure_parent_dir(out_dir / "plots" / "foo.png")
        fig.savefig(p)
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def atomic_write_path(target: str | Path) -> Iterator[Path]:
    """
    Yield a tmp ``Path`` to write to; ``os.replace`` it onto ``target`` on clean exit.

    Stages writes to ``<stem>.tmp.<pid>.<tid><suffix>`` next to ``target``
    so concurrent writers from the same process tree don't collide on the
    staging file. On clean exit, ``os.replace(tmp, target)`` makes the new
    content visible atomically (POSIX guarantee; Windows ≥3.3). On
    exception, the tmp file is removed best-effort and the exception
    re-raises so callers see the original failure.

    Used by callers that need crash-safe replacement of a single file
    (study state JSON, parquet cache writes). The caller is responsible
    for whatever serialization touches the yielded path:

        with atomic_write_path(target) as tmp:
            df.to_parquet(tmp)

    ``BaseException`` is caught (not just ``Exception``) so cleanup also
    fires on ``KeyboardInterrupt`` during a long write — important for
    multi-day HPO runs where the user may Ctrl+C mid-trial.
    """

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(
        f"{target_path.stem}.tmp.{os.getpid()}.{threading.get_ident()}{target_path.suffix}"
    )
    try:
        yield tmp
        os.replace(tmp, target_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> Path:
    """
    Atomically write ``text`` as UTF-8 to ``path``.

    Convenience wrapper around :func:`atomic_write_path` for the common
    plain-text-payload case (LaTeX tables, README fragments).
    """

    with atomic_write_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return Path(path)
