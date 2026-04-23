"""Generic filesystem helpers.

Intentionally free of domain imports (no pandas, numpy, sklearn, torch,
TrainingMetadata, etc.) so this module is reusable anywhere a file I/O
primitive is needed without dragging the ML stack along. Pairs with
``src.core.json_io`` (generic JSON) and stays out of ``persistence.py``
(domain-specific model/run layout) per the module-cohesion convention.
"""

from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """Create ``path``'s parent directory tree if missing and return ``Path(path)``.

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
