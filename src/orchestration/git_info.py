"""Best-effort git-sha reader shared by experiment + standalone-model runs.

Lives in a dedicated module rather than inside either ``experiment.py`` or
``standalone_training.py`` so both callers converge on one implementation
— drift between "experiment manifest SHA format" and "model artifact
manifest SHA format" would be a silent reproducibility bug.
"""

from __future__ import annotations

import subprocess

_GIT_SHA_UNKNOWN = "unknown"
_GIT_SHORT_LENGTH = 7
_GIT_SUBPROCESS_TIMEOUT_S = 2


def read_git_sha() -> str:
    """Short git SHA for the current HEAD, or ``"unknown"`` if unavailable.

    "Unavailable" covers: not in a git tree, git not on PATH, subprocess
    timeout, or a corrupt repo. Callers persist the returned string
    verbatim — consumers that want to detect the unknown case compare to
    ``"unknown"`` or to a known non-empty SHA.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", f"--short={_GIT_SHORT_LENGTH}", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_SUBPROCESS_TIMEOUT_S,
        )
        return out.stdout.strip() or _GIT_SHA_UNKNOWN
    except (subprocess.SubprocessError, FileNotFoundError):
        return _GIT_SHA_UNKNOWN
