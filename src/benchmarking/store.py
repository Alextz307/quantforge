"""JSONL-backed persistence for ``BenchmarkRun``.

Layout on disk:

    benchmark_results/
        runs/<timestamp>_<sha7>.jsonl        # ephemeral; gitignored
        baselines/<name>.jsonl               # committed; anchors regression reports

Each file holds a single ``BenchmarkRun`` as one JSON object on one line.
The ``.jsonl`` extension preserves the option to append further runs
without a schema change, but callers today should treat each file as one
run.

The store never mutates an existing file. Overwriting a baseline requires
passing ``overwrite=True``; otherwise the write raises ``FileExistsError``
so accidental re-runs do not clobber the thesis anchor.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.benchmarking.types import BenchmarkRun
from src.core import json_io

RUNS_DIR = "runs"
BASELINES_DIR = "baselines"


class BenchmarkStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._runs = self._root / RUNS_DIR
        self._baselines = self._root / BASELINES_DIR
        self._runs.mkdir(parents=True, exist_ok=True)
        self._baselines.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def save_run(self, run: BenchmarkRun) -> Path:
        path = self._runs / f"{run.run_id}.jsonl"
        json_io.write(path, run.to_dict())
        return path

    def load_run(self, path: Path) -> BenchmarkRun:
        return BenchmarkRun.from_dict(json_io.read_dict(path))

    def save_baseline(self, run: BenchmarkRun, name: str, *, overwrite: bool = False) -> Path:
        path = self._baselines / f"{name}.jsonl"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"baseline {name!r} already exists at {path}; pass overwrite=True to replace"
            )
        json_io.write(path, run.to_dict())
        return path

    def load_baseline(self, name: str) -> BenchmarkRun:
        path = self._baselines / f"{name}.jsonl"
        if not path.exists():
            available = sorted(p.stem for p in self._baselines.glob("*.jsonl"))
            raise FileNotFoundError(f"baseline {name!r} not found; available: {available}")
        return BenchmarkRun.from_dict(json_io.read_dict(path))

    def list_baselines(self) -> tuple[str, ...]:
        return tuple(sorted(p.stem for p in self._baselines.glob("*.jsonl")))

    def list_runs(self) -> tuple[Path, ...]:
        return tuple(sorted(self._runs.glob("*.jsonl")))

    def load_runs(
        self,
        *,
        since: str | None = None,
        tags: Iterable[str] | None = None,
        commit: str | None = None,
    ) -> tuple[BenchmarkRun, ...]:
        """Load runs matching all provided filters.

        ``since`` is an ISO-timestamp prefix comparison — filenames sort
        correctly by timestamp, so no datetime parsing needed. ``tags``
        requires the run to carry every listed tag. ``commit`` matches the
        run's hardware git_sha prefix (allowing short/long SHAs).
        """
        tag_set = set(tags) if tags is not None else set()
        # Run filenames encode the ISO timestamp with colons replaced by dashes
        # (`<safe_ts>_<sha7>.jsonl`). For the `since` filter we can compare the
        # filename stem prefix and skip the JSON parse entirely.
        since_prefix = since.replace(":", "-") if since is not None else None
        out: list[BenchmarkRun] = []
        for path in self.list_runs():
            if since_prefix is not None and path.stem < since_prefix:
                continue
            run = BenchmarkRun.from_dict(json_io.read_dict(path))
            if tag_set and not tag_set.issubset(run.tags):
                continue
            if commit is not None and not run.hardware.git_sha.startswith(commit):
                continue
            out.append(run)
        return tuple(out)
