"""Typed manifest for a persisted experiment run.

Every field below is the answer to a question a post-run consumer
(holdout-eval, HPO resume) MUST answer to run safely. A
typed dataclass over a ``dict[str, object]`` catches typos like
``holdoutStart`` vs ``holdout_start`` at static-check time rather than
after hours of HPO compute.

Rationale for each field:

* ``experiment_id``    — opaque dir name consumers index by.
* ``name``             — human-readable label lifted from the config.
* ``created_at``       — UTC timestamp of the run start.
* ``git_sha``          — short SHA for reproducibility; best-effort
                         (``"unknown"`` if the run happens outside git).
* ``seed``             — int seeded into numpy / torch / random at run
                         start; required to reproduce walk-forward output.
* ``data_hash``        — ``fingerprint_bars(df)`` output; catches vendor
                         drift between runs.
* ``holdout_start``    — absolute pinned boundary timestamp (ISO string in
                         JSON). ``None`` when no holdout was reserved.
* ``slippage_scenario``— the ``SlippageScenario`` enum value used, so
                         downstream consumers know which friction model
                         produced the equity curve.

``to_dict`` / ``from_dict`` mirror the conventions used elsewhere
(timestamps → ISO strings; ``None`` holdout → ``null``). A typo in the
``holdout_start`` key would previously fail at runtime after potentially
hours of HPO compute — with a frozen dataclass, mypy + pytest catch it at
static-check time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.core import json_io
from src.engine.scenarios import SlippageScenario


@dataclass(frozen=True)
class PretrainedLeafRecord:
    """Per-leaf provenance entry on :class:`Manifest`.

    Enough to reproduce holdout-eval from the manifest alone: ``path`` to
    the artifact, ``data_hash`` the artifact trained on (cross-check
    against the artifact's own manifest for drift), and the full training
    window so downstream checks can validate temporal separation and
    re-fingerprint the experiment's bars over the same range to refute
    cross-universe contamination — without reloading the artifact.
    """

    key: str
    path: str
    data_hash: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "path": self.path,
            "data_hash": self.data_hash,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> PretrainedLeafRecord:
        return cls(
            key=json_io.get_str(d, "key"),
            path=json_io.get_str(d, "path"),
            data_hash=json_io.get_str(d, "data_hash"),
            train_start=pd.Timestamp(json_io.get_str(d, "train_start")),
            train_end=pd.Timestamp(json_io.get_str(d, "train_end")),
        )


@dataclass(frozen=True)
class Manifest:
    """Canonical, round-tripable manifest for an experiment run."""

    experiment_id: str
    name: str
    created_at: datetime
    git_sha: str
    seed: int
    data_hash: str
    slippage_scenario: SlippageScenario
    holdout_start: pd.Timestamp | None = None
    pretrained_leaves: tuple[PretrainedLeafRecord, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "git_sha": self.git_sha,
            "seed": self.seed,
            "data_hash": self.data_hash,
            "slippage_scenario": self.slippage_scenario.value,
            "holdout_start": (
                self.holdout_start.isoformat() if self.holdout_start is not None else None
            ),
            "pretrained_leaves": [r.to_dict() for r in self.pretrained_leaves],
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Manifest:
        holdout = json_io.get_optional_timestamp(d, "holdout_start")
        raw_leaves = d.get("pretrained_leaves", [])
        if not isinstance(raw_leaves, list):
            raise ValueError(
                f"JSON field 'pretrained_leaves' must be a list, got {type(raw_leaves).__name__}"
            )
        leaves_parsed: list[PretrainedLeafRecord] = []
        for raw in raw_leaves:
            if not isinstance(raw, dict):
                raise ValueError(
                    f"JSON field 'pretrained_leaves' entries must be dicts, "
                    f"got {type(raw).__name__}"
                )
            leaves_parsed.append(PretrainedLeafRecord.from_dict(raw))
        leaves = tuple(leaves_parsed)
        return cls(
            experiment_id=json_io.get_str(d, "experiment_id"),
            name=json_io.get_str(d, "name"),
            created_at=datetime.fromisoformat(json_io.get_str(d, "created_at")),
            git_sha=json_io.get_str(d, "git_sha"),
            seed=json_io.get_int(d, "seed"),
            data_hash=json_io.get_str(d, "data_hash"),
            slippage_scenario=SlippageScenario(json_io.get_str(d, "slippage_scenario")),
            holdout_start=holdout,
            pretrained_leaves=leaves,
        )
