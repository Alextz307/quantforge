"""
Typed manifest for a persisted experiment run.

Every field below is the answer to a question a post-run consumer
(holdout-eval, HPO resume) MUST answer to run safely. A
typed dataclass over a ``dict[str, object]`` catches typos like
``holdoutStart`` vs ``holdout_start`` at static-check time rather than
after hours of HPO compute.

Rationale for each field:

* ``experiment_id``    - opaque dir name consumers index by.
* ``name``             - human-readable label lifted from the config.
* ``created_at``       - UTC timestamp of the run start.
* ``git_sha``          - short SHA for reproducibility; best-effort
                         (``"unknown"`` if the run happens outside git).
* ``seed``             - int seeded into numpy / torch / random at run
                         start; required to reproduce walk-forward output.
* ``data_hash``        - ``fingerprint_bars(df)`` output; catches vendor
                         drift between runs.
* ``holdout_start``    - absolute pinned boundary timestamp (ISO string in
                         JSON). ``None`` when no holdout was reserved.
* ``slippage_scenario``- the ``SlippageScenario`` enum value used, so
                         downstream consumers know which friction model
                         produced the equity curve.
* ``interval``         - the data ``Interval`` the run was computed on, so
                         reuse/report paths can recover the annualization
                         factor without re-loading the frozen config.yaml.
* ``risk_free_rate``   - the rate subtracted when computing Sharpe, so a
                         recomputed pooled Sharpe stays on the same scale as
                         the persisted per-fold metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.core import json_io
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario


@dataclass(frozen=True)
class Manifest:
    """
    Canonical, round-tripable manifest for an experiment run.
    """

    experiment_id: str
    name: str
    created_at: datetime
    git_sha: str
    seed: int
    data_hash: str
    slippage_scenario: SlippageScenario
    interval: Interval
    risk_free_rate: float
    holdout_start: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "git_sha": self.git_sha,
            "seed": self.seed,
            "data_hash": self.data_hash,
            "slippage_scenario": self.slippage_scenario.value,
            "interval": self.interval.value,
            "risk_free_rate": self.risk_free_rate,
            "holdout_start": (
                self.holdout_start.isoformat() if self.holdout_start is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Manifest:
        holdout = json_io.get_optional_timestamp(d, "holdout_start")
        return cls(
            experiment_id=json_io.get_str(d, "experiment_id"),
            name=json_io.get_str(d, "name"),
            created_at=datetime.fromisoformat(json_io.get_str(d, "created_at")),
            git_sha=json_io.get_str(d, "git_sha"),
            seed=json_io.get_int(d, "seed"),
            data_hash=json_io.get_str(d, "data_hash"),
            slippage_scenario=SlippageScenario(json_io.get_str(d, "slippage_scenario")),
            interval=_read_interval(d),
            risk_free_rate=_read_risk_free_rate(d),
            holdout_start=holdout,
        )


def _read_interval(d: dict[str, object]) -> Interval:
    """
    Read ``interval``, defaulting to daily for manifests predating the field.

    Manifests written before ``interval`` was added carry no value; only the
    webapp run listing loads such legacy manifests, and it reports the
    interval from the frozen ``config.yaml`` (not from here), so the daily
    fallback is never surfaced. Every current run writes the real interval.
    """

    raw = d.get("interval")
    return Interval(raw) if isinstance(raw, str) else Interval.DAILY


def _read_risk_free_rate(d: dict[str, object]) -> float:
    """
    Read ``risk_free_rate``, defaulting to ``0.0`` for legacy manifests.

    Pre-field runs used the config default of ``0.0``, so that is the correct
    historical value to assume when the key is absent.
    """

    raw = d.get("risk_free_rate")
    return float(raw) if isinstance(raw, (int, float)) else 0.0
