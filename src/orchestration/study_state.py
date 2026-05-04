"""State dataclasses for the empirical-study orchestrator.

The study runs a long sequence of (strategy x universe) legs and must
survive process death + resume cleanly across days of compute. This
module owns the round-tripable JSON state that captures "what's done"
so a rerun can skip completed work.

Why a separate state file (not Manifest):

* Manifest is per-run, frozen, and conceptually immutable — adding an
  ``is_complete`` field there would muddle that contract.
* A leg has multiple sub-steps (tune, run, regime, holdout_eval); we
  need fine-grained state to resume mid-leg if a later step fails.
* Study-level concerns (which legs ran, against which spec snapshot)
  don't belong in any individual run's manifest.

Atomic write: :func:`write_study_state` writes via
``<path>.tmp -> os.replace(<path>)`` so a crash mid-write leaves the
prior valid state intact.

Spec-hash check: the orchestrator computes a sha256 over the spec YAML
bytes at the start of every run; resuming refuses if the hash differs
from the one recorded in the state file (prevents silent partial-runs
against a mutated spec).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from src.core import json_io
from src.core.fs import atomic_write_path


class LegStep(StrEnum):
    """The four sub-steps of a study leg, in canonical execution order."""

    TUNE = "tune"
    RUN = "run"
    REGIME = "regime"
    HOLDOUT_EVAL = "holdout_eval"


# Module-level aliases let call sites read as ``LEG_STEP_TUNE`` (matches the
# rest of the codebase's bare-constant style for things-named-once-and-imported).
LEG_STEP_TUNE = LegStep.TUNE
LEG_STEP_RUN = LegStep.RUN
LEG_STEP_REGIME = LegStep.REGIME
LEG_STEP_HOLDOUT_EVAL = LegStep.HOLDOUT_EVAL

# Canonical step ordering — consumed by the orchestrator and by tests
# asserting resume-from-mid-leg semantics.
LEG_STEPS_ORDER: tuple[LegStep, ...] = (
    LegStep.TUNE,
    LegStep.RUN,
    LegStep.REGIME,
    LegStep.HOLDOUT_EVAL,
)


@dataclass(frozen=True)
class LegState:
    """Per-leg progress record.

    ``is_complete`` is stored explicitly (not derived from
    ``steps_completed``) because the expected step set varies per leg —
    a universe with ``holdout_pct=0`` legitimately skips ``holdout_eval``
    yet is still "complete". The orchestrator owns that decision.

    ``run_experiment_id`` is populated after the ``run`` step finishes
    so cross-strategy compare can later resolve
    ``<study_dir>/runs/<run_experiment_id>/`` for each leg without
    re-walking the runs directory.
    """

    leg_id: str
    strategy: str
    universe: str
    started_at: datetime | None
    completed_at: datetime | None
    steps_completed: tuple[LegStep, ...]
    is_complete: bool
    error: str | None
    run_experiment_id: str | None

    @classmethod
    def initial(cls, leg_id: str, strategy: str, universe: str) -> Self:
        """Construct a fresh, never-started leg state."""
        return cls(
            leg_id=leg_id,
            strategy=strategy,
            universe=universe,
            started_at=None,
            completed_at=None,
            steps_completed=(),
            is_complete=False,
            error=None,
            run_experiment_id=None,
        )

    def with_step_completed(self, step: LegStep) -> Self:
        """Return a copy with ``step`` appended (idempotent on re-add)."""
        if step in self.steps_completed:
            return self
        return replace(self, steps_completed=(*self.steps_completed, step))

    def to_dict(self) -> dict[str, object]:
        return {
            "leg_id": self.leg_id,
            "strategy": self.strategy,
            "universe": self.universe,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at is not None else None
            ),
            "steps_completed": list(self.steps_completed),
            "is_complete": self.is_complete,
            "error": self.error,
            "run_experiment_id": self.run_experiment_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        return cls(
            leg_id=json_io.get_str(d, "leg_id"),
            strategy=json_io.get_str(d, "strategy"),
            universe=json_io.get_str(d, "universe"),
            started_at=json_io.get_optional_iso_datetime(d, "started_at"),
            completed_at=json_io.get_optional_iso_datetime(d, "completed_at"),
            steps_completed=tuple(LegStep(s) for s in json_io.get_str_list(d, "steps_completed")),
            is_complete=json_io.get_bool(d, "is_complete"),
            error=json_io.get_optional_str(d, "error"),
            run_experiment_id=json_io.get_optional_str(d, "run_experiment_id"),
        )


@dataclass(frozen=True)
class StudyState:
    """Top-level study progress: leg roster + cross-strategy compare status."""

    spec_name: str
    spec_hash: str
    started_at: datetime
    legs: tuple[LegState, ...]
    cross_strategy_compares_done: tuple[str, ...]

    def with_leg(self, updated: LegState) -> Self:
        """Return a copy with ``updated`` replacing the leg of the same ``leg_id``."""
        new_legs = tuple(updated if leg.leg_id == updated.leg_id else leg for leg in self.legs)
        if not any(leg.leg_id == updated.leg_id for leg in self.legs):
            raise KeyError(f"leg_id '{updated.leg_id}' not in StudyState")
        return replace(self, legs=new_legs)

    def with_compare_done(self, universe: str) -> Self:
        if universe in self.cross_strategy_compares_done:
            return self
        return replace(
            self, cross_strategy_compares_done=(*self.cross_strategy_compares_done, universe)
        )

    def get_leg(self, leg_id: str) -> LegState:
        for leg in self.legs:
            if leg.leg_id == leg_id:
                return leg
        raise KeyError(f"leg_id '{leg_id}' not in StudyState")

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_name": self.spec_name,
            "spec_hash": self.spec_hash,
            "started_at": self.started_at.isoformat(),
            "legs": [leg.to_dict() for leg in self.legs],
            "cross_strategy_compares_done": list(self.cross_strategy_compares_done),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        leg_dicts = json_io.get_list_of_dicts(d, "legs")
        return cls(
            spec_name=json_io.get_str(d, "spec_name"),
            spec_hash=json_io.get_str(d, "spec_hash"),
            started_at=datetime.fromisoformat(json_io.get_str(d, "started_at")),
            legs=tuple(LegState.from_dict(raw) for raw in leg_dicts),
            cross_strategy_compares_done=tuple(
                json_io.get_str_list(d, "cross_strategy_compares_done")
            ),
        )


def compute_spec_hash(spec_path: Path) -> str:
    """SHA-256 of the spec YAML bytes — pins which spec the state belongs to."""
    return hashlib.sha256(spec_path.read_bytes()).hexdigest()


def write_study_state(path: Path, state: StudyState) -> None:
    """Atomically persist ``state`` at ``path``.

    A crash mid-write leaves the prior valid file intact (no half-written
    JSON) — the orchestrator may take days to complete and routinely
    survives Ctrl+C between legs.
    """
    with atomic_write_path(path) as tmp:
        json_io.write(tmp, state.to_dict())


def read_study_state(path: Path) -> StudyState:
    """Load and validate a previously-written :class:`StudyState`."""
    return StudyState.from_dict(json_io.read_dict(path))
