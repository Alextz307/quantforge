"""Wire DTOs for the jobs subsystem."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

from src.orchestration.comparison import SignificanceTest
from src.orchestration.holdout_eval import SourceKind

# Slug charset matching the CLI's ``--out-name`` / ``--publish-label`` validators
# (letters, digits, underscore, hyphen, colon). Keep in sync with
# ``_validate_publish_label`` in ``scripts/experiment.py``.
_SLUG_PATTERN = r"^[A-Za-z0-9_\-:]+$"

_MIN_COMPARE_RUNS = 2
_MAX_COMPARE_RUNS = 8
_MAX_COMPARE_N_JOBS = 8


class JobKind(StrEnum):
    RUN = "run"
    TUNE = "tune"
    COMPARE = "compare"
    HOLDOUT = "holdout"
    STUDY = "study"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class ComparePayload(BaseModel):
    """Inputs for ``experiment compare`` in ``--reuse-runs`` mode.

    Each ``run_ids[i]`` must resolve to a completed run dir under the
    server's ``store_root``. The webapp passes each run's existing
    ``config.yaml`` to the CLI's ``--config`` flag in matching order.
    From-scratch compare (no ``--reuse-runs``) is intentionally out of
    scope here — heavy and rarely interactive.
    """

    run_ids: list[str] = Field(min_length=_MIN_COMPARE_RUNS, max_length=_MAX_COMPARE_RUNS)
    out_name: str = Field(min_length=1, pattern=_SLUG_PATTERN)
    significance_test: SignificanceTest = SignificanceTest.BOOTSTRAP
    n_jobs: int = Field(default=1, ge=1, le=_MAX_COMPARE_N_JOBS)
    write_report: bool = True
    publish_label: str | None = Field(default=None, pattern=_SLUG_PATTERN)


class HoldoutPayload(BaseModel):
    """Inputs for ``experiment holdout-eval`` (run-dir XOR hpo-best source).

    ``source_id`` is interpreted as a run id when ``source_kind='run'``
    and as an HPO study name when ``source_kind='hpo'``. The job_service
    resolves it to an on-disk path and surfaces a 422 with structured
    ``loc`` if the source is missing the required artifacts (a non-null
    ``holdout_start`` for runs, ``best_config.yaml`` for HPO studies).
    """

    source_kind: SourceKind
    source_id: str = Field(min_length=1)
    out_name: str | None = Field(default=None, min_length=1, pattern=_SLUG_PATTERN)
    write_report: bool = True
    publish_label: str | None = Field(default=None, pattern=_SLUG_PATTERN)


class StudyPayload(BaseModel):
    """Inputs for ``experiment study run`` (cross-strategy × cross-universe sweep).

    ``spec_name`` resolves to ``config/study/<spec_name>.yaml``; the
    job_service parses it via ``StudySpec.model_validate`` and surfaces a
    422 with structured ``loc`` on schema errors. ``only_legs`` entries
    are validated against the spec's expanded leg ids in the handler.
    """

    spec_name: str = Field(min_length=1, pattern=_SLUG_PATTERN)
    force_rerun: bool = False
    only_legs: list[str] = Field(default_factory=list, max_length=128)
    skip_compares: bool = False
    skip_holdout_eval: bool = False


# Per-kind payload contract: every kind owns exactly one set of payload fields,
# enforced by ``JobSubmission._validate_payload_per_kind``. Keeping this table
# next to the model makes the contract explicit at one glance.
_PAYLOAD_FIELDS: tuple[str, ...] = (
    "config_payload",
    "hpo_payload",
    "compare_payload",
    "holdout_payload",
    "study_payload",
)
_REQUIRED_PAYLOADS: dict[JobKind, frozenset[str]] = {
    JobKind.RUN: frozenset({"config_payload"}),
    JobKind.TUNE: frozenset({"config_payload", "hpo_payload"}),
    JobKind.COMPARE: frozenset({"compare_payload"}),
    JobKind.HOLDOUT: frozenset({"holdout_payload"}),
    JobKind.STUDY: frozenset({"study_payload"}),
}


class JobSubmission(BaseModel):
    kind: JobKind
    config_payload: dict[str, object] | None = Field(default=None, min_length=1)
    hpo_payload: dict[str, object] | None = Field(default=None, min_length=1)
    compare_payload: ComparePayload | None = None
    holdout_payload: HoldoutPayload | None = None
    study_payload: StudyPayload | None = None
    overrides: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _validate_payload_per_kind(self) -> Self:
        required = _REQUIRED_PAYLOADS[self.kind]
        for field in _PAYLOAD_FIELDS:
            present = getattr(self, field) is not None
            needed = field in required
            if needed and not present:
                raise ValueError(f"{field} is required when kind={self.kind.value!r}")
            if not needed and present:
                raise ValueError(f"{field} must be omitted when kind={self.kind.value!r}")
        return self


class JobRow(BaseModel):
    id: str
    user_id: int
    kind: JobKind
    status: JobStatus
    started_at: datetime | None
    finished_at: datetime | None
    exit_code: int | None
    experiment_id: str | None
    log_path: str
    pid: int | None


class JobLogFrame(BaseModel):
    type: Literal["log"] = "log"
    line: str


class JobStatusFrame(BaseModel):
    type: Literal["status"] = "status"
    status: JobStatus
    exit_code: int | None = None
    experiment_id: str | None = None


JobStreamFrame = Annotated[JobLogFrame | JobStatusFrame, Field(discriminator="type")]
