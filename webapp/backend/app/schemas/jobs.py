"""Wire DTOs for the jobs subsystem."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator


class JobKind(StrEnum):
    RUN = "run"
    TUNE = "tune"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class JobSubmission(BaseModel):
    kind: JobKind
    config_payload: dict[str, object] = Field(min_length=1)
    hpo_payload: dict[str, object] | None = Field(default=None, min_length=1)
    overrides: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _validate_hpo_payload(self) -> Self:
        if self.kind is JobKind.TUNE and self.hpo_payload is None:
            raise ValueError("hpo_payload is required when kind='tune'")
        if self.kind is JobKind.RUN and self.hpo_payload is not None:
            raise ValueError("hpo_payload must be omitted when kind='run'")
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
