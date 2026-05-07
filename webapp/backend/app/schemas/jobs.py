"""Wire DTOs for the jobs subsystem."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class JobKind(StrEnum):
    RUN = "run"


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
    overrides: list[str] = Field(default_factory=list, max_length=64)


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
