"""
Wire DTOs for user-authored study-spec uploads.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Tighter than ``StudyPayload.spec_name`` in jobs.py: uploads must start with
# an alphanumeric and exclude ``:`` so the on-disk path stays simple.
# StudyPayload covers the wider set of user-supplied identifiers across all
# job kinds; uploads own the filename, so we keep them ASCII-safe.
_SLUG_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_\-]*$"


class StudySpecUploadCreate(BaseModel):
    """
    POST /configs/study/uploads body.
    """

    slug: str = Field(min_length=1, max_length=64, pattern=_SLUG_PATTERN)
    yaml: str = Field(min_length=1, max_length=131072)


class StudySpecUploadSummary(BaseModel):
    """
    Listing entry - no YAML body, no validation errors.
    """

    slug: str
    created_at: datetime
    updated_at: datetime
    owner_user_id: int
    owner_username: str


class StudySpecUploadDetail(BaseModel):
    """
    Single-upload read shape with the full YAML body.
    """

    slug: str
    yaml: str
    created_at: datetime
    updated_at: datetime
    owner_user_id: int
    owner_username: str
