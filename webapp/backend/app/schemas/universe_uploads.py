"""Wire DTOs for user-authored universe-spec uploads."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Same slug rules as study uploads: alphanum-leading, ASCII-only, no
# path separators (the slug becomes a filename segment under uploads_root).
_SLUG_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_\-]*$"
_MAX_YAML_BYTES = 128 * 1024


class UniverseSpecUploadCreate(BaseModel):
    """POST /configs/universe/uploads body."""

    slug: str = Field(min_length=1, max_length=64, pattern=_SLUG_PATTERN)
    yaml: str = Field(min_length=1, max_length=_MAX_YAML_BYTES)


class UniverseSpecUploadSummary(BaseModel):
    """Listing entry — no YAML body."""

    slug: str
    created_at: datetime
    updated_at: datetime
    owner_user_id: int
    owner_username: str


class UniverseSpecUploadDetail(BaseModel):
    """Single-upload read shape with the full YAML body."""

    slug: str
    yaml: str
    created_at: datetime
    updated_at: datetime
    owner_user_id: int
    owner_username: str
