"""Wire DTO for the public-settings endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class PublicSettings(BaseModel):
    jobs_enabled: bool
