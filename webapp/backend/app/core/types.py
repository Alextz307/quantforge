"""
Cross-layer enums shared by schemas, services, and dependencies.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"
