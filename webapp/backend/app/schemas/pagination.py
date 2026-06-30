"""
Shared pagination + sort primitives for the list read APIs.
"""

from __future__ import annotations

from enum import StrEnum


class SortOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"
