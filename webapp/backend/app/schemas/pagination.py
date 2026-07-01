"""
Shared pagination + sort primitives for the list read APIs.
"""

from __future__ import annotations

from enum import StrEnum

# The default and maximum page sizes every list endpoint accepts. MAX_PAGE_LIMIT
# is the FastAPI ``Query(le=...)`` cap; the frontend mirrors it as
# ``MAX_PAGE_LIMIT`` in ``api/client.ts`` and a frontend test asserts the two
# agree against the committed OpenAPI snapshot, so the cap has one source here.
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 500


class SortOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"
