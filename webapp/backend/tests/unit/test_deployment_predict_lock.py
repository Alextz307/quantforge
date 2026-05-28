"""
The per-deployment predict lock that stops concurrent predict-if-stale calls
from both missing the cache and appending a duplicate signal row.

A real two-thread race is non-deterministic to assert; this pins the lock's
contract instead — one stable lock per deployment id (so two callers serialize
on the same object), independent locks across deployments (so they don't block
each other), and a clean reset between tests.
"""

from __future__ import annotations

from webapp.backend.app.services import deployment_service


def test_predict_lock_is_stable_per_deployment() -> None:
    first = deployment_service._predict_lock("dep_a")
    again = deployment_service._predict_lock("dep_a")
    other = deployment_service._predict_lock("dep_b")

    assert first is again
    assert first is not other


def test_clear_caches_resets_predict_locks() -> None:
    original = deployment_service._predict_lock("dep_a")
    deployment_service._clear_caches_for_tests()

    assert deployment_service._predict_lock("dep_a") is not original
