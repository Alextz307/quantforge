"""Tests for :func:`src.core.logging.get_logger` — context-bound logger."""

from __future__ import annotations

import logging

import pytest

from src.core.logging import get_logger


class TestContextualLogger:
    def test_empty_context_passes_message_through(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.empty")
        with caplog.at_level(logging.INFO, logger="test.empty"):
            logger.info("hello")
        assert any(r.message == "hello" for r in caplog.records)

    def test_context_prefixes_message(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.prefix", experiment_id="abc123", strategy="FooStrat")
        with caplog.at_level(logging.INFO, logger="test.prefix"):
            logger.info("fold 2/5")
        messages = [r.message for r in caplog.records]
        assert any("[experiment_id=abc123 strategy=FooStrat]" in m for m in messages)
        assert any("fold 2/5" in m for m in messages)

    def test_logger_name_preserved(self) -> None:
        """``name`` must flow through ``logging.getLogger(name)`` so hierarchy
        + per-module level filters still apply."""
        logger = get_logger("src.some.module", run_id="x")
        assert logger.logger.name == "src.some.module"

    def test_interpolation_args_honored(self, caplog: pytest.LogCaptureFixture) -> None:
        """``logger.info('n=%d', 7)`` must still interpolate correctly once the
        context prefix has been prepended."""
        logger = get_logger("test.args", run="abc")
        with caplog.at_level(logging.INFO, logger="test.args"):
            logger.info("n=%d", 7)
        assert any("n=7" in r.message for r in caplog.records)
