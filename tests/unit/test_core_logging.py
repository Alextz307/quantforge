"""
Tests for :func:`src.core.logging.get_logger` - context-bound logger.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.core.logging import attach_run_log_file, get_logger
from src.core.persistence import EXPERIMENT_RUN_LOG


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
        """
        ``name`` must flow through ``logging.getLogger(name)`` so hierarchy
        + per-module level filters still apply."""

        logger = get_logger("src.some.module", run_id="x")
        assert logger.logger.name == "src.some.module"

    def test_interpolation_args_honored(self, caplog: pytest.LogCaptureFixture) -> None:
        """
        ``logger.info('n=%d', 7)`` must still interpolate correctly once the
        context prefix has been prepended."""

        logger = get_logger("test.args", run="abc")
        with caplog.at_level(logging.INFO, logger="test.args"):
            logger.info("n=%d", 7)
        assert any("n=7" in r.message for r in caplog.records)


class TestAttachRunLogFile:
    def test_writes_messages_into_run_dir_log(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "abc123"
        logger = logging.getLogger("test.run_log.write")
        logger.setLevel(logging.INFO)
        with attach_run_log_file(run_dir) as log_path:
            logger.info("inside the context")
        assert log_path == run_dir / EXPERIMENT_RUN_LOG
        body = log_path.read_text(encoding="utf-8")
        assert "inside the context" in body

    def test_handler_removed_on_exit(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "xyz"
        before = len(logging.getLogger().handlers)
        with attach_run_log_file(run_dir):
            pass
        assert len(logging.getLogger().handlers) == before

    def test_post_exit_message_not_written(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "post"
        logger = logging.getLogger("test.run_log.post")
        logger.setLevel(logging.INFO)
        with attach_run_log_file(run_dir) as log_path:
            logger.info("during run")
        logger.info("after run")
        body = log_path.read_text(encoding="utf-8")
        assert "during run" in body
        assert "after run" not in body
