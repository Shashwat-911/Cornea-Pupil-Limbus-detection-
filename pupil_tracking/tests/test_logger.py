"""Tests for AuditLogger, including the exception() method."""

import logging
import uuid

import pytest

from pupil_tracking.utils.logger import AuditLogger


@pytest.fixture
def audit_logger(tmp_path):
    """Create a fresh AuditLogger with a temporary log directory."""
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    logger = AuditLogger(log_dir=str(tmp_path), session_id=session_id)
    yield logger
    logger.close()


class TestAuditLoggerException:
    """Tests for the exception() method added to fix GUI crash handlers."""

    def test_exception_method_exists(self):
        """AuditLogger must expose an exception() method."""
        assert hasattr(AuditLogger, "exception")
        assert callable(getattr(AuditLogger, "exception"))

    def test_exception_delegates_to_logging(self, audit_logger, tmp_path):
        """exception() must write to the underlying Python logger."""
        try:
            raise ValueError("test error")
        except ValueError:
            audit_logger.exception("caught error")

        log_file = tmp_path / f"session_{audit_logger.session_id}.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "caught error" in content
        assert "ERROR" in content

    def test_exception_preserves_traceback(self, audit_logger, tmp_path):
        """exception() must include traceback information in the log."""
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            audit_logger.exception("traceback test")

        log_file = tmp_path / f"session_{audit_logger.session_id}.log"
        content = log_file.read_text(encoding="utf-8")
        assert "Traceback" in content or "most recent call last" in content

    def test_exception_with_exc_info_kwarg(self, audit_logger):
        """exception() must accept exc_info= kwarg without TypeError.

        This is the exact call pattern from gui_app.py crash handlers:
            self.logger.exception("msg", exc_info=(exc_type, exc_value, exc_tb))
        """
        try:
            raise TypeError("type error")
        except TypeError as e:
            exc_type, exc_value, exc_tb = type(e), e, e.__traceback__
            # Must NOT raise TypeError about unexpected keyword argument
            audit_logger.exception(
                "gui crash handler pattern",
                exc_info=(exc_type, exc_value, exc_tb),
            )

    def test_exception_does_not_raise_attributeerror(self, audit_logger):
        """The original bug: calling exception() raised AttributeError.

        Before the fix, this would raise:
            AttributeError: 'AuditLogger' object has no attribute 'exception'
        """
        audit_logger.exception("no attributeerror please")

    def test_exception_with_format_args(self, audit_logger, tmp_path):
        """exception() must support printf-style format arguments."""
        try:
            raise OSError("disk full")
        except OSError:
            audit_logger.exception("operation %s failed on device %s", "write", "sda")

        log_file = tmp_path / f"session_{audit_logger.session_id}.log"
        content = log_file.read_text(encoding="utf-8")
        assert "operation write failed on device sda" in content
