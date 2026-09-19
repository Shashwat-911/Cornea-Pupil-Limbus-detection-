"""
Audit logging for surgical traceability.

Every detection, every alert, and every configuration change is
recorded in a machine-readable JSON-lines audit file that can be
reviewed post-operatively.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class AuditLogger:
    """Thread-safe logger with console + file + JSON audit trail."""

    def __init__(
        self,
        log_dir: str = "logs",
        session_id: Optional[str] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self._py = logging.getLogger(f"pupil_tracking.{self.session_id}")
        self._py.setLevel(logging.DEBUG)
        self._py.propagate = False

        if not self._py.handlers:
            # console — force UTF-8 on Windows
            try:
                console_stream = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace",
                    line_buffering=True,
                )
            except AttributeError:
                console_stream = sys.stdout

            ch = logging.StreamHandler(stream=console_stream)
            ch.setLevel(logging.INFO)
            ch.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self._py.addHandler(ch)

            # file — always UTF-8
            fh = logging.FileHandler(
                str(self.log_dir / f"session_{self.session_id}.log"),
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
            )
            self._py.addHandler(fh)

        self._audit_path = (
            self.log_dir / f"audit_{self.session_id}.jsonl"
        )
        self._audit_fh = open(
            str(self._audit_path), "a", encoding="utf-8"
        )
        self._py.info("Session started: %s", self.session_id)

    # ── structured audit ────────────────────────────────────────

    def _write_audit(self, record: Dict[str, Any]) -> None:
        record.setdefault("timestamp", time.time())
        self._audit_fh.write(json.dumps(record, default=str) + "\n")
        self._audit_fh.flush()

    def log_detection(
        self, result_dict: Dict[str, Any], frame: int = -1
    ) -> None:
        self._write_audit(
            {"type": "detection", "frame": frame, "result": result_dict}
        )

    def log_alert(self, message: str, severity: str = "WARNING") -> None:
        self._py.warning("ALERT [%s]: %s", severity, message)
        self._write_audit(
            {"type": "alert", "severity": severity, "message": message}
        )

    # ── convenience ─────────────────────────────────────────────

    def info(self, msg: str, *args: Any) -> None:
        self._py.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._py.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._py.error(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self._py.debug(msg, *args)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ERROR level with traceback information.

        Mirrors ``logging.Logger.exception()`` so that GUI crash
        handlers (``gui_app.py``) can call ``self.logger.exception(...)``
        without raising ``AttributeError``.
        """
        kwargs.setdefault("exc_info", True)
        self._py.error(msg, *args, **kwargs)

    # ── lifecycle ───────────────────────────────────────────────

    def close(self) -> None:
        if self._audit_fh and not self._audit_fh.closed:
            self._audit_fh.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── global singleton ────────────────────────────────────────────────

_logger: Optional[AuditLogger] = None


def get_logger() -> AuditLogger:
    global _logger
    if _logger is None:
        _logger = AuditLogger()
    return _logger


def set_logger(logger: AuditLogger) -> None:
    global _logger
    _logger = logger