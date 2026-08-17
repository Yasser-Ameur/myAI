"""Structured, correlatable logging.

Emits single-line JSON records so every log entry can be correlated across
a session/turn by tools. Falls back to the module's plain text formatter when
the console is not configured for JSON.

The companion.yaml `logging:` section is honoured here (level + file).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format a LogRecord as a single-line JSON object.

    Message is the structured payload; the timestamp is ISO-8601 UTC. Extra
    context passed via `log.info(..., extra={"session_id": ...})` is merged in.
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    entry[key] = value
                else:
                    entry[key] = repr(value)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable single-line formatter with optional trace extras."""

    def __init__(self, fmt: str | None = None, datefmt: str = "%H:%M:%S") -> None:
        super().__init__(fmt or "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt)

    def format(self, record: logging.LogRecord) -> str:
        trace = ""
        sid = getattr(record, "session_id", None)
        tid = getattr(record, "turn_id", None)
        if sid or tid:
            trace = f" [session={sid} turn={tid}]"
        base = f"%(asctime)s %(levelname)s %(name)s{trace}: %(message)s"
        return logging.Formatter(base, self.datefmt).format(record)


def configure_logging(level: str | int | None = None, file: str | None = None,
                      json_output: bool = True) -> None:
    """Configure root logging from the companion `logging:` config section.

    Args:
        level: log level name or int; defaults to WARNING.
        file: optional path to also write structured records.
        json_output: use JSON records (console). Always JSON in the file.
    """
    if level is None:
        level = logging.WARNING
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = JsonFormatter() if json_output else TextFormatter()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if file:
        try:
            fh = logging.FileHandler(file, encoding="utf-8")
            fh.setFormatter(JsonFormatter())
            root.addHandler(fh)
        except OSError as exc:  # e.g. data/ dir missing
            logging.getLogger(__name__).warning("cannot open log file %s: %s", file, exc)
