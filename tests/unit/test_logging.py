
import json
import logging
import os

from companion.runtime.logging import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
)


def _record(session_id: str = "", turn_id: str = ""):
    logger = logging.getLogger("test.logger")
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 12,
        "hello %s", ("world",), None, extra={"session_id": session_id, "turn_id": turn_id},
    )
    return record


def test_json_formatter_emits_parseable_json():
    line = JsonFormatter().format(_record("sess_1", "turn_2"))
    parsed = json.loads(line)
    assert parsed["ts"]
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "hello world"
    assert parsed["session_id"] == "sess_1"
    assert parsed["turn_id"] == "turn_2"


def test_json_formatter_omits_reserved_fields():
    parsed = json.loads(JsonFormatter().format(_record()))
    assert "args" not in parsed
    assert "created" not in parsed
    assert "msg" not in parsed
    assert parsed["module"] == "test_logging"


def test_json_formatter_exc_info_serialized():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.getLogger("test.logger").makeRecord(
            "test.logger", logging.ERROR, __file__, 12,
            "failed", (), exc_info=__import__("sys").exc_info(),
        )
    parsed = json.loads(JsonFormatter().format(record))
    assert "boom" in parsed["exc"]


def test_text_formatter_includes_trace():
    out = TextFormatter().format(_record("sess_a", "turn_b"))
    assert "[session=sess_a turn=turn_b]" in out


def test_configure_logging_sets_level_and_handlers(tmp_path):
    logfile = str(tmp_path / "companion.log")
    configure_logging("info", file=logfile)
    root = logging.getLogger()
    assert root.level == logging.INFO
    assert len(root.handlers) >= 2
    assert os.path.exists(logfile)


def test_configure_logging_missing_dir_is_safe():
    configure_logging("warning", file="Z:/no/such/dir/companion.log")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert any(h.level == logging.NOTSET for h in root.handlers)


def test_configure_logging_defaults_to_warning():
    configure_logging()
    assert logging.getLogger().level == logging.WARNING
