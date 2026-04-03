"""Tests for JSON log format (Phase 4 Task 2).

Validates that LOG_FORMAT=json produces valid JSON lines with required fields
per docs/design-docs/logging-format.md.
"""

from __future__ import annotations

import io
import json
import logging
import os

import pytest

from team_memory.bootstrap import _JsonFormatter
from team_memory.config import load_settings, reset_settings


def _parse_json_log_line(line: str) -> dict:
    """Parse a single JSON log line; raise if invalid."""
    return json.loads(line.strip())


def _assert_required_fields(obj: dict) -> None:
    """Assert required fields exist and have correct types."""
    assert "timestamp" in obj, "missing timestamp"
    assert "level" in obj, "missing level"
    assert "logger" in obj, "missing logger"
    assert "message" in obj, "missing message"
    assert isinstance(obj["timestamp"], str), "timestamp must be string"
    assert isinstance(obj["level"], str), "level must be string"
    assert isinstance(obj["logger"], str), "logger must be string"
    assert isinstance(obj["message"], str), "message must be string"
    # ISO 8601-like format
    assert "T" in obj["timestamp"] and "Z" in obj["timestamp"], "timestamp should be ISO 8601"
    assert obj["level"] in ("DEBUG", "INFO", "WARNING", "ERROR"), "level must be valid"


class TestJsonFormatter:
    """Test _JsonFormatter output structure."""

    def test_required_fields_present_and_typed(self) -> None:
        """Parse log samples; assert timestamp, level, logger, message exist and typed."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        log = logging.getLogger("team_memory.test_logging_json")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

        log.info("test message one")
        log.warning("test message two")
        log.error("test message three")

        lines = buf.getvalue().strip().split("\n")
        assert len(lines) >= 3
        for line in lines:
            obj = _parse_json_log_line(line)
            _assert_required_fields(obj)

    def test_extra_fields_in_output(self) -> None:
        """Extra dict is serialized into 'extra' key."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        log = logging.getLogger("team_memory.test_logging_json_extra")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

        log.info("request", extra={"path": "/api/v1/experiences", "method": "GET", "status": 200})

        obj = _parse_json_log_line(buf.getvalue().strip())
        _assert_required_fields(obj)
        assert obj["message"] == "request"
        assert "extra" in obj
        assert obj["extra"]["path"] == "/api/v1/experiences"
        assert obj["extra"]["method"] == "GET"
        assert obj["extra"]["status"] == 200

    def test_request_id_when_present(self) -> None:
        """request_id is included when set on record."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        log = logging.getLogger("team_memory.test_logging_json_reqid")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

        log.info("error", extra={"request_id": "req-abc123"})

        obj = _parse_json_log_line(buf.getvalue().strip())
        _assert_required_fields(obj)
        assert obj.get("request_id") == "req-abc123"

    def test_sensitive_keys_redacted(self) -> None:
        """Sensitive keys in extra are redacted to ***."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        log = logging.getLogger("team_memory.test_logging_json_sensitive")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

        log.info("auth", extra={"path": "/login", "api_key": "sk-12345", "password": "secret123"})

        obj = _parse_json_log_line(buf.getvalue().strip())
        _assert_required_fields(obj)
        assert obj["extra"]["path"] == "/login"
        assert obj["extra"]["api_key"] == "***"
        assert obj["extra"]["password"] == "***"

    def test_request_middleware_payload_structure(self) -> None:
        """Simulate request_log_middleware: message=request, extra has event/path/method/status."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        log = logging.getLogger("team_memory.web.request")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

        payload = {
            "event": "request",
            "path": "/api/v1/experiences",
            "method": "GET",
            "status": 200,
            "duration_ms": 12,
            "ip": "127.0.0.1",
            "user": "",
        }
        log.info("request", extra=payload)

        obj = _parse_json_log_line(buf.getvalue().strip())
        _assert_required_fields(obj)
        assert obj["message"] == "request"
        assert obj["logger"] == "team_memory.web.request"
        assert obj["extra"]["event"] == "request"
        assert obj["extra"]["path"] == "/api/v1/experiences"
        assert obj["extra"]["method"] == "GET"
        assert obj["extra"]["status"] == 200
        assert obj["extra"]["duration_ms"] == 12


class TestLogFormatConfig:
    """Test LOG_FORMAT config switch."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> None:
        reset_settings()
        yield
        reset_settings()

    def test_log_format_env_json(self) -> None:
        """LOG_FORMAT=json sets log_format to 'json'."""
        os.environ["LOG_FORMAT"] = "json"
        try:
            settings = load_settings()
            assert settings.log_format == "json"
        finally:
            os.environ.pop("LOG_FORMAT", None)

    def test_log_format_env_human(self) -> None:
        """LOG_FORMAT=human sets log_format to 'human'."""
        os.environ["LOG_FORMAT"] = "human"
        try:
            settings = load_settings()
            assert settings.log_format == "human"
        finally:
            os.environ.pop("LOG_FORMAT", None)

    def test_log_format_default_human(self) -> None:
        """When LOG_FORMAT unset, default is human."""
        from unittest.mock import patch

        os.environ.pop("LOG_FORMAT", None)
        os.environ.pop("TEAM_MEMORY_LOG_FORMAT", None)
        with patch("team_memory.config.settings._load_dotenv_if_available"):
            settings = load_settings()
        assert settings.log_format == "human"
