"""Tests for platform-agnostic payload field reading in process_after_response."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from team_memory.daemon.refinement.pipeline import process_after_response


@pytest.fixture(autouse=True)
def _mock_auto_infer(monkeypatch):
    """Mock auto-inference so tests only exercise path_patterns."""
    monkeypatch.setattr("team_memory.project_resolver.read_tm_toml", lambda _: None)
    monkeypatch.setattr("team_memory.project_resolver.infer_project_name", lambda _cwd="": "default")


def _make_mocks(converged: bool = False):
    config = MagicMock()
    pm = MagicMock()
    pm.path_patterns = ["/repo"]
    pm.name = "proj"
    config.projects = [pm]

    sink = MagicMock()

    buf = AsyncMock()
    buf.add_message = AsyncMock(return_value=1)
    buf.ensure_session = AsyncMock(return_value="sess-1")
    buf.get_pending = AsyncMock(return_value=[])
    buf.count_messages = AsyncMock(return_value=0)

    detector = MagicMock()
    detector.detect_convergence = MagicMock(return_value=converged)

    refiner = AsyncMock()
    refiner.mark_for_refinement = AsyncMock(return_value=None)

    return config, sink, buf, detector, refiner


@pytest.mark.asyncio
async def test_response_text_field_is_used():
    """Claude Code sends 'response_text'; pipeline should use it as content."""
    config, sink, buf, detector, refiner = _make_mocks(converged=False)

    payload = {
        "conversation_id": "sess-1",
        "response_text": "Here is what I learned today",
        "prompt": "",  # empty prompt, response_text should win
        "workspace_roots": ["/repo"],
    }

    await process_after_response(payload, config, sink, buf, detector, refiner)

    buf.add_message.assert_called_once_with("sess-1", "assistant", "Here is what I learned today", project="proj")


@pytest.mark.asyncio
async def test_prompt_field_fallback():
    """Hermes / tm_hook.py sends 'prompt'; pipeline should fall back to it."""
    config, sink, buf, detector, refiner = _make_mocks(converged=False)

    payload = {
        "conversation_id": "sess-2",
        "prompt": "Agent replied with this text",
        "workspace_roots": ["/repo"],
    }

    await process_after_response(payload, config, sink, buf, detector, refiner)

    buf.add_message.assert_called_once_with("sess-2", "assistant", "Agent replied with this text", project="proj")


@pytest.mark.asyncio
async def test_response_text_takes_precedence_over_prompt():
    """When both fields are present, response_text wins."""
    config, sink, buf, detector, refiner = _make_mocks(converged=False)

    payload = {
        "conversation_id": "sess-3",
        "response_text": "correct content",
        "prompt": "wrong content",
        "workspace_roots": ["/repo"],
    }

    await process_after_response(payload, config, sink, buf, detector, refiner)

    buf.add_message.assert_called_once()
    args = buf.add_message.call_args
    assert args[0][2] == "correct content"  # content argument


@pytest.mark.asyncio
async def test_empty_content_skips_save():
    """If both fields are empty (or whitespace-only), pipeline skips draft creation."""
    config, sink, buf, detector, refiner = _make_mocks(converged=False)

    for payload in [
        {"conversation_id": "sess-4", "response_text": "", "prompt": "", "workspace_roots": ["/repo"]},
        {"conversation_id": "sess-5", "response_text": "\n\n", "prompt": "\n", "workspace_roots": ["/repo"]},
    ]:
        buf.add_message.reset_mock()
        result = await process_after_response(payload, config, sink, buf, detector, refiner)
        assert isinstance(result, dict)
        buf.add_message.assert_not_called()
