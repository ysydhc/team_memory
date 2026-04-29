"""Tests for platform-agnostic payload field reading in process_after_response."""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from daemon.pipeline import process_after_response


def _make_mocks(converged: bool = False):
    config = MagicMock()
    pm = MagicMock()
    pm.path_patterns = ["/repo"]
    pm.name = "proj"
    config.projects = [pm]

    sink = MagicMock()

    buf = MagicMock()
    buf.get_pending = AsyncMock(return_value=[])
    buf.create_draft = AsyncMock(return_value=None)
    buf.update_draft = AsyncMock(return_value=None)

    detector = MagicMock()
    detector.detect_convergence = MagicMock(return_value=converged)

    refiner = MagicMock()
    refiner.save_draft = AsyncMock(return_value={"id": "draft-123", "status": "draft"})
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

    # save_draft was called — capture the content argument
    refiner.save_draft.assert_called_once()
    args, kwargs = refiner.save_draft.call_args
    content_arg = args[2] if len(args) > 2 else kwargs.get("content", "")
    assert "Here is what I learned today" in content_arg


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

    refiner.save_draft.assert_called_once()
    args, kwargs = refiner.save_draft.call_args
    content_arg = args[2] if len(args) > 2 else kwargs.get("content", "")
    assert "Agent replied with this text" in content_arg


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

    refiner.save_draft.assert_called_once()
    args, kwargs = refiner.save_draft.call_args
    content_arg = args[2] if len(args) > 2 else kwargs.get("content", "")
    assert "correct content" in content_arg
    assert "wrong content" not in content_arg


@pytest.mark.asyncio
async def test_empty_content_skips_save():
    """If both fields are empty (or whitespace-only), pipeline skips draft creation."""
    config, sink, buf, detector, refiner = _make_mocks(converged=False)

    for payload in [
        {"conversation_id": "sess-4", "response_text": "", "prompt": "", "workspace_roots": ["/repo"]},
        {"conversation_id": "sess-5", "response_text": "\n\n", "prompt": "\n", "workspace_roots": ["/repo"]},
    ]:
        refiner.save_draft.reset_mock()
        result = await process_after_response(payload, config, sink, buf, detector, refiner)
        assert isinstance(result, dict)
        refiner.save_draft.assert_not_called()
