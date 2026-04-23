"""Tests for scripts/hooks/cursor_after_response.py — afterAgentResponse hook with full draft pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import cursor_after_response
from convergence_detector import ConvergenceDetector
from draft_buffer import DraftBuffer
from draft_refiner import DraftRefiner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _hook_input(
    conversation_id: str = "conv-123",
    prompt: str = "hello world",
    workspace_roots: list | None = None,
    model: str = "claude-3.5",
) -> dict:
    """Build a hook input dict."""
    if workspace_roots is None:
        workspace_roots = ["/Users/yeshouyou/Work/agent/team_doc"]
    return {
        "conversation_id": conversation_id,
        "prompt": prompt,
        "workspace_roots": workspace_roots,
        "model": model,
    }


def _make_config(db_path: str | None = None) -> dict:
    """Build a pipeline config dict."""
    if db_path is None:
        db_path = _make_db_path()
    return {
        "tm": {"base_url": "http://localhost:3900"},
        "draft": {"db_path": db_path},
        "projects": {
            "team_doc": {"path_patterns": ["team_doc"]},
        },
    }


async def _create_draft(buf: DraftBuffer, project: str, conv_id: str, content: str) -> str:
    """Helper to create a draft in the buffer (for sync test setup)."""
    async with buf:
        return await buf.create_draft(project, conv_id, content)


# ---------------------------------------------------------------------------
# Test: process_response — first response creates draft
# ---------------------------------------------------------------------------

class TestFirstResponseCreatesDraft:
    """First agent response for a session creates a new draft."""

    @pytest.mark.asyncio
    async def test_creates_draft_on_first_response(self):
        """When no pending draft exists, save_draft is called."""
        db_path = _make_db_path()
        config = _make_config(db_path)
        input_data = _hook_input(prompt="因为服务器配置错误导致超时")

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-001"})
            mock_tm.draft_publish = AsyncMock(return_value={"status": "ok"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "draft_saved"
        assert result["convergence"] is False
        assert result["draft_id"] == "tm-draft-001"
        mock_tm.draft_save.assert_awaited_once()

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_first_response_stores_content(self):
        """First response stores the response text as draft content."""
        db_path = _make_db_path()
        config = _make_config(db_path)
        input_data = _hook_input(prompt="因为服务器超时")

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-002"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        # Verify buffer has the draft with the content
        buf = DraftBuffer(db_path)
        async with buf:
            pending = await buf.get_pending("conv-123")
        assert len(pending) == 1
        assert "因为服务器超时" in pending[0]["content"]

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — subsequent response updates draft (accumulated text)
# ---------------------------------------------------------------------------

class TestSubsequentResponseUpdatesDraft:
    """Subsequent agent response appends to existing draft content."""

    @pytest.mark.asyncio
    async def test_accumulates_text(self):
        """Second response appends to existing draft content."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        # First response: create a draft in the buffer
        buf = DraftBuffer(db_path)
        async with buf:
            await buf.upsert_draft(
                session_id="conv-acc",
                title="Session conv-acc draft",
                content="First response content",
                project="team_doc",
            )

        input_data = _hook_input(
            conversation_id="conv-acc",
            prompt="Second response content",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-acc"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "draft_saved"
        # The accumulated text should contain both pieces
        call_args = mock_tm.draft_save.call_args
        content_arg = call_args.kwargs.get("content") or call_args[1].get("content")
        assert "First response content" in content_arg
        assert "Second response content" in content_arg

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — converged response publishes draft
# ---------------------------------------------------------------------------

class TestConvergedResponsePublishesDraft:
    """When convergence is detected and a draft exists, publish it."""

    @pytest.mark.asyncio
    async def test_publishes_on_convergence_with_existing(self):
        """Converged response with existing draft triggers refine_and_publish."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        # Create a pending draft
        buf = DraftBuffer(db_path)
        async with buf:
            draft_id = await buf.create_draft("team_doc", "conv-pub", "因为服务器超时。")

        input_data = _hook_input(
            conversation_id="conv-pub",
            prompt="问题已经解决了",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-pub"})
            mock_tm.draft_publish = AsyncMock(return_value={"status": "ok"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "published"
        assert result["convergence"] is True
        assert result["status"] == "published"
        mock_tm.draft_publish.assert_awaited_once()

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_publishes_on_first_converged_response(self):
        """Converged response without existing draft saves then publishes."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(
            conversation_id="conv-first-pub",
            prompt="搞定了",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-first"})
            mock_tm.draft_publish = AsyncMock(return_value={"status": "ok"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "published"
        assert result["convergence"] is True
        mock_tm.draft_save.assert_awaited_once()
        mock_tm.draft_publish.assert_awaited_once()

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_published_draft_not_pending(self):
        """After publishing, the draft should no longer be pending."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        # Create a pending draft
        buf = DraftBuffer(db_path)
        async with buf:
            await buf.create_draft("team_doc", "conv-np", "因为需要重启。")

        input_data = _hook_input(
            conversation_id="conv-np",
            prompt="完成了，一切正常",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-np"})
            mock_tm.draft_publish = AsyncMock(return_value={"status": "ok"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                await cursor_after_response.process_response(input_data, config)

        # Verify draft is no longer pending
        buf2 = DraftBuffer(db_path)
        async with buf2:
            pending = await buf2.get_pending("conv-np")
        assert len(pending) == 0

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — non-converged response saves draft
# ---------------------------------------------------------------------------

class TestNonConvergedResponseSavesDraft:
    """Non-converged responses save/update drafts without publishing."""

    @pytest.mark.asyncio
    async def test_saves_without_publishing(self):
        """Non-converged response calls save_draft, not draft_publish."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(
            conversation_id="conv-nc",
            prompt="还在修改代码中",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-nc"})
            mock_tm.draft_publish = AsyncMock(return_value={"status": "ok"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "draft_saved"
        assert result["convergence"] is False
        mock_tm.draft_save.assert_awaited_once()
        mock_tm.draft_publish.assert_not_awaited()

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — empty response handled gracefully
# ---------------------------------------------------------------------------

class TestEmptyResponse:
    """Empty or missing response text is handled gracefully."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Empty prompt string still creates draft (no crash)."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(prompt="")

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-empty"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "draft_saved"

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_missing_prompt_key(self):
        """Missing prompt key in input still works (no crash)."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = {
            "conversation_id": "conv-no-prompt",
            "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
        }

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-mp"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "draft_saved"

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_none_prompt(self):
        """None prompt value is handled gracefully."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(prompt=None)
        # Override prompt to None explicitly
        input_data["prompt"] = None

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-draft-none"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "draft_saved"

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — project extracted from workspace_roots
# ---------------------------------------------------------------------------

class TestProjectExtraction:
    """Project is correctly extracted from workspace_roots."""

    @pytest.mark.asyncio
    async def test_project_from_workspace(self):
        """Project is resolved from workspace_roots[0]."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(workspace_roots=["/Users/yeshouyou/Work/agent/team_doc"])

        with patch("cursor_after_response.get_project_from_path", return_value="team_doc") as mock_get:
            with patch("cursor_after_response.TMClient") as MockTM:
                mock_tm = AsyncMock()
                mock_tm.draft_save = AsyncMock(return_value={"id": "tm-1"})
                MockTM.return_value = mock_tm

                result = await cursor_after_response.process_response(input_data, config)

        mock_get.assert_called_once_with(
            "/Users/yeshouyou/Work/agent/team_doc",
            config=config,
        )
        assert result["action"] == "draft_saved"

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_no_workspace_returns_ok(self):
        """No workspace_roots returns early with ok action."""
        config = _make_config()
        input_data = {"conversation_id": "conv-no-ws", "prompt": "hello"}

        result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "ok"
        assert result["draft_id"] == ""

        if os.path.exists(config["draft"]["db_path"]):
            os.unlink(config["draft"]["db_path"])

    @pytest.mark.asyncio
    async def test_no_project_returns_ok(self):
        """Unresolvable project returns early with ok action."""
        config = _make_config()
        input_data = _hook_input(workspace_roots=["/unknown/path"])

        with patch("cursor_after_response.get_project_from_path", return_value=None):
            result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "ok"
        assert result["draft_id"] == ""

        if os.path.exists(config["draft"]["db_path"]):
            os.unlink(config["draft"]["db_path"])

    @pytest.mark.asyncio
    async def test_empty_workspace_list_returns_ok(self):
        """Empty workspace_roots list returns early with ok action."""
        config = _make_config()
        # Override the default by explicitly setting workspace_roots to empty
        input_data = {
            "conversation_id": "conv-empty-ws",
            "prompt": "hello",
            "workspace_roots": [],
        }

        result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "ok"
        assert result["draft_id"] == ""

        if os.path.exists(config["draft"]["db_path"]):
            os.unlink(config["draft"]["db_path"])


# ---------------------------------------------------------------------------
# Test: main — integration entry point
# ---------------------------------------------------------------------------

class TestMainEntryPoint:
    """main() entry point works correctly."""

    def test_main_returns_valid_json(self):
        """main() should return valid JSON."""
        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = _hook_input(prompt="还在修改代码中")
            with patch("cursor_after_response.load_config") as mock_config:
                mock_config.return_value = _make_config()
                with patch("cursor_after_response.TMClient") as MockTM:
                    mock_tm = AsyncMock()
                    mock_tm.draft_save = AsyncMock(return_value={"id": "tm-main-1"})
                    MockTM.return_value = mock_tm
                    with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                        output = cursor_after_response.main()

        result = json.loads(output)
        assert result["action"] == "draft_saved"
        assert "convergence" in result
        assert "draft_id" in result

    def test_main_error_handling(self):
        """main() returns error JSON on exception."""
        with patch("cursor_after_response.parse_hook_input", side_effect=Exception("parse error")):
            output = cursor_after_response.main()

        result = json.loads(output)
        assert result["action"] == "error"
        assert result["convergence"] is False

    def test_main_converged_with_existing_draft(self):
        """main() with converged response and existing draft publishes."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        # Create existing draft
        buf = DraftBuffer(db_path)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_create_draft(buf, "team_doc", "conv-main-pub", "content"))
        finally:
            loop.close()

        with patch("cursor_after_response.parse_hook_input") as mock_parse:
            mock_parse.return_value = _hook_input(
                conversation_id="conv-main-pub",
                prompt="搞定了",
            )
            with patch("cursor_after_response.load_config", return_value=config):
                with patch("cursor_after_response.TMClient") as MockTM:
                    mock_tm = AsyncMock()
                    mock_tm.draft_save = AsyncMock(return_value={"id": "tm-mp"})
                    mock_tm.draft_publish = AsyncMock(return_value={"status": "ok"})
                    MockTM.return_value = mock_tm
                    with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                        output = cursor_after_response.main()

        result = json.loads(output)
        assert result["action"] == "published"
        assert result["convergence"] is True

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — session_id fallback
# ---------------------------------------------------------------------------

class TestSessionIdFallback:
    """conversation_id fallback to 'unknown' when missing."""

    @pytest.mark.asyncio
    async def test_missing_conversation_id_uses_unknown(self):
        """Missing conversation_id falls back to 'unknown'."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = {
            "prompt": "因为配置错误",
            "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
        }

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-unknown"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        assert result["action"] == "draft_saved"
        # Verify save_draft was called with session_id="unknown"
        call_args = mock_tm.draft_save.call_args
        conv_id = call_args.kwargs.get("conversation_id")
        assert conv_id == "unknown"

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_empty_conversation_id_uses_unknown(self):
        """Empty string conversation_id falls back to 'unknown'."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(conversation_id="", prompt="因为配置错误")

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-unknown2"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        call_args = mock_tm.draft_save.call_args
        conv_id = call_args.kwargs.get("conversation_id")
        assert conv_id == "unknown"

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — convergence detection integration
# ---------------------------------------------------------------------------

class TestConvergenceDetectionIntegration:
    """ConvergenceDetector is called with correct arguments."""

    @pytest.mark.asyncio
    async def test_detect_convergence_called_with_response_text(self):
        """ConvergenceDetector.detect_convergence is called with the response text."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(prompt="还在修改中")

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-cd"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                with patch.object(
                    ConvergenceDetector, "detect_convergence", return_value=False
                ) as mock_detect:
                    result = await cursor_after_response.process_response(input_data, config)

        mock_detect.assert_called_once_with(
            "还在修改中",
            recent_tools=[],
            current_path="team_doc",
            previous_path=None,
        )

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_convergence_with_explicit_signal(self):
        """Explicit convergence signal like '解决了' triggers publishing."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        # Pre-create a draft
        buf = DraftBuffer(db_path)
        async with buf:
            await buf.create_draft("team_doc", "conv-sig", "因为配置错误。")

        input_data = _hook_input(
            conversation_id="conv-sig",
            prompt="问题已经解决了",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-sig"})
            mock_tm.draft_publish = AsyncMock(return_value={"status": "ok"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        # Should detect convergence via ConvergenceDetector
        assert result["convergence"] is True
        assert result["action"] == "published"

        if os.path.exists(db_path):
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: process_response — accumulated text correctness
# ---------------------------------------------------------------------------

class TestAccumulatedTextCorrectness:
    """Accumulated text correctly concatenates all responses."""

    @pytest.mark.asyncio
    async def test_accumulated_with_existing_draft(self):
        """Accumulated text includes both old and new content."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        # Create existing draft with content
        buf = DraftBuffer(db_path)
        async with buf:
            await buf.create_draft("team_doc", "conv-acc2", "First part")

        input_data = _hook_input(
            conversation_id="conv-acc2",
            prompt="Second part",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-acc2"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        # Verify the accumulated text was passed to save_draft
        call_args = mock_tm.draft_save.call_args
        content = call_args.kwargs.get("content")
        assert "First part" in content
        assert "Second part" in content
        # Ensure newline-separated
        assert content == "First part\nSecond part"

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_first_response_no_existing(self):
        """First response with no existing draft uses response text directly."""
        db_path = _make_db_path()
        config = _make_config(db_path)

        input_data = _hook_input(
            conversation_id="conv-first",
            prompt="Initial content",
        )

        with patch("cursor_after_response.TMClient") as MockTM:
            mock_tm = AsyncMock()
            mock_tm.draft_save = AsyncMock(return_value={"id": "tm-first"})
            MockTM.return_value = mock_tm

            with patch("cursor_after_response.get_project_from_path", return_value="team_doc"):
                result = await cursor_after_response.process_response(input_data, config)

        call_args = mock_tm.draft_save.call_args
        content = call_args.kwargs.get("content")
        assert content == "Initial content"

        if os.path.exists(db_path):
            os.unlink(db_path)
