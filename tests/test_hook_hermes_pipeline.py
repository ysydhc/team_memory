"""Tests for scripts/hooks/hermes_pipeline.py — thin forwarder to TM Daemon.

The HermesPipeline now delegates all heavy logic to the TM Daemon HTTP API.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from hooks.hermes_pipeline import HermesPipeline


class TestHermesPipelineOnTurnStart:
    """Tests for HermesPipeline.on_turn_start — forwards to daemon /hooks/before_prompt."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "project_context", "context": "some context"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response) as mock_post:
            result = await pipeline.on_turn_start("之前的问题", project="team_doc")

        assert result == {"action": "project_context", "context": "some context"}
        mock_post.assert_called_once()
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_daemon_not_running(self) -> None:
        pipeline = HermesPipeline()

        with patch.object(pipeline._client, "post", side_effect=httpx.ConnectError("refused")):
            result = await pipeline.on_turn_start("hello", project="team_doc")

        assert result == {"action": "ok", "message": "daemon not running"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_generic_error(self) -> None:
        pipeline = HermesPipeline()

        with patch.object(pipeline._client, "post", side_effect=RuntimeError("fail")):
            result = await pipeline.on_turn_start("hello")

        assert result == {"action": "error", "message": "fail"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_sends_prompt_and_project(self) -> None:
        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "ok"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response) as mock_post:
            await pipeline.on_turn_start("之前遇到的问题", project="my_proj")

        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["prompt"] == "之前遇到的问题"
        assert call_kwargs[1]["json"]["project"] == "my_proj"
        await pipeline.close()


class TestHermesPipelineOnTurnEnd:
    """Tests for HermesPipeline.on_turn_end — forwards to daemon /hooks/after_response."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "draft_saved", "draft_id": "d1"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/after_response"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response):
            result = await pipeline.on_turn_end("sess-1", "response text", project="team_doc")

        assert result == {"action": "draft_saved", "draft_id": "d1"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_daemon_not_running(self) -> None:
        pipeline = HermesPipeline()

        with patch.object(pipeline._client, "post", side_effect=httpx.ConnectError("refused")):
            result = await pipeline.on_turn_end("sess-1", "response text")

        assert result == {"action": "ok", "message": "daemon not running"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_generic_error(self) -> None:
        pipeline = HermesPipeline()

        with patch.object(pipeline._client, "post", side_effect=RuntimeError("fail")):
            result = await pipeline.on_turn_end("sess-1", "hello")

        assert result == {"action": "error", "message": "fail"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_sends_conversation_id_and_prompt(self) -> None:
        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "draft_saved"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/after_response"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response) as mock_post:
            await pipeline.on_turn_end(
                "sess-42", "agent response", project="my_proj",
                recent_tools=[{"name": "edit"}],
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["conversation_id"] == "sess-42"
        assert payload["prompt"] == "agent response"
        assert payload["project"] == "my_proj"
        assert payload["recent_tools"] == [{"name": "edit"}]
        await pipeline.close()


class TestHermesPipelineOnSessionEnd:
    """Tests for HermesPipeline.on_session_end — forwards to daemon /hooks/session_end."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        pipeline = HermesPipeline()

        mock_response = httpx.Response(
            200,
            json={"action": "published", "draft_id": "d1"},
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/session_end"),
        )

        with patch.object(pipeline._client, "post", return_value=mock_response):
            result = await pipeline.on_session_end("sess-1")

        assert result == {"action": "published", "draft_id": "d1"}
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_daemon_not_running_returns_none(self) -> None:
        pipeline = HermesPipeline()

        with patch.object(pipeline._client, "post", side_effect=httpx.ConnectError("refused")):
            result = await pipeline.on_session_end("sess-1")

        assert result is None
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_generic_error(self) -> None:
        pipeline = HermesPipeline()

        with patch.object(pipeline._client, "post", side_effect=RuntimeError("fail")):
            result = await pipeline.on_session_end("sess-1")

        assert result == {"action": "error", "message": "fail"}
        await pipeline.close()


class TestHermesPipelineContextManager:
    """Tests for HermesPipeline async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with HermesPipeline() as pipeline:
            mock_response = httpx.Response(
                200,
                json={"action": "ok"},
                request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
            )
            with patch.object(pipeline._client, "post", return_value=mock_response):
                result = await pipeline.on_turn_start("test")
                assert result == {"action": "ok"}

    @pytest.mark.asyncio
    async def test_custom_daemon_url(self) -> None:
        pipeline = HermesPipeline(daemon_url="http://localhost:9999")
        assert pipeline._daemon_url == "http://localhost:9999"
        await pipeline.close()
