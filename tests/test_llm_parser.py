"""Tests for LLM parser service (services/llm_parser.py).

Validates parse_content, parse_personal_memory, suggest_experience_type,
and generate_summary by mocking the httpx HTTP calls to the LLM provider.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from team_memory.services.llm_parser import (
    LLMParseError,
    _extract_json,
    _normalize_single,
    generate_summary,
    parse_content,
    parse_personal_memory,
    suggest_experience_type,
)

# ============================================================
# Helpers
# ============================================================


def _make_llm_config(
    model: str = "test-model",
    base_url: str = "http://localhost:11434",
    prompt_dir: str | None = None,
) -> MagicMock:
    """Build a minimal LLMConfig-like mock."""
    cfg = MagicMock()
    cfg.model = model
    cfg.base_url = base_url
    cfg.prompt_dir = prompt_dir
    return cfg


def _ollama_response(content: str) -> dict:
    """Wrap a string into the Ollama API response shape."""
    return {"message": {"content": content}}


def _mock_httpx_post(return_json: dict, status_code: int = 200) -> AsyncMock:
    """Create an AsyncMock for httpx.AsyncClient.post that returns *return_json*."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = return_json
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.text = json.dumps(return_json)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ============================================================
# parse_content
# ============================================================


class TestParseContent:
    """Tests for parse_content() -- single-experience parsing."""

    @pytest.mark.asyncio
    async def test_parse_content_valid_json(self):
        """LLM returns valid JSON -- correct extraction."""
        llm_output = json.dumps(
            {
                "title": "Fix timeout issue",
                "problem": "API returned 502",
                "solution": "Increase proxy_read_timeout",
                "tags": ["nginx", "timeout"],
                "experience_type": "bugfix",
            }
        )
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_content("some content", llm_config=_make_llm_config())

        assert result["title"] == "Fix timeout issue"
        assert result["problem"] == "API returned 502"
        assert result["solution"] == "Increase proxy_read_timeout"
        assert "nginx" in result["tags"]
        assert result["experience_type"] == "bugfix"

    @pytest.mark.asyncio
    async def test_parse_content_malformed_json(self):
        """LLM returns broken JSON -- raises LLMParseError."""
        llm_output = "This is not valid JSON at all {{{broken"
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMParseError, match="Failed to parse"):
                await parse_content("some content", llm_config=_make_llm_config())

    @pytest.mark.asyncio
    async def test_parse_content_empty_response(self):
        """LLM returns empty content -- raises LLMParseError."""
        resp = _mock_httpx_post(_ollama_response(""))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMParseError, match="empty response"):
                await parse_content("some content", llm_config=_make_llm_config())

    @pytest.mark.asyncio
    async def test_parse_content_missing_fields(self):
        """JSON lacks optional fields -- handled gracefully with defaults."""
        llm_output = json.dumps({"title": "Minimal"})
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_content("some content", llm_config=_make_llm_config())

        assert result["title"] == "Minimal"
        assert result["problem"] == ""
        assert result["solution"] is None
        assert result["tags"] == []
        assert result["experience_type"] == "general"

    @pytest.mark.asyncio
    async def test_parse_content_timeout(self):
        """LLM times out -- raises LLMParseError."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMParseError, match="timed out"):
                await parse_content("some content", llm_config=_make_llm_config())

    @pytest.mark.asyncio
    async def test_parse_content_connect_error(self):
        """LLM connection refused -- raises LLMParseError."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMParseError, match="Cannot connect"):
                await parse_content("some content", llm_config=_make_llm_config())

    @pytest.mark.asyncio
    async def test_parse_content_invalid_experience_type_defaults_to_general(self):
        """If LLM returns an unknown experience_type, it normalises to 'general'."""
        llm_output = json.dumps(
            {
                "title": "Something",
                "problem": "A problem",
                "solution": "A solution",
                "tags": ["test"],
                "experience_type": "totally_invalid_type",
            }
        )
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_content("some content", llm_config=_make_llm_config())

        assert result["experience_type"] == "general"

    @pytest.mark.asyncio
    async def test_parse_content_json_in_code_block(self):
        """LLM wraps JSON in markdown code block -- still parsed."""
        inner = json.dumps(
            {
                "title": "Wrapped",
                "problem": "In code block",
                "solution": "Strip it",
                "tags": ["markdown"],
                "experience_type": "general",
            }
        )
        llm_output = f"```json\n{inner}\n```"
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_content("some content", llm_config=_make_llm_config())

        assert result["title"] == "Wrapped"


# ============================================================
# parse_personal_memory
# ============================================================


class TestParsePersonalMemory:
    """Tests for parse_personal_memory() -- preference extraction."""

    @pytest.mark.asyncio
    async def test_parse_personal_memory_valid(self):
        """Returns list of preference items from valid LLM response."""
        llm_output = json.dumps(
            [
                {
                    "content": "Prefers concise responses",
                    "profile_kind": "static",
                    "scope": "generic",
                    "context_hint": None,
                },
                {
                    "content": "Working on MCP server refactor",
                    "profile_kind": "dynamic",
                    "scope": "context",
                    "context_hint": "MCP refactoring sprint",
                },
            ]
        )
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_personal_memory("some conversation", llm_config=_make_llm_config())

        assert len(result) == 2
        assert result[0]["content"] == "Prefers concise responses"
        assert result[0]["profile_kind"] == "static"
        assert result[0]["scope"] == "generic"
        assert result[1]["profile_kind"] == "dynamic"
        assert result[1]["scope"] == "context"
        assert result[1]["context_hint"] == "MCP refactoring sprint"

    @pytest.mark.asyncio
    async def test_parse_personal_memory_empty(self):
        """LLM returns empty array -- returns empty list."""
        llm_output = "[]"
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_personal_memory("trivial chat", llm_config=_make_llm_config())

        assert result == []

    @pytest.mark.asyncio
    async def test_parse_personal_memory_timeout(self):
        """Timeout handled gracefully -- returns empty list, no exception."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_personal_memory("some content", llm_config=_make_llm_config())

        assert result == []

    @pytest.mark.asyncio
    async def test_parse_personal_memory_connect_error(self):
        """Connection error handled gracefully -- returns empty list."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_personal_memory("some content", llm_config=_make_llm_config())

        assert result == []

    @pytest.mark.asyncio
    async def test_parse_personal_memory_malformed_json(self):
        """LLM returns non-JSON text -- returns empty list."""
        llm_output = "Not valid JSON at all"
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_personal_memory("some content", llm_config=_make_llm_config())

        assert result == []

    @pytest.mark.asyncio
    async def test_parse_personal_memory_empty_content_items_filtered(self):
        """Items with empty content are filtered out."""
        llm_output = json.dumps(
            [
                {
                    "content": "Valid preference",
                    "profile_kind": "static",
                    "scope": "generic",
                    "context_hint": None,
                },
                {"content": "", "profile_kind": "static", "scope": "generic", "context_hint": None},
                {
                    "content": "  ",
                    "profile_kind": "static",
                    "scope": "generic",
                    "context_hint": None,
                },
            ]
        )
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_personal_memory("some content", llm_config=_make_llm_config())

        assert len(result) == 1
        assert result[0]["content"] == "Valid preference"

    @pytest.mark.asyncio
    async def test_parse_personal_memory_scope_normalization(self):
        """Invalid scope/profile_kind values are normalised."""
        llm_output = json.dumps(
            [
                {
                    "content": "Preference with bad scope",
                    "profile_kind": "something_wrong",
                    "scope": "invalid_scope",
                    "context_hint": None,
                },
            ]
        )
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await parse_personal_memory("some content", llm_config=_make_llm_config())

        assert len(result) == 1
        # Invalid scope normalises to "generic"; invalid profile_kind with generic scope -> "static"
        assert result[0]["profile_kind"] == "static"
        assert result[0]["scope"] == "generic"


# ============================================================
# suggest_experience_type
# ============================================================


class TestSuggestExperienceType:
    """Tests for suggest_experience_type() -- lightweight classification."""

    @pytest.mark.asyncio
    async def test_suggest_experience_type_valid(self):
        """Returns valid type with confidence."""
        llm_output = json.dumps(
            {
                "type": "bugfix",
                "confidence": 0.9,
                "reason": "Contains error and fix",
                "fallback_types": ["incident"],
            }
        )
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await suggest_experience_type(
                "Fix 502 error", problem="API timeout", llm_config=_make_llm_config()
            )

        assert result["suggested_type"] == "bugfix"
        assert result["confidence"] == 0.9
        assert result["reason"] == "Contains error and fix"
        assert "incident" in result["fallback_types"]

    @pytest.mark.asyncio
    async def test_suggest_experience_type_invalid_response(self):
        """LLM returns unparseable JSON -- returns default 'general'."""
        llm_output = "Not JSON at all"
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await suggest_experience_type("Some title", llm_config=_make_llm_config())

        assert result["suggested_type"] == "general"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_suggest_experience_type_timeout(self):
        """Timeout -- returns default 'general' without raising."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await suggest_experience_type("Some title", llm_config=_make_llm_config())

        assert result["suggested_type"] == "general"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_suggest_experience_type_empty_response(self):
        """Empty LLM response -- returns default 'general'."""
        resp = _mock_httpx_post(_ollama_response(""))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await suggest_experience_type("Some title", llm_config=_make_llm_config())

        assert result["suggested_type"] == "general"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_suggest_experience_type_unknown_type_defaults(self):
        """LLM returns unknown type -- normalises to 'general'."""
        llm_output = json.dumps(
            {
                "type": "made_up_type",
                "confidence": 0.8,
                "reason": "unsure",
                "fallback_types": [],
            }
        )
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await suggest_experience_type("Some title", llm_config=_make_llm_config())

        assert result["suggested_type"] == "general"


# ============================================================
# generate_summary
# ============================================================


class TestGenerateSummary:
    """Tests for generate_summary() -- summary generation."""

    @pytest.mark.asyncio
    async def test_generate_summary_valid(self):
        """Returns summary string from LLM."""
        llm_output = "This experience describes fixing a 502 error by increasing timeout."
        resp = _mock_httpx_post(_ollama_response(llm_output))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await generate_summary("Fix 502 error...", llm_config=_make_llm_config())

        assert result == "This experience describes fixing a 502 error by increasing timeout."

    @pytest.mark.asyncio
    async def test_generate_summary_timeout(self):
        """Timeout -- raises LLMParseError."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMParseError, match="timed out"):
                await generate_summary("some content", llm_config=_make_llm_config())

    @pytest.mark.asyncio
    async def test_generate_summary_empty_response(self):
        """Empty LLM response -- raises LLMParseError."""
        resp = _mock_httpx_post(_ollama_response(""))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(return_value=resp))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMParseError, match="empty summary"):
                await generate_summary("some content", llm_config=_make_llm_config())

    @pytest.mark.asyncio
    async def test_generate_summary_connect_error(self):
        """Connection error -- raises LLMParseError."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(LLMParseError, match="Cannot connect"):
                await generate_summary("some content", llm_config=_make_llm_config())


# ============================================================
# Internal helpers
# ============================================================


class TestExtractJson:
    """Tests for _extract_json() -- JSON extraction from LLM text."""

    def test_plain_json(self):
        raw = '{"title": "Hello"}'
        assert _extract_json(raw) == {"title": "Hello"}

    def test_json_in_code_block(self):
        raw = '```json\n{"title": "Hello"}\n```'
        assert _extract_json(raw) == {"title": "Hello"}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result: {"title": "Hello"} -- done.'
        assert _extract_json(raw) == {"title": "Hello"}

    def test_invalid_json_raises(self):
        with pytest.raises(LLMParseError, match="Failed to parse"):
            _extract_json("not json at all")


class TestNormalizeSingle:
    """Tests for _normalize_single() -- field normalization."""

    def test_tags_normalised_to_lowercase(self):
        result = _normalize_single({"title": "T", "problem": "P", "tags": ["Python", "DJANGO"]})
        assert result["tags"] == ["python", "django"]

    def test_non_list_tags_reset(self):
        result = _normalize_single({"title": "T", "tags": "not-a-list"})
        assert result["tags"] == []
