"""Tests for MCP Server tool registration, namespace, and functionality.

Tests that all tools use the tm_ namespace, are properly registered,
and that new workflow tools (tm_solve, tm_learn, tm_suggest) and
token budget guard work correctly.

Full integration tests (with real DB) are in test_integration.py.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.server import _guard_output, mcp

# ============================================================
# C4: Namespace tests — all tools must use tm_ prefix
# ============================================================


class TestToolNamespace:
    """Verify that all tools follow the tm_ naming convention."""

    @pytest.mark.asyncio
    async def test_all_tools_have_tm_prefix(self):
        """Every registered tool name must start with 'tm_'."""
        tools = await mcp.get_tools()
        for name in tools:
            assert name.startswith("tm_"), f"Tool '{name}' does not use tm_ prefix"

    @pytest.mark.asyncio
    async def test_expected_tools_registered(self):
        """All expected tools should be present."""
        tools = await mcp.get_tools()
        expected = {
            "tm_search", "tm_save", "tm_save_group",
            "tm_feedback", "tm_update",
            "tm_solve", "tm_learn", "tm_suggest",
        }
        for name in expected:
            assert name in tools, f"Tool '{name}' not registered"

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions(self):
        tools = await mcp.get_tools()
        for name, tool in tools.items():
            assert tool.description, f"Tool {name} has no description"

    @pytest.mark.asyncio
    async def test_tool_descriptions_have_token_hints(self):
        """Workflow tools should mention approximate token output size."""
        tools = await mcp.get_tools()
        for name in ("tm_search", "tm_solve", "tm_suggest", "tm_save",
                      "tm_save_group", "tm_feedback", "tm_update"):
            desc = tools[name].description or ""
            assert "token" in desc.lower(), (
                f"Tool '{name}' description should mention token output size"
            )


# ============================================================
# Resource registration
# ============================================================


class TestMCPResourceRegistration:
    """Verify that all expected resources are registered."""

    @pytest.mark.asyncio
    async def test_recent_resource_registered(self):
        resources = await mcp.get_resources()
        assert "experiences://recent" in resources

    @pytest.mark.asyncio
    async def test_stats_resource_registered(self):
        resources = await mcp.get_resources()
        assert "experiences://stats" in resources


# ============================================================
# C3: Token budget guard tests
# ============================================================


class TestGuardOutput:
    """Test the _guard_output token budget enforcement."""

    def test_small_output_unchanged(self):
        """Output within budget should pass through unchanged."""
        data = {"message": "ok", "results": [{"title": "A"}]}
        raw = json.dumps(data)
        result = _guard_output(raw, max_tokens=5000)
        assert json.loads(result) == data

    def test_large_output_truncates_results(self):
        """Output exceeding budget should remove trailing results."""
        results = [
            {"title": f"Experience {i}", "solution": "x" * 3000}
            for i in range(10)
        ]
        data = {"message": "Found 10", "results": results}
        raw = json.dumps(data, ensure_ascii=False)

        result = _guard_output(raw, max_tokens=500)
        parsed = json.loads(result)
        assert len(parsed["results"]) < 10
        assert parsed.get("truncated") is True

    def test_truncates_long_solution_field(self):
        """Long solution fields should be truncated."""
        results = [{"title": "A", "solution": "x" * 5000}]
        data = {"message": "ok", "results": results}
        raw = json.dumps(data)

        with patch("team_memory.server._get_settings") as mock_settings:
            mock_cfg = MagicMock()
            mock_cfg.mcp.max_output_tokens = 4000
            mock_cfg.mcp.truncate_solution_at = 100
            mock_cfg.mcp.include_code_snippets = True
            mock_settings.return_value = mock_cfg

            result = _guard_output(raw, max_tokens=4000)
            parsed = json.loads(result)
            assert "[truncated]" in parsed["results"][0]["solution"]
            assert parsed["truncated"] is True

    def test_removes_low_confidence_first(self):
        """Low-confidence results should be removed before truncating fields."""
        results = [
            {"title": "A", "confidence": "high", "solution": "good answer"},
            {"title": "B", "confidence": "medium", "solution": "ok answer"},
            {"title": "C", "confidence": "low", "solution": "x" * 5000},
        ]
        data = {"message": "Found 3", "results": results}
        raw = json.dumps(data)

        result = _guard_output(raw, max_tokens=200)
        parsed = json.loads(result)
        # Low confidence result should be removed
        titles = [r["title"] for r in parsed["results"]]
        assert "C" not in titles

    def test_no_results_passthrough(self):
        """Output with no results should pass through."""
        data = {"message": "No matches", "results": []}
        raw = json.dumps(data)
        result = _guard_output(raw, max_tokens=100)
        assert json.loads(result) == data

    def test_strips_code_snippets_when_disabled(self):
        """code_snippets should be removed when include_code_snippets=False."""
        results = [{"title": "A", "code_snippets": "print('hi')", "solution": "ok"}]
        data = {"message": "ok", "results": results}
        raw = json.dumps(data)

        with patch("team_memory.server._get_settings") as mock_settings:
            mock_cfg = MagicMock()
            mock_cfg.mcp.max_output_tokens = 100
            mock_cfg.mcp.truncate_solution_at = 5000
            mock_cfg.mcp.include_code_snippets = False
            mock_settings.return_value = mock_cfg

            result = _guard_output(raw, max_tokens=100)
            parsed = json.loads(result)
            assert "code_snippets" not in parsed["results"][0]


# ============================================================
# tm_search (renamed from search_experiences)
# ============================================================


class TestTdSearch:
    """Test tm_search tool function."""

    @pytest.mark.asyncio
    async def test_search_returns_json(self):
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "title": "Test experience",
                "description": "Test problem",
                "solution": "Test solution",
                "tags": ["python"],
                "similarity": 0.85,
            }
        ]

        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            search_fn = tools["tm_search"].fn
            result = await search_fn(query="test problem")

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Test experience"

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            search_fn = tools["tm_search"].fn
            result = await search_fn(query="nonexistent")

        data = json.loads(result)
        assert data["results"] == []
        assert "No matching" in data["message"]


# ============================================================
# tm_save (renamed from save_experience)
# ============================================================


class TestTdSave:
    """Test tm_save tool function."""

    @pytest.mark.asyncio
    async def test_save_returns_json(self):
        exp_id = str(uuid.uuid4())
        mock_result = {
            "id": exp_id,
            "title": "Fix Docker issue",
            "created_at": "2026-01-01T00:00:00+00:00",
        }

        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session, \
             patch("team_memory.server._get_current_user", return_value="alice"):

            mock_service = MagicMock()
            mock_service.save = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            save_fn = tools["tm_save"].fn
            result = await save_fn(
                title="Fix Docker issue",
                problem="Container won't start",
                solution="Check port conflicts",
                tags=["docker"],
            )

        data = json.loads(result)
        assert "experience" in data
        assert data["experience"]["id"] == exp_id
        assert "saved successfully" in data["message"].lower()


# ============================================================
# tm_feedback (renamed from feedback_experience)
# ============================================================


class TestTdFeedback:
    """Test tm_feedback tool function."""

    @pytest.mark.asyncio
    async def test_feedback_success(self):
        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session, \
             patch("team_memory.server._get_current_user", return_value="alice"):

            mock_service = MagicMock()
            mock_service.feedback = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            feedback_fn = tools["tm_feedback"].fn
            result = await feedback_fn(
                experience_id=str(uuid.uuid4()),
                rating=5,
                comment="Worked great!",
            )

        data = json.loads(result)
        assert "recorded" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_feedback_invalid_rating(self):
        """Rating outside 1-5 should return error."""
        tools = await mcp.get_tools()
        feedback_fn = tools["tm_feedback"].fn
        result = await feedback_fn(
            experience_id=str(uuid.uuid4()),
            rating=0,
        )
        data = json.loads(result)
        assert "error" in data


# ============================================================
# C1-a: tm_solve tests
# ============================================================


class TestTdSolve:
    """Test tm_solve workflow tool."""

    @pytest.mark.asyncio
    async def test_solve_returns_results_and_marks_used(self):
        """tm_solve should search, format, and increment use_count."""
        exp_id = str(uuid.uuid4())
        mock_results = [
            {
                "group_id": exp_id,
                "score": 0.88,
                "parent": {"id": exp_id, "title": "Docker fix", "solution": "..."},
                "children": [],
                "total_children": 0,
            }
        ]

        mock_repo = MagicMock()
        mock_repo.increment_use_count = AsyncMock()

        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session, \
             patch("team_memory.storage.repository.ExperienceRepository", return_value=mock_repo):

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            solve_fn = tools["tm_solve"].fn
            result = await solve_fn(
                problem="Docker container won't start",
                language="python",
            )

        data = json.loads(result)
        assert len(data["results"]) == 1
        assert "tm_feedback" in data["message"]
        # use_count should have been incremented
        mock_repo.increment_use_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_no_results_suggests_save(self):
        """tm_solve with no results should suggest tm_save."""
        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            solve_fn = tools["tm_solve"].fn
            result = await solve_fn(problem="Unique unseen problem")

        data = json.loads(result)
        assert data["results"] == []
        assert data["suggestion"] == "tm_save"

    @pytest.mark.asyncio
    async def test_solve_builds_enhanced_query(self):
        """tm_solve should combine problem + language + framework into query."""
        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            solve_fn = tools["tm_solve"].fn
            await solve_fn(
                problem="API error",
                language="python",
                framework="fastapi",
                file_path="src/api/routes.py",
            )

        # Check that search was called with enriched query
        call_args = mock_service.search.call_args
        query = call_args.kwargs.get("query", "")
        assert "API error" in query
        assert "python" in query
        assert "fastapi" in query
        assert "routes.py" in query


# ============================================================
# C1-b: tm_learn tests
# ============================================================


class TestTdLearn:
    """Test tm_learn workflow tool."""

    @pytest.mark.asyncio
    async def test_learn_single_experience(self):
        """tm_learn should parse and save a single experience."""
        parsed = {
            "title": "Fix database timeout",
            "problem": "Queries were timing out",
            "solution": "Added connection pooling",
            "tags": ["database", "performance"],
            "root_cause": "Too many connections",
            "language": "python",
            "framework": None,
            "code_snippets": None,
        }
        saved = {
            "id": str(uuid.uuid4()),
            "title": "Fix database timeout",
            "tags": ["database", "performance"],
            "created_at": "2026-01-01T00:00:00",
        }

        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server._get_settings") as mock_get_settings, \
             patch("team_memory.server.get_session") as mock_get_session, \
             patch("team_memory.services.llm_parser.parse_content", new_callable=AsyncMock, return_value=parsed):

            mock_service = MagicMock()
            mock_service.save = AsyncMock(return_value=saved)
            mock_get_service.return_value = mock_service

            mock_settings_obj = MagicMock()
            mock_settings_obj.llm = MagicMock()
            mock_get_settings.return_value = mock_settings_obj

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            learn_fn = tools["tm_learn"].fn
            result = await learn_fn(
                conversation="We fixed the database timeout issue by adding pooling...",
            )

        data = json.loads(result)
        assert "experience" in data
        assert data["experience"]["title"] == "Fix database timeout"
        mock_service.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_learn_parse_error(self):
        """tm_learn should handle LLM parse failures gracefully."""
        from team_memory.services.llm_parser import LLMParseError

        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server._get_settings") as mock_get_settings, \
             patch("team_memory.services.llm_parser.parse_content", new_callable=AsyncMock,
                   side_effect=LLMParseError("Connection refused")):

            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            mock_settings_obj = MagicMock()
            mock_settings_obj.llm = MagicMock()
            mock_get_settings.return_value = mock_settings_obj

            tools = await mcp.get_tools()
            learn_fn = tools["tm_learn"].fn
            result = await learn_fn(conversation="some text")

        data = json.loads(result)
        assert data.get("error") is True
        assert "Connection refused" in data["message"]


# ============================================================
# C2: tm_suggest tests
# ============================================================


class TestTdSuggest:
    """Test tm_suggest context-based recommendation tool."""

    @pytest.mark.asyncio
    async def test_suggest_builds_query_from_context(self):
        """tm_suggest should build a query from file_path + language + framework."""
        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            suggest_fn = tools["tm_suggest"].fn
            await suggest_fn(
                file_path="tests/test_auth.py",
                language="python",
                framework="pytest",
            )

        call_args = mock_service.search.call_args
        query = call_args.kwargs.get("query", "")
        # Should include directory hint + filename + language
        assert "test" in query.lower()
        assert "python" in query.lower()

    @pytest.mark.asyncio
    async def test_suggest_no_context_error(self):
        """tm_suggest with no context params should return error."""
        tools = await mcp.get_tools()
        suggest_fn = tools["tm_suggest"].fn
        result = await suggest_fn()

        data = json.loads(result)
        assert data["results"] == []
        assert "No context" in data["message"]

    @pytest.mark.asyncio
    async def test_suggest_returns_lightweight_format(self):
        """tm_suggest results should be lightweight (no full solution text)."""
        mock_results = [
            {
                "group_id": str(uuid.uuid4()),
                "score": 0.75,
                "confidence": "high",
                "parent": {
                    "id": str(uuid.uuid4()),
                    "title": "Auth testing guide",
                    "tags": ["pytest", "auth"],
                    "solution": "Very long solution text..." * 100,
                },
                "children": [],
                "total_children": 2,
            }
        ]

        with patch("team_memory.server._get_service") as mock_get_service, \
             patch("team_memory.server.get_session") as mock_get_session:

            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_results)
            mock_get_service.return_value = mock_service

            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await mcp.get_tools()
            suggest_fn = tools["tm_suggest"].fn
            result = await suggest_fn(error_message="AssertionError in test_auth")

        data = json.loads(result)
        assert len(data["results"]) == 1
        suggestion = data["results"][0]
        # Should have lightweight fields
        assert "title" in suggestion
        assert "tags" in suggestion
        assert "score" in suggestion
        assert "id" in suggestion
        # Should NOT have full solution text
        assert "solution" not in suggestion


# ============================================================
# LLM Parser tests
# ============================================================


class TestLLMParser:
    """Test the shared llm_parser module."""

    def test_extract_json_plain(self):
        from team_memory.services.llm_parser import _extract_json
        result = _extract_json('{"title": "test", "problem": "p"}')
        assert result["title"] == "test"

    def test_extract_json_with_markdown_wrapper(self):
        from team_memory.services.llm_parser import _extract_json
        text = '```json\n{"title": "test"}\n```'
        result = _extract_json(text)
        assert result["title"] == "test"

    def test_extract_json_with_prefix(self):
        from team_memory.services.llm_parser import _extract_json
        text = 'Here is the result: {"title": "test"}'
        result = _extract_json(text)
        assert result["title"] == "test"

    def test_extract_json_invalid_raises(self):
        from team_memory.services.llm_parser import LLMParseError, _extract_json
        with pytest.raises(LLMParseError):
            _extract_json("not json at all")

    def test_normalize_single(self):
        from team_memory.services.llm_parser import _normalize_single
        result = _normalize_single({
            "title": " Fix bug ",
            "problem": "crash",
            "solution": "patch",
            "tags": ["Python", " Docker "],
            "root_cause": "",
            "language": "python",
        })
        assert result["title"] == "Fix bug"
        assert result["root_cause"] is None
        assert result["tags"] == ["python", "docker"]

    def test_normalize_group(self):
        from team_memory.services.llm_parser import _normalize_group
        result = _normalize_group({
            "parent": {"title": "Parent", "problem": "p", "solution": "s", "tags": []},
            "children": [
                {"title": "Child1", "problem": "cp", "solution": "cs", "tags": ["go"]},
            ],
        })
        assert result["parent"]["title"] == "Parent"
        assert len(result["children"]) == 1
        assert result["children"][0]["tags"] == ["go"]
