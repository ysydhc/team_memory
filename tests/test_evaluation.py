"""Tests for EvaluationService — marker injection and was_used detection."""
from __future__ import annotations

import pytest

from team_memory.services.evaluation import EvaluationService


class TestInjectMarkers:
    """Test EvaluationService.inject_markers."""

    def test_adds_marker_to_solution_field(self):
        """inject_markers appends [mem:xxx] to the solution field."""
        svc = EvaluationService()
        results = [{"id": "exp-001", "solution": "Use async/await"}]
        out = svc.inject_markers(results)
        assert out[0]["solution"] == "Use async/await [mem:exp-001]"
        assert out[0]["_marker"] == "[mem:exp-001]"

    def test_adds_marker_to_description_when_no_solution(self):
        """When no solution field, marker goes to description."""
        svc = EvaluationService()
        results = [{"id": "exp-002", "description": "Some desc"}]
        out = svc.inject_markers(results)
        assert "solution" not in out[0]
        assert out[0]["description"] == "Some desc [mem:exp-002]"
        assert out[0]["_marker"] == "[mem:exp-002]"

    def test_does_not_duplicate_markers(self):
        """If marker already present, don't add it again."""
        svc = EvaluationService()
        results = [{"id": "exp-003", "solution": "Fix [mem:exp-003]"}]
        out = svc.inject_markers(results)
        assert out[0]["solution"] == "Fix [mem:exp-003]"
        assert out[0]["_marker"] == "[mem:exp-003]"

    def test_skips_entries_without_id(self):
        """Entries with no id are left untouched."""
        svc = EvaluationService()
        results = [{"solution": "No id here"}]
        out = svc.inject_markers(results)
        assert out[0]["solution"] == "No id here"
        assert "_marker" not in out[0]

    def test_multiple_results(self):
        """Each result gets its own marker based on its id."""
        svc = EvaluationService()
        results = [
            {"id": "a1", "solution": "Sol A"},
            {"id": "b2", "description": "Desc B"},
        ]
        out = svc.inject_markers(results)
        assert out[0]["solution"] == "Sol A [mem:a1]"
        assert out[0]["_marker"] == "[mem:a1]"
        assert out[1]["description"] == "Desc B [mem:b2]"
        assert out[1]["_marker"] == "[mem:b2]"

    def test_empty_solution_falls_back_to_description(self):
        """Empty string solution is treated as absent; falls back to description."""
        svc = EvaluationService()
        results = [{"id": "exp-004", "solution": "", "description": "Fallback"}]
        out = svc.inject_markers(results)
        # Empty string is falsy, so marker goes to description
        assert out[0]["description"] == "Fallback [mem:exp-004]"
        assert out[0]["_marker"] == "[mem:exp-004]"

    def test_empty_results_list(self):
        """Empty list returns empty list."""
        svc = EvaluationService()
        out = svc.inject_markers([])
        assert out == []


class TestCheckWasUsed:
    """Test EvaluationService.check_was_used."""

    def test_marker_present_returns_true(self):
        """If the agent response contains the marker, was_used is True."""
        svc = EvaluationService()
        result = svc.check_was_used(
            "Here is the answer [mem:exp-001] done",
            ["exp-001"],
        )
        assert result == {"exp-001": True}

    def test_marker_absent_returns_false(self):
        """If the agent response lacks the marker, was_used is False."""
        svc = EvaluationService()
        result = svc.check_was_used(
            "No marker here",
            ["exp-001"],
        )
        assert result == {"exp-001": False}

    def test_multiple_ids(self):
        """Check multiple ids at once — some present, some absent."""
        svc = EvaluationService()
        result = svc.check_was_used(
            "Used [mem:exp-001] and [mem:exp-003]",
            ["exp-001", "exp-002", "exp-003"],
        )
        assert result == {"exp-001": True, "exp-002": False, "exp-003": True}

    def test_empty_agent_response(self):
        """Empty agent response means no markers used."""
        svc = EvaluationService()
        result = svc.check_was_used("", ["exp-001", "exp-002"])
        assert result == {"exp-001": False, "exp-002": False}

    def test_empty_ids_list(self):
        """No ids to check returns empty dict."""
        svc = EvaluationService()
        result = svc.check_was_used("Some response", [])
        assert result == {}

    def test_partial_marker_does_not_match(self):
        """A substring like [mem:exp-00] should not match [mem:exp-001]."""
        svc = EvaluationService()
        result = svc.check_was_used(
            "Partial [mem:exp-00]",
            ["exp-001"],
        )
        assert result == {"exp-001": False}


class TestCheckWasUsedFuzzy:
    """Test EvaluationService.check_was_used_fuzzy."""

    @pytest.mark.asyncio
    async def test_high_overlap_returns_true(self):
        """When keywords overlap > threshold, fuzzy returns True."""
        svc = EvaluationService()
        results = [{"id": "exp-001", "solution": "Use async await pattern"}]
        out = await svc.check_was_used_fuzzy("Use async await pattern", results)
        assert out == {"exp-001": True}

    @pytest.mark.asyncio
    async def test_low_overlap_returns_false(self):
        """When keywords overlap < threshold, fuzzy returns False."""
        svc = EvaluationService()
        results = [{"id": "exp-001", "solution": "Use async await pattern"}]
        out = await svc.check_was_used_fuzzy("Docker compose networking", results)
        assert out == {"exp-001": False}

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Empty results list returns empty dict."""
        svc = EvaluationService()
        out = await svc.check_was_used_fuzzy("response", [])
        assert out == {}


class TestGetWeeklyStats:
    """Test EvaluationService.get_weekly_stats."""

    @pytest.mark.asyncio
    async def test_default_stats(self):
        """Without DB connection, returns zeroed stats."""
        svc = EvaluationService()
        stats = await svc.get_weekly_stats()
        assert stats == {"total": 0, "hit": 0, "used": 0, "use_rate": 0.0}
