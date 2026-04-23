"""Tests for was_used auto-detection — exact + fuzzy matching."""
from __future__ import annotations

import pytest

from team_memory.services.evaluation import EvaluationService


class TestCheckWasUsedExact:
    """Exact [mem:xxx] marker matching."""

    def test_exact_match_present(self):
        """agent_response contains [mem:exp-123] → was_used=True."""
        svc = EvaluationService()
        result = svc.check_was_used(
            "Here is the fix [mem:exp-123] for the issue",
            ["exp-123"],
        )
        assert result == {"exp-123": True}

    def test_exact_match_absent(self):
        """agent_response does not contain marker → was_used=False."""
        svc = EvaluationService()
        result = svc.check_was_used(
            "No marker at all",
            ["exp-123"],
        )
        assert result == {"exp-123": False}

    def test_exact_match_multiple(self):
        """Multiple ids — some present, some absent."""
        svc = EvaluationService()
        result = svc.check_was_used(
            "Used [mem:exp-1] but not exp-2",
            ["exp-1", "exp-2", "exp-3"],
        )
        assert result == {"exp-1": True, "exp-2": False, "exp-3": False}


class TestCheckWasUsedFuzzy:
    """Fuzzy keyword-overlap matching for was_used detection."""

    @pytest.mark.asyncio
    async def test_fuzzy_high_overlap_true(self):
        """Keywords overlap > 80% → was_used=True."""
        svc = EvaluationService()
        results = [
            {
                "id": "exp-001",
                "description": "use async await pattern for concurrency",
                "solution": "",
            }
        ]
        out = await svc.check_was_used_fuzzy(
            "use async await pattern for concurrency handling",
            results,
        )
        assert out == {"exp-001": True}

    @pytest.mark.asyncio
    async def test_fuzzy_low_overlap_false(self):
        """Keywords overlap < 80% → was_used=False."""
        svc = EvaluationService()
        results = [
            {
                "id": "exp-002",
                "description": "deploy docker container kubernetes helm",
                "solution": "",
            }
        ]
        out = await svc.check_was_used_fuzzy(
            "use async await pattern for concurrency",
            results,
        )
        assert out == {"exp-002": False}

    @pytest.mark.asyncio
    async def test_fuzzy_empty_results(self):
        """Empty results list returns empty dict."""
        svc = EvaluationService()
        out = await svc.check_was_used_fuzzy("some response", [])
        assert out == {}

    @pytest.mark.asyncio
    async def test_fuzzy_uses_solution_over_description(self):
        """When solution field has content, keywords are extracted from it."""
        svc = EvaluationService()
        results = [
            {
                "id": "exp-003",
                "description": "database migration strategy",
                "solution": "use alembic for database migration versioning",
            }
        ]
        out = await svc.check_was_used_fuzzy(
            "use alembic for database migration versioning",
            results,
        )
        assert out == {"exp-003": True}

    @pytest.mark.asyncio
    async def test_fuzzy_custom_threshold(self):
        """Custom threshold controls sensitivity."""
        svc = EvaluationService()
        results = [
            {
                "id": "exp-004",
                "description": "docker compose networking bridge host",
                "solution": "",
            }
        ]
        # With high threshold (0.95), partial overlap fails
        out_high = await svc.check_was_used_fuzzy(
            "docker compose networking bridge",
            results,
            threshold=0.95,
        )
        assert out_high == {"exp-004": False}

        # With low threshold (0.5), partial overlap passes
        out_low = await svc.check_was_used_fuzzy(
            "docker compose networking bridge",
            results,
            threshold=0.5,
        )
        assert out_low == {"exp-004": True}

    @pytest.mark.asyncio
    async def test_fuzzy_result_without_id_uses_empty_key(self):
        """Results missing id still produce an entry with empty-string key."""
        svc = EvaluationService()
        results = [{"description": "some keywords here", "solution": ""}]
        out = await svc.check_was_used_fuzzy("some keywords here", results)
        assert "" in out

    @pytest.mark.asyncio
    async def test_fuzzy_result_without_description_or_solution(self):
        """Result with no text fields → no keywords → was_used=False."""
        svc = EvaluationService()
        results = [{"id": "exp-005"}]
        out = await svc.check_was_used_fuzzy("any response", results)
        assert out == {"exp-005": False}


class TestCheckWasUsedMixed:
    """Mixed exact + fuzzy matching scenarios."""

    @pytest.mark.asyncio
    async def test_exact_match_and_fuzzy_match_both_present(self):
        """One result matched exactly, another by fuzzy overlap."""
        svc = EvaluationService()

        # Exact match check
        exact = svc.check_was_used(
            "Applied the fix [mem:exp-100] and also used redis cache",
            ["exp-100", "exp-200"],
        )
        assert exact == {"exp-100": True, "exp-200": False}

        # Fuzzy match for the unmatched one
        results = [
            {
                "id": "exp-200",
                "description": "redis cache invalidation strategy",
                "solution": "",
            }
        ]
        fuzzy = await svc.check_was_used_fuzzy(
            "Applied the fix and also used redis cache invalidation strategy",
            results,
        )
        assert fuzzy == {"exp-200": True}

    @pytest.mark.asyncio
    async def test_exact_match_but_fuzzy_also_applicable(self):
        """When exact match is True, fuzzy should also confirm True."""
        svc = EvaluationService()
        results = [
            {
                "id": "exp-300",
                "description": "python type hint annotation static analysis",
                "solution": "",
            }
        ]
        # The response includes the marker AND overlapping keywords
        fuzzy = await svc.check_was_used_fuzzy(
            "use python type hint annotation static analysis [mem:exp-300]",
            results,
        )
        assert fuzzy == {"exp-300": True}
