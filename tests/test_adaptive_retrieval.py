"""Tests for AdaptiveRetriever — multi-hop loop logic."""

from __future__ import annotations

import pytest

from team_memory.services.search.adaptive_retriever import (
    AdaptiveResult,
    AdaptiveRetriever,
)


class FakeLLMClient:
    """Controllable fake LLM that returns pre-configured JSON responses."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def chat_json(self, system: str, user: str, **kwargs) -> dict:
        self.calls.append((system, user))
        if self._responses:
            return self._responses.pop(0)
        return {"is_complex": False}


class FakeSearchOrchestrator:
    """Returns pre-configured search results."""

    def __init__(self, results_per_call: list[list[dict]]):
        self._results = list(results_per_call)
        self.search_count = 0

    async def search(self, query: str, **kwargs):
        self.search_count += 1
        results = self._results.pop(0) if self._results else []

        class _R:
            pass

        r = _R()
        r.results = results
        r.reranked = False
        r.cached = False
        r.intent_type = "general"
        return r


@pytest.mark.asyncio
async def test_simple_query_bypasses_loop():
    """Simple queries should use single-pass retrieval, no multi-hop."""
    llm = FakeLLMClient([{"is_complex": False, "initial_plan": ""}])
    orchestrator = FakeSearchOrchestrator(
        [
            [{"id": "e1", "title": "Result 1", "solution": "Answer", "score": 0.9}],
        ]
    )
    retriever = AdaptiveRetriever(
        search_orchestrator=orchestrator,
        llm_client=llm,
        max_loops=3,
    )
    result = await retriever.retrieve(query="Where is the style guide?", max_results=5)

    assert isinstance(result, AdaptiveResult)
    assert result.hops == 1
    assert len(result.results) == 1
    assert result.results[0]["id"] == "e1"


@pytest.mark.asyncio
async def test_complex_query_triggers_multihop():
    """Complex queries should trigger the multi-hop loop and collect results from multiple hops."""
    llm = FakeLLMClient(
        [
            # Intent evaluation: complex
            {"is_complex": True, "initial_plan": "Search A then B"},
            # Critic after hop 1: not sufficient
            {
                "is_sufficient": False,
                "missing_information": "Missing details about solution B",
                "next_queries": ["solution B details"],
            },
            # Critic after hop 2: sufficient
            {"is_sufficient": True, "missing_information": "", "next_queries": []},
        ]
    )
    orchestrator = FakeSearchOrchestrator(
        [
            # Hop 1 results
            [{"id": "e1", "title": "Problem A", "solution": "Old approach", "score": 0.9}],
            # Hop 2 results
            [
                {"id": "e1", "title": "Problem A", "solution": "Old approach", "score": 0.9},  # duplicate
                {"id": "e2", "title": "Solution B", "solution": "New approach", "score": 0.85},
            ],
        ]
    )
    retriever = AdaptiveRetriever(
        search_orchestrator=orchestrator,
        llm_client=llm,
        max_loops=3,
    )
    result = await retriever.retrieve(query="Why was plan A abandoned and how does B fix it?", max_results=5)

    assert result.hops == 2
    assert len(result.results) == 2  # e1 + e2, deduped
    assert result.results[0]["id"] == "e1"
    assert result.results[1]["id"] == "e2"
    assert "solution B details" in result.queries_used
    assert orchestrator.search_count == 2


@pytest.mark.asyncio
async def test_max_loops_respected():
    """Loop should stop at max_loops even if critic says not sufficient."""
    llm = FakeLLMClient(
        [
            {"is_complex": True, "initial_plan": "Deep search"},
            # Critic always says not sufficient
            {"is_sufficient": False, "missing_information": "Need more", "next_queries": ["q2"]},
            {"is_sufficient": False, "missing_information": "Still need more", "next_queries": ["q3"]},
        ]
    )
    orchestrator = FakeSearchOrchestrator(
        [
            [{"id": "e1", "title": "R1", "solution": "S1", "score": 0.9}],
            [{"id": "e2", "title": "R2", "solution": "S2", "score": 0.8}],
            [{"id": "e3", "title": "R3", "solution": "S3", "score": 0.7}],
        ]
    )
    retriever = AdaptiveRetriever(
        search_orchestrator=orchestrator,
        llm_client=llm,
        max_loops=3,
    )
    result = await retriever.retrieve(query="Very complex question", max_results=5)

    assert result.hops == 3
    assert len(result.results) == 3


@pytest.mark.asyncio
async def test_no_new_results_breaks_early():
    """If a hop returns no new results (all dupes), stop early."""
    llm = FakeLLMClient(
        [
            {"is_complex": True, "initial_plan": "Search"},
            {"is_sufficient": False, "missing_information": "Need X", "next_queries": ["query X"]},
        ]
    )
    orchestrator = FakeSearchOrchestrator(
        [
            [{"id": "e1", "title": "R1", "solution": "S1", "score": 0.9}],
            # Hop 2 returns only duplicates
            [{"id": "e1", "title": "R1", "solution": "S1", "score": 0.9}],
        ]
    )
    retriever = AdaptiveRetriever(
        search_orchestrator=orchestrator,
        llm_client=llm,
        max_loops=3,
    )
    result = await retriever.retrieve(query="Complex question", max_results=5)

    assert result.hops == 2
    assert len(result.results) == 1  # only e1, hop 2 added nothing
