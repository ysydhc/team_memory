"""Tests for AdaptiveRetriever — multi-hop loop logic."""
from __future__ import annotations

import pytest

from team_memory.services.search.adaptive_retriever import (
    AdaptiveRetriever,
    AdaptiveResult,
    IntentEvaluation,
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
    orchestrator = FakeSearchOrchestrator([
        [{"id": "e1", "title": "Result 1", "solution": "Answer", "score": 0.9}],
    ])
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
