# Adaptive Multi-hop Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a loop-based adaptive multi-hop retrieval system that automatically generates sub-queries when initial results are insufficient, and integrate FaithfulnessJudge for A/B quality evaluation.

**Architecture:** A new `AdaptiveRetriever` in `services/search/` orchestrates Intent Evaluation → Search Pipeline → Critic Evaluation → Next Query Generation in a controlled loop (max 3 hops). It reuses the existing `SearchOrchestrator.search()` as its underlying retrieval engine and `LLMClient` for LLM calls. The `memory_recall` MCP tool gains an `adaptive` flag to opt in. Runtime evaluation is done asynchronously via `FaithfulnessJudge`.

**Tech Stack:** Python 3.11+, existing `LLMClient` (Ollama/OpenAI-compatible), `SearchOrchestrator`, `FaithfulnessJudge`, pytest

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/team_memory/services/search/adaptive_retriever.py` | Core multi-hop loop: Intent Evaluator, Critic, Loop Controller |
| Modify | `src/team_memory/services/memory/memory_operations.py` | Add `adaptive` param to `op_recall`, wire to `AdaptiveRetriever` |
| Modify | `src/team_memory/server.py` | Add `adaptive` param to `memory_recall` MCP tool |
| Modify | `src/team_memory/bootstrap.py` | Instantiate `AdaptiveRetriever` in `AppContext` |
| Create | `tests/test_adaptive_retrieval.py` | Unit tests for loop logic, dedup, early termination |
| Create | `tests/eval_adaptive_ab.py` | A/B evaluation script: single-pass vs multi-hop faithfulness scores |

---

### Task 1: Create `AdaptiveRetriever` with Intent Evaluator

**Files:**
- Create: `src/team_memory/services/search/adaptive_retriever.py`
- Test: `tests/test_adaptive_retrieval.py`

- [ ] **Step 1: Write the failing test for Intent Evaluator**

```python
# tests/test_adaptive_retrieval.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_retrieval.py::test_simple_query_bypasses_loop -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'team_memory.services.search.adaptive_retriever'`

- [ ] **Step 3: Implement `AdaptiveRetriever` with Intent Evaluator**

```python
# src/team_memory/services/search/adaptive_retriever.py
"""Adaptive multi-hop retriever — loop-based agentic retrieval.

Uses LLM to evaluate query complexity, then iteratively retrieves
and critiques results until sufficient context is collected.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from team_memory.services.llm.llm_client import LLMClient

logger = logging.getLogger("team_memory.adaptive_retriever")

INTENT_SYSTEM_PROMPT = (
    "You are a search query complexity classifier. "
    "Determine if the query requires multi-hop retrieval (multiple searches) "
    "or can be answered with a single search.\n\n"
    "Output JSON:\n"
    '{"is_complex": true/false, "initial_plan": "brief plan if complex, empty string if simple"}'
)

CRITIC_SYSTEM_PROMPT = (
    "You are a retrieval critic. Given a user's original query and the context "
    "retrieved so far, determine if the context is SUFFICIENT to fully answer the query.\n\n"
    "Output JSON:\n"
    "{\n"
    '  "is_sufficient": true/false,\n'
    '  "missing_information": "what is still missing (empty if sufficient)",\n'
    '  "next_queries": ["query1", "query2"] // 1-3 queries for next hop, empty if sufficient\n'
    "}"
)


@dataclass
class IntentEvaluation:
    is_complex: bool
    initial_plan: str = ""


@dataclass
class CriticEvaluation:
    is_sufficient: bool
    missing_information: str = ""
    next_queries: list[str] = field(default_factory=list)


@dataclass
class AdaptiveResult:
    results: list[dict]
    hops: int = 1
    queries_used: list[str] = field(default_factory=list)
    reranked: bool = False
    cached: bool = False
    intent_type: str = "general"
    hop_log: list[dict] = field(default_factory=list)


class AdaptiveRetriever:
    """Loop-based adaptive multi-hop retriever.

    Orchestrates: Intent Evaluation → Search → Critic → (loop) → Return.
    Reuses SearchOrchestrator for the underlying retrieval.
    """

    def __init__(
        self,
        search_orchestrator: object,
        llm_client: LLMClient,
        max_loops: int = 3,
    ) -> None:
        self._orchestrator = search_orchestrator
        self._llm = llm_client
        self._max_loops = max_loops

    async def evaluate_intent(self, query: str) -> IntentEvaluation:
        """Classify query complexity via LLM."""
        try:
            data = await self._llm.chat_json(
                system=INTENT_SYSTEM_PROMPT,
                user=f"Query: {query}",
                temperature=0.1,
            )
            return IntentEvaluation(
                is_complex=bool(data.get("is_complex", False)),
                initial_plan=data.get("initial_plan", ""),
            )
        except Exception as e:
            logger.warning("Intent evaluation failed, defaulting to simple: %s", e)
            return IntentEvaluation(is_complex=False)

    async def evaluate_critic(
        self, query: str, collected_context: list[dict]
    ) -> CriticEvaluation:
        """Evaluate whether collected context is sufficient."""
        context_summary = "\n\n".join(
            f"[{i+1}] {r.get('title', '')}: {r.get('solution', r.get('description', ''))}"
            for i, r in enumerate(collected_context)
        )
        try:
            data = await self._llm.chat_json(
                system=CRITIC_SYSTEM_PROMPT,
                user=(
                    f"Original Query: {query}\n\n"
                    f"Retrieved Context:\n{context_summary[:3000]}"
                ),
                temperature=0.1,
            )
            return CriticEvaluation(
                is_sufficient=bool(data.get("is_sufficient", True)),
                missing_information=data.get("missing_information", ""),
                next_queries=data.get("next_queries", []),
            )
        except Exception as e:
            logger.warning("Critic evaluation failed, treating as sufficient: %s", e)
            return CriticEvaluation(is_sufficient=True)

    async def retrieve(
        self,
        query: str,
        *,
        max_results: int = 10,
        tags: list[str] | None = None,
        user_name: str = "",
        project: str | None = None,
        include_archives: bool = False,
    ) -> AdaptiveResult:
        """Execute adaptive multi-hop retrieval."""
        start = time.monotonic()
        queries_used = [query]
        hop_log: list[dict] = []
        seen_ids: set[str] = set()
        collected: list[dict] = []

        # Step 1: Evaluate intent
        intent = await self.evaluate_intent(query)
        logger.info("[Adaptive] Query=%s is_complex=%s", query[:60], intent.is_complex)

        # Step 2: First hop — always execute
        search_result = await self._orchestrator.search(
            query=query,
            tags=tags,
            max_results=max_results,
            min_similarity=0.6,
            user_name=user_name,
            source="mcp",
            grouped=True,
            top_k_children=3,
            project=project,
            include_archives=include_archives,
        )
        new_items = self._merge_dedup(collected, search_result.results, seen_ids)
        hop_log.append({"hop": 1, "query": query, "new_results": len(new_items)})
        logger.info("[Hop 1] Query=%s -> %d new results", query[:60], len(new_items))

        # If simple query, return immediately
        if not intent.is_complex:
            return AdaptiveResult(
                results=collected,
                hops=1,
                queries_used=queries_used,
                reranked=search_result.reranked,
                cached=search_result.cached,
                intent_type=search_result.intent_type,
                hop_log=hop_log,
            )

        # Step 3: Multi-hop loop
        loop_count = 1
        previous_queries: list[str] = [query]

        while loop_count < self._max_loops:
            # Critic evaluation
            critic = await self.evaluate_critic(query, collected)
            hop_log[-1]["critic_sufficient"] = critic.is_sufficient
            hop_log[-1]["missing"] = critic.missing_information

            if critic.is_sufficient:
                logger.info("[Critic] Context sufficient after hop %d", loop_count)
                break

            if not critic.next_queries:
                logger.info("[Critic] No next queries suggested, stopping")
                break

            # Anti-degradation: skip queries already used
            new_queries = [q for q in critic.next_queries if q not in previous_queries]
            if not new_queries:
                logger.info("[Critic] All suggested queries already used, stopping")
                break

            # Execute next hop with each new query
            loop_count += 1
            hop_new_total = 0
            for nq in new_queries[:2]:  # cap at 2 sub-queries per hop
                queries_used.append(nq)
                previous_queries.append(nq)
                sub_result = await self._orchestrator.search(
                    query=nq,
                    tags=tags,
                    max_results=max_results,
                    min_similarity=0.6,
                    user_name=user_name,
                    source="mcp",
                    grouped=True,
                    top_k_children=3,
                    project=project,
                    include_archives=include_archives,
                )
                new_items = self._merge_dedup(collected, sub_result.results, seen_ids)
                hop_new_total += len(new_items)

            hop_log.append({
                "hop": loop_count,
                "queries": new_queries[:2],
                "new_results": hop_new_total,
            })
            logger.info(
                "[Hop %d] Queries=%s -> %d new results",
                loop_count, new_queries[:2], hop_new_total,
            )

            # Anti-degradation: no new results means we've exhausted the knowledge base
            if hop_new_total == 0:
                logger.info("[Adaptive] No new results in hop %d, stopping", loop_count)
                break

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "[Adaptive] Completed: %d hops, %d total results, %dms",
            loop_count, len(collected), duration_ms,
        )

        return AdaptiveResult(
            results=collected,
            hops=loop_count,
            queries_used=queries_used,
            reranked=search_result.reranked,
            intent_type=search_result.intent_type,
            hop_log=hop_log,
        )

    @staticmethod
    def _merge_dedup(
        collected: list[dict],
        new_results: list[dict],
        seen_ids: set[str],
    ) -> list[dict]:
        """Merge new results into collected, deduplicating by ID. Returns only new items."""
        added: list[dict] = []
        for r in new_results:
            rid = str(r.get("group_id") or r.get("id", ""))
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                collected.append(r)
                added.append(r)
        return added
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_retrieval.py::test_simple_query_bypasses_loop -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/team_memory/services/search/adaptive_retriever.py tests/test_adaptive_retrieval.py
git commit -m "feat: add AdaptiveRetriever with intent evaluator and basic test"
```

---

### Task 2: Test and verify multi-hop loop logic

**Files:**
- Modify: `tests/test_adaptive_retrieval.py`

- [ ] **Step 1: Write the failing test for multi-hop loop**

Append to `tests/test_adaptive_retrieval.py`:

```python
@pytest.mark.asyncio
async def test_complex_query_triggers_multihop():
    """Complex queries should trigger the multi-hop loop and collect results from multiple hops."""
    llm = FakeLLMClient([
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
    ])
    orchestrator = FakeSearchOrchestrator([
        # Hop 1 results
        [{"id": "e1", "title": "Problem A", "solution": "Old approach", "score": 0.9}],
        # Hop 2 results
        [
            {"id": "e1", "title": "Problem A", "solution": "Old approach", "score": 0.9},  # duplicate
            {"id": "e2", "title": "Solution B", "solution": "New approach", "score": 0.85},
        ],
    ])
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
    llm = FakeLLMClient([
        {"is_complex": True, "initial_plan": "Deep search"},
        # Critic always says not sufficient
        {"is_sufficient": False, "missing_information": "Need more", "next_queries": ["q2"]},
        {"is_sufficient": False, "missing_information": "Still need more", "next_queries": ["q3"]},
    ])
    orchestrator = FakeSearchOrchestrator([
        [{"id": "e1", "title": "R1", "solution": "S1", "score": 0.9}],
        [{"id": "e2", "title": "R2", "solution": "S2", "score": 0.8}],
        [{"id": "e3", "title": "R3", "solution": "S3", "score": 0.7}],
    ])
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
    llm = FakeLLMClient([
        {"is_complex": True, "initial_plan": "Search"},
        {"is_sufficient": False, "missing_information": "Need X", "next_queries": ["query X"]},
    ])
    orchestrator = FakeSearchOrchestrator([
        [{"id": "e1", "title": "R1", "solution": "S1", "score": 0.9}],
        # Hop 2 returns only duplicates
        [{"id": "e1", "title": "R1", "solution": "S1", "score": 0.9}],
    ])
    retriever = AdaptiveRetriever(
        search_orchestrator=orchestrator,
        llm_client=llm,
        max_loops=3,
    )
    result = await retriever.retrieve(query="Complex question", max_results=5)

    assert result.hops == 2
    assert len(result.results) == 1  # only e1, hop 2 added nothing
```

- [ ] **Step 2: Run tests to verify they fail then pass**

Run: `pytest tests/test_adaptive_retrieval.py -v`
Expected: All 4 tests PASS (implementation already handles these cases)

- [ ] **Step 3: Commit**

```bash
git add tests/test_adaptive_retrieval.py
git commit -m "test: add multi-hop loop, max_loops, and early-termination tests"
```

---

### Task 3: Wire `AdaptiveRetriever` into `memory_recall`

**Files:**
- Modify: `src/team_memory/bootstrap.py` (~3 lines)
- Modify: `src/team_memory/services/memory/memory_operations.py` (~30 lines)
- Modify: `src/team_memory/server.py` (~5 lines)

- [ ] **Step 1: Add `AdaptiveRetriever` to `AppContext` in bootstrap.py**

In `src/team_memory/bootstrap.py`, add the field to the `AppContext` dataclass:

```python
# After the line: entity_extractor: object | None = None
adaptive_retriever: object | None = None  # AdaptiveRetriever (lazy import)
```

Then find where the `AppContext` is instantiated (in the `bootstrap()` function) and add:

```python
# After search_orchestrator is created, add:
from team_memory.services.search.adaptive_retriever import AdaptiveRetriever
adaptive_retriever = AdaptiveRetriever(
    search_orchestrator=search_orchestrator,
    llm_client=LLMClient.from_config(settings.llm),
    max_loops=3,
)
```

And pass `adaptive_retriever=adaptive_retriever` to `AppContext(...)`.

- [ ] **Step 2: Add `adaptive` parameter to `op_recall` in memory_operations.py**

In `src/team_memory/services/memory/memory_operations.py`, modify `op_recall` signature to accept `adaptive: bool = False` and `adaptive_retriever: object | None = None`. When `adaptive=True` and a `query` is provided, use the adaptive path:

```python
async def op_recall(
    user: str,
    *,
    search_orchestrator: object,
    db_url: str,
    settings: object,
    embedding_provider: object,
    query: str | None = None,
    problem: str | None = None,
    file_path: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 10,
    project: str | None = None,
    include_archives: bool | None = None,
    include_user_profile: bool = False,
    adaptive: bool = False,
    adaptive_retriever: object | None = None,
) -> dict:
```

In the `if query:` branch, before calling `_recall_search`, add:

```python
    if query:
        if adaptive and adaptive_retriever is not None:
            raw = await _recall_adaptive(
                query=query,
                tags=tags,
                max_results=max_results,
                adaptive_retriever=adaptive_retriever,
                user=user,
                project=resolved_project,
                include_archives=include_archives,
            )
        else:
            raw = await _recall_search(
                # ... existing code unchanged ...
            )
```

Add the `_recall_adaptive` helper function near `_recall_search`:

```python
async def _recall_adaptive(
    *,
    query: str,
    tags: list[str] | None,
    max_results: int,
    adaptive_retriever: object,
    user: str,
    project: str,
    include_archives: bool,
) -> dict:
    """Adaptive multi-hop retrieval mode."""
    from team_memory.services.search.adaptive_retriever import AdaptiveResult

    result: AdaptiveResult = await adaptive_retriever.retrieve(
        query=query,
        tags=tags,
        max_results=max_results,
        user_name=user,
        project=project,
        include_archives=include_archives,
    )

    if not result.results:
        return {
            "message": "No matching experiences found.",
            "results": [],
            "reranked": False,
            "intent_type": result.intent_type,
            "adaptive": {"hops": result.hops, "queries_used": result.queries_used},
        }

    best_id = result.results[0].get("group_id") or result.results[0].get("id", "")
    return {
        "message": f"Found {len(result.results)} result(s) across {result.hops} hop(s).",
        "results": result.results,
        "reranked": result.reranked,
        "intent_type": result.intent_type,
        "adaptive": {
            "hops": result.hops,
            "queries_used": result.queries_used,
            "hop_log": result.hop_log,
        },
        "feedback_hint": f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)",
    }
```

- [ ] **Step 3: Add `adaptive` parameter to `memory_recall` MCP tool in server.py**

In `src/team_memory/server.py`, add `adaptive: bool = False` to the `memory_recall` function signature:

```python
async def memory_recall(
    query: str | None = None,
    problem: str | None = None,
    file_path: str | None = None,
    language: str | None = None,
    framework: str | None = None,
    tags: list[str] | None = None,
    max_results: int = 10,
    project: str | None = None,
    include_archives: bool | None = None,
    include_user_profile: bool = False,
    adaptive: bool = False,
) -> str:
```

In the `else` branch (local mode), pass the new params:

```python
        result = await memory_operations.op_recall(
            user,
            search_orchestrator=ctx.search_orchestrator,
            db_url=ctx.db_url,
            settings=ctx.settings,
            embedding_provider=ctx.embedding,
            query=query,
            problem=problem,
            file_path=file_path,
            language=language,
            framework=framework,
            tags=tags,
            max_results=max_results,
            project=resolved_project,
            include_archives=include_archives,
            include_user_profile=include_user_profile,
            adaptive=adaptive,
            adaptive_retriever=ctx.adaptive_retriever,
        )
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `pytest tests/test_server.py -v -x --timeout=60`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/team_memory/bootstrap.py src/team_memory/services/memory/memory_operations.py src/team_memory/server.py
git commit -m "feat: wire AdaptiveRetriever into memory_recall with adaptive=True flag"
```

---

### Task 4: Add runtime evaluation logging

**Files:**
- Modify: `src/team_memory/services/memory/memory_operations.py` (~15 lines)

- [ ] **Step 1: Add async faithfulness logging to adaptive recall path**

In `_recall_adaptive`, after building the result dict, add an async background task that logs the hop trace to the search log. The search log already receives data via `_maybe_write_search_log`. We enhance the result dict with the `adaptive` metadata (already done in Step 2 of Task 3), which will be persisted by the existing `_maybe_write_search_log` function.

No additional code is needed beyond what Task 3 already writes — `_maybe_write_search_log` is called in `op_recall` after the raw result is built, and the `adaptive` field will be included in the log JSON. The `hop_log` inside `result["adaptive"]` provides full observability (hop count, queries used per hop, new results per hop).

- [ ] **Step 2: Verify search log includes adaptive metadata**

Run: `pytest tests/test_adaptive_retrieval.py -v`
Expected: PASS (adaptive metadata is in the result dict)

- [ ] **Step 3: Commit**

```bash
git add src/team_memory/services/memory/memory_operations.py
git commit -m "feat: include adaptive hop metadata in search log for runtime evaluation"
```

---

### Task 5: Create A/B evaluation script

**Files:**
- Create: `tests/eval_adaptive_ab.py`

- [ ] **Step 1: Write the A/B evaluation script**

```python
# tests/eval_adaptive_ab.py
"""A/B Evaluation: Single-pass vs Adaptive Multi-hop Retrieval.

Runs complex queries through both retrieval modes and compares
FaithfulnessJudge scores to quantify the improvement.

Usage:
    python -m tests.eval_adaptive_ab [--db-url URL]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EVAL_QUERIES = [
    {
        "query": "为什么之前的搜索方案被废弃，现在的 RRF 融合方案是如何解决旧方案缺陷的？",
        "description": "Multi-hop: requires finding old approach + new approach + comparison",
    },
    {
        "query": "embedding 模型从旧版本迁移到新版本时遇到了什么问题，最终是怎么解决的？",
        "description": "Multi-hop: requires migration history + problem + solution chain",
    },
    {
        "query": "entity extraction 和 search pipeline 是如何协作的？二者的数据流是什么？",
        "description": "Multi-hop: requires understanding two subsystems and their interaction",
    },
]


async def run_single_pass(ctx, query: str, max_results: int = 10) -> dict:
    """Run standard single-pass retrieval."""
    from team_memory.services.memory.memory_operations import op_recall

    return await op_recall(
        "eval",
        search_orchestrator=ctx.search_orchestrator,
        db_url=ctx.db_url,
        settings=ctx.settings,
        embedding_provider=ctx.embedding,
        query=query,
        max_results=max_results,
        adaptive=False,
    )


async def run_adaptive(ctx, query: str, max_results: int = 10) -> dict:
    """Run adaptive multi-hop retrieval."""
    from team_memory.services.memory.memory_operations import op_recall

    return await op_recall(
        "eval",
        search_orchestrator=ctx.search_orchestrator,
        db_url=ctx.db_url,
        settings=ctx.settings,
        embedding_provider=ctx.embedding,
        query=query,
        max_results=max_results,
        adaptive=True,
        adaptive_retriever=ctx.adaptive_retriever,
    )


def format_context(results: list[dict]) -> list[str]:
    """Extract context strings from search results for FaithfulnessJudge."""
    contexts = []
    for r in results:
        title = r.get("title", "")
        solution = r.get("solution", r.get("description", ""))
        contexts.append(f"{title}: {solution}")
    return contexts


async def main():
    from team_memory.bootstrap import bootstrap, get_context
    from team_memory.daemon.quality.faithfulness_judge import FaithfulnessJudge

    bootstrap(enable_background=False)
    ctx = get_context()
    judge = FaithfulnessJudge()

    print(f"\n{'=' * 70}")
    print("  A/B Evaluation: Single-pass vs Adaptive Multi-hop Retrieval")
    print(f"{'=' * 70}\n")

    results_table = []

    for eq in EVAL_QUERIES:
        query = eq["query"]
        print(f"  Query: {query[:60]}...")

        # Run both modes
        single_result = await run_single_pass(ctx, query)
        adaptive_result = await run_adaptive(ctx, query)

        single_contexts = format_context(single_result.get("results", []))
        adaptive_contexts = format_context(adaptive_result.get("results", []))

        # Use the contexts themselves as a pseudo-response for faithfulness scoring
        single_response = "\n".join(single_contexts)
        adaptive_response = "\n".join(adaptive_contexts)

        # Score both
        single_score = await judge.evaluate(
            query=query, contexts=single_contexts, response=single_response
        )
        adaptive_score = await judge.evaluate(
            query=query, contexts=adaptive_contexts, response=adaptive_response
        )

        hops = adaptive_result.get("adaptive", {}).get("hops", 1)
        row = {
            "query": query[:50],
            "single_results": len(single_result.get("results", [])),
            "single_score": single_score.score,
            "adaptive_results": len(adaptive_result.get("results", [])),
            "adaptive_score": adaptive_score.score,
            "adaptive_hops": hops,
            "delta": round(adaptive_score.score - single_score.score, 3),
        }
        results_table.append(row)
        print(f"    Single: {row['single_results']} results, score={row['single_score']:.3f}")
        print(f"    Adaptive: {row['adaptive_results']} results ({hops} hops), score={row['adaptive_score']:.3f}")
        print(f"    Delta: {row['delta']:+.3f}")
        print()

    # Summary
    avg_single = sum(r["single_score"] for r in results_table) / len(results_table) if results_table else 0
    avg_adaptive = sum(r["adaptive_score"] for r in results_table) / len(results_table) if results_table else 0

    print(f"{'─' * 70}")
    print(f"  Average Single-pass Score:  {avg_single:.3f}")
    print(f"  Average Adaptive Score:     {avg_adaptive:.3f}")
    print(f"  Average Delta:              {avg_adaptive - avg_single:+.3f}")
    print(f"{'─' * 70}\n")

    # Save JSON report
    report = {
        "queries": results_table,
        "avg_single_score": round(avg_single, 3),
        "avg_adaptive_score": round(avg_adaptive, 3),
        "avg_delta": round(avg_adaptive - avg_single, 3),
    }
    report_path = os.path.join(os.path.dirname(__file__), "..", "eval_adaptive_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Commit**

```bash
git add tests/eval_adaptive_ab.py
git commit -m "feat: add A/B evaluation script comparing single-pass vs adaptive retrieval"
```

---

### Task 6: Final integration test and verification

**Files:**
- All files from previous tasks

- [ ] **Step 1: Run full test suite to verify no regressions**

Run: `make verify`
Expected: All lint checks pass, all tests pass

- [ ] **Step 2: Run just the adaptive retrieval tests**

Run: `pytest tests/test_adaptive_retrieval.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix: address any integration issues from adaptive retrieval"
```
