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

    Orchestrates: Intent Evaluation -> Search -> Critic -> (loop) -> Return.
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

        intent = await self.evaluate_intent(query)
        logger.info("[Adaptive] Query=%s is_complex=%s", query[:60], intent.is_complex)

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

        loop_count = 1
        previous_queries: list[str] = [query]

        while loop_count < self._max_loops:
            critic = await self.evaluate_critic(query, collected)
            hop_log[-1]["critic_sufficient"] = critic.is_sufficient
            hop_log[-1]["missing"] = critic.missing_information

            if critic.is_sufficient:
                logger.info("[Critic] Context sufficient after hop %d", loop_count)
                break

            if not critic.next_queries:
                logger.info("[Critic] No next queries suggested, stopping")
                break

            new_queries = [q for q in critic.next_queries if q not in previous_queries]
            if not new_queries:
                logger.info("[Critic] All suggested queries already used, stopping")
                break

            loop_count += 1
            hop_new_total = 0
            for nq in new_queries[:2]:
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
