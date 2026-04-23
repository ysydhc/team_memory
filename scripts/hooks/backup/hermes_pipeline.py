"""HermesPipeline — Hermes-side memory pipeline entry point.

Hermes does not use external Cursor/Claude Code Hook mechanisms.
Instead, it calls this pipeline directly for retrieval and draft management.

The pipeline integrates three components:
  1. TMClient — for retrieval (recall, get_context) and publishing (draft_save, draft_publish)
  2. DraftBuffer — local SQLite buffer for accumulating conversation drafts
  3. ConvergenceDetector — detects when a task has reached a natural conclusion
  4. DraftRefiner — extracts facts and publishes refined drafts to TM
"""
from __future__ import annotations

from typing import Any

from convergence_detector import ConvergenceDetector
from draft_buffer import DraftBuffer
from draft_refiner import DraftRefiner
from shared import TMClient


class HermesPipeline:
    """Hermes-side memory pipeline.

    Hermes does not use Cursor/Claude Code Hook mechanisms,
    but directly calls TMClient for retrieval and writing.

    Usage::

        pipeline = HermesPipeline("http://localhost:3900")
        async with pipeline._buffer:
            result = await pipeline.on_turn_start("之前的问题", project="team_doc")
            ...
            result = await pipeline.on_turn_end("sess-1", "解决了", project="team_doc")
            ...
            result = await pipeline.on_session_end("sess-1")
    """

    # Keywords that trigger full retrieval in on_turn_start
    _RETRIEVAL_KEYWORDS: list[str] = [
        "之前", "上次", "经验", "踩坑", "remember", "previously",
    ]

    def __init__(self, tm_url: str, draft_db_path: str | None = None) -> None:
        self._tm = TMClient(tm_url)
        self._buffer = DraftBuffer(draft_db_path or ":memory:")
        self._detector = ConvergenceDetector()
        self._refiner = DraftRefiner(self._tm, self._buffer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def on_turn_start(
        self,
        user_message: str,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Called at the start of each conversation turn for auto-retrieval.

        Similar to sessionStart Hook, but called internally by Hermes.

        If the user message contains any retrieval keyword, a full recall
        is performed. Otherwise, only project-level context is retrieved.

        Args:
            user_message: The user's input message for this turn.
            project: Optional project name to scope the retrieval.

        Returns:
            Dict with:
              - "action": "full_retrieval" or "project_context"
              - "results" or "context": the retrieval payload
        """
        should_retrieve = any(
            kw in user_message.lower() if kw.isascii() else kw in user_message
            for kw in self._RETRIEVAL_KEYWORDS
        )

        if not should_retrieve:
            # Always do a project-level context retrieval
            context = await self._tm.get_context(project=project)
            return {"action": "project_context", "context": context}

        # Keyword triggered → full retrieval
        results = await self._tm.recall(query=user_message, project=project)
        return {"action": "full_retrieval", "results": results}

    async def on_turn_end(
        self,
        session_id: str,
        agent_response: str,
        project: str | None = None,
        recent_tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Called at the end of each conversation turn to update drafts.

        Similar to afterAgentResponse Hook, but called internally by Hermes.

        Accumulates agent response text into the draft buffer, checks for
        convergence, and either saves the draft or publishes it.

        Args:
            session_id: The conversation / session identifier.
            agent_response: The agent's response text for this turn.
            project: Optional project name.
            recent_tools: Optional list of recent tool invocations for
                convergence detection.

        Returns:
            Dict with:
              - "action": "published" or "draft_saved"
              - "result": the result from DraftRefiner
        """
        # 1. Check for existing pending drafts and accumulate text
        existing = await self._buffer.get_pending(session_id)
        if existing:
            accumulated = existing[0].get("content", "") + "\n" + agent_response
        else:
            accumulated = agent_response

        # 2. Check for convergence
        converged = self._detector.detect_convergence(
            agent_response,
            recent_tools=recent_tools,
            current_path=project,
        )

        if converged and existing:
            # Converged with existing draft → publish
            result = await self._refiner.refine_and_publish(session_id)
            return {"action": "published", "result": result}
        else:
            # Not converged or no existing draft → save draft
            title = f"Session {session_id[:8]} draft"
            result = await self._refiner.save_draft(
                session_id, title, accumulated, project=project,
            )
            return {"action": "draft_saved", "result": result}

    async def on_session_end(self, session_id: str) -> dict[str, Any] | None:
        """Force-publish any remaining unpublished drafts at session end.

        Args:
            session_id: The conversation / session identifier.

        Returns:
            The result dict from refine_and_publish, or None if no pending
            drafts exist.
        """
        existing = await self._buffer.get_pending(session_id)
        if not existing:
            return None
        return await self._refiner.refine_and_publish(session_id)
