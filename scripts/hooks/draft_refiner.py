"""DraftRefiner — extracts facts from conversation text and publishes drafts via TM API.

Orchestrates the flow:
  1. Accumulate conversation content in DraftBuffer (local SQLite).
  2. Extract key factual sentences from accumulated text (heuristic).
  3. Push the refined draft to TeamMemory via TMClient.
  4. Mark the local buffer entry as published.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Factual-keyword list (simple heuristic)
# ---------------------------------------------------------------------------

_FACT_KEYWORDS: list[str] = [
    # Chinese
    "是", "因为", "所以", "需要", "使用", "配置", "原因", "方法", "解决",
    # English
    "is", "because", "need", "use", "config",
]

_MAX_FACTS = 10

# Sentence splitter: Chinese punctuation, English period/question/exclamation,
# and newlines.
_SENTENCE_RE = re.compile(r"[。！？.!?\n]+")


class DraftRefiner:
    """Coordinates fact extraction and draft publishing.

    Args:
        tm_client: TMClient instance for calling TM MCP endpoints.
        draft_buffer: DraftBuffer instance for local SQLite persistence.
    """

    def __init__(self, tm_client: Any, draft_buffer: Any) -> None:
        self._tm = tm_client
        self._buf = draft_buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_draft(
        self,
        session_id: str,
        title: str,
        content: str,
        project: str | None = None,
        group_key: str | None = None,
    ) -> dict[str, Any]:
        """Save a draft to TM and mirror it locally in DraftBuffer.

        Args:
            session_id: Conversation / session identifier (used as local key).
            title: Draft title.
            content: Raw draft content.
            project: Optional project name.
            group_key: Optional grouping key.

        Returns:
            The TM response dict (contains at least ``id``).
        """
        tm_response = await self._tm.draft_save(
            title=title,
            content=content,
            project=project,
            group_key=group_key,
            conversation_id=session_id,
        )

        # Sync to local buffer so we can track pending drafts by session.
        await self._buf.upsert_draft(
            session_id=session_id,
            title=title,
            content=content,
            project=project,
            group_key=group_key,
        )

        return tm_response

    async def refine_and_publish(self, session_id: str) -> dict | None:
        """Refine pending drafts for *session_id* and publish to TM.

        Steps:
          1. Get pending drafts from DraftBuffer for this session.
          2. If none → return None.
          3. Extract key facts from accumulated text.
          4. Publish via tm_client.draft_publish(draft_id, refined_content).
          5. Mark local buffer entries as published.
          6. Return {"draft_id": ..., "status": "published"}.

        Args:
            session_id: The conversation / session identifier.

        Returns:
            Dict with draft_id and status, or None if nothing to publish.
        """
        pending = await self._buf.get_pending(session_id)
        if not pending:
            return None

        # Accumulate text from all pending drafts.
        accumulated = "\n".join(d.get("content", "") for d in pending)

        # Extract facts.
        facts = self.extract_facts(accumulated)
        refined_content = "\n".join(facts) if facts else accumulated

        # Use the first pending draft's id as the TM draft_id.
        first_draft = pending[0]
        draft_id = first_draft.get("id", first_draft.get("draft_id", ""))

        # Publish via TM.
        await self._tm.draft_publish(draft_id, refined_content=refined_content)

        # Mark all pending drafts for this session as published locally.
        await self._buf.mark_published_by_session(session_id)

        return {"draft_id": draft_id, "status": "published"}

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_facts(text: str) -> list[str]:
        """Extract factual sentences from *text* using keyword heuristics.

        Splits text into sentences, then keeps only those containing at
        least one factual keyword.  Returns at most ``_MAX_FACTS`` sentences.

        Args:
            text: Raw conversation / draft text.

        Returns:
            List of factual sentence strings (max 10).
        """
        if not text:
            return []

        # Split on sentence boundaries.
        raw_sentences = _SENTENCE_RE.split(text)
        # Strip whitespace and drop empties.
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        facts: list[str] = []
        for sentence in sentences:
            if any(kw in sentence for kw in _FACT_KEYWORDS):
                facts.append(sentence)
                if len(facts) >= _MAX_FACTS:
                    break

        return facts
