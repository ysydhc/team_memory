"""DraftRefiner — daemon adaptation.

Uses ConversationBuffer (structured messages) instead of old DraftBuffer.
Extracts facts from assistant messages and publishes to TM.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from team_memory.daemon.integration.tm_sink import TMSink

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factual-keyword list (simple heuristic)
# ---------------------------------------------------------------------------

_FACT_KEYWORDS: list[str] = [
    # Chinese
    "是",
    "因为",
    "所以",
    "需要",
    "使用",
    "配置",
    "原因",
    "方法",
    "解决",
    # English
    "is",
    "because",
    "need",
    "use",
    "config",
]

_MAX_FACTS = 10

# Sentence splitter: Chinese punctuation, English period/question/exclamation,
# and newlines.
_SENTENCE_RE = re.compile(r"[。！？.!?\n]+")

# ---------------------------------------------------------------------------
# Quality Gate hard format constraints (shared with RefinementWorker)
# ---------------------------------------------------------------------------

MIN_TITLE_LEN = 5
MAX_TITLE_LEN = 120
MIN_PROBLEM_LEN = 50
MIN_SOLUTION_LEN = 50
MIN_TAGS = 1

_GENERIC_TITLES: frozenset[str] = frozenset(
    {
        "调试",
        "修复",
        "配置",
        "优化",
        "更新",
        "修改",
        "排查",
        "总结",
        "经验",
        "问题",
        "无标题",
        "debug",
        "fix",
        "config",
        "update",
        "untitled",
        "draft",
    }
)

_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {
        "",
        "n/a",
        "na",
        "none",
        "null",
        "无",
        "暂无",
        "见上文",
        "同上",
        "略",
        "-",
        "todo",
    }
)

_TECH_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_.-]{1,24}\b")
_STOP_TAG_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "have",
        "when",
        "then",
        "else",
        "none",
        "null",
        "true",
        "false",
        "error",
        "issue",
        "problem",
        "solution",
        "check",
        "after",
        "before",
        "using",
        "should",
        "could",
        "would",
        "about",
        "there",
        "where",
        "which",
    }
)


def _is_placeholder(text: str | None) -> bool:
    return (text or "").strip().lower() in _PLACEHOLDER_VALUES


def is_empty_shell_draft(structured: dict[str, Any]) -> bool:
    """Return True if *structured* is an empty shell / pure-noise draft."""
    title = str(structured.get("title", "") or "").strip()
    problem = str(structured.get("problem") or structured.get("description") or "").strip()
    solution = str(structured.get("solution", "") or "").strip()

    if _is_placeholder(problem) and _is_placeholder(solution):
        return True
    if len(problem) < 20 and len(solution) < 20 and len(title) < MIN_TITLE_LEN:
        return True
    return False


def validate_structured_draft(structured: dict[str, Any]) -> tuple[bool, str]:
    """Validate hard Quality Gate format constraints on a structured draft.

    Constraints:
      - ``MIN_TITLE_LEN <= len(title) <= MAX_TITLE_LEN`` and not a generic placeholder
      - ``problem`` required, non-placeholder, ``len >= MIN_PROBLEM_LEN``
      - ``solution`` required, non-placeholder, ``len >= MIN_SOLUTION_LEN``
      - ``tags`` required, at least ``MIN_TAGS`` non-empty tag
      - ``experience_type`` required, non-empty
    """
    title = str(structured.get("title", "") or "").strip()
    problem = str(structured.get("problem") or structured.get("description") or "").strip()
    solution = str(structured.get("solution", "") or "").strip()
    exp_type = str(structured.get("experience_type", "") or "").strip()
    raw_tags = structured.get("tags")
    tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []

    if not title or len(title) < MIN_TITLE_LEN:
        return False, "discarded: title too short"
    if len(title) > MAX_TITLE_LEN:
        return False, "discarded: title too long"
    if title.lower() in _GENERIC_TITLES or title.startswith("Draft "):
        return False, "discarded: title too generic"
    if _is_placeholder(problem) or len(problem) < MIN_PROBLEM_LEN:
        return False, "discarded: problem too short"
    if _is_placeholder(solution) or len(solution) < MIN_SOLUTION_LEN:
        return False, "discarded: solution too short"
    if len(tags) < MIN_TAGS:
        return False, "discarded: tags empty"
    if not exp_type:
        return False, "discarded: experience_type empty"

    return True, ""


def _synthesize_title(problem: str, solution: str, fallback_title: str = "") -> str:
    """Derive a concise, valid title from problem/solution or non-generic draft title."""
    fb = (fallback_title or "").strip()
    if MIN_TITLE_LEN <= len(fb) <= MAX_TITLE_LEN and fb.lower() not in _GENERIC_TITLES and not fb.startswith("Draft "):
        return fb

    # Clean markdown / whitespace and take the first informative clause of problem
    clean_prob = re.sub(r"[#*`>\[\]]+", " ", problem or "")
    clean_prob = re.sub(r"\s+", " ", clean_prob).strip()
    first_clause = _SENTENCE_RE.split(clean_prob)[0].strip(" ，,;；:：-—") if clean_prob else ""

    if len(first_clause) < MIN_TITLE_LEN:
        clean_sol = re.sub(r"[#*`>\[\]]+", " ", solution or "")
        clean_sol = re.sub(r"\s+", " ", clean_sol).strip()
        sol_clause = _SENTENCE_RE.split(clean_sol)[0].strip(" ，,;；:：-—") if clean_sol else ""
        candidate = f"{first_clause} {sol_clause}".strip() if first_clause else sol_clause
    else:
        candidate = first_clause

    candidate = re.sub(r"\s+", " ", candidate).strip()
    if len(candidate) > MAX_TITLE_LEN:
        candidate = candidate[: MAX_TITLE_LEN - 1].rstrip(" ，,;；:：-—") + "…"
    return candidate


def _extract_fallback_tags(title: str, problem: str, solution: str, exp_type: str = "") -> list[str]:
    """Extract lightweight lowercase technical tags when LLM omitted tags."""
    combined = f"{title} {problem} {solution}"
    seen: list[str] = []
    for match in _TECH_TOKEN_RE.findall(combined):
        tok = match.strip("._-").lower()
        if len(tok) < 2 or tok in _STOP_TAG_WORDS or tok.isdigit():
            continue
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 5:
            break
    if not seen:
        seen.append((exp_type or "general").strip().lower() or "general")
    return seen


def auto_repair_structured_draft(
    structured: dict[str, Any],
    *,
    draft_title: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Attempt a single lightweight format auto-repair on a high-value draft.

    Only repairs drafts that already contain substantial ``problem`` and
    ``solution`` content (never fabricates content for empty-shell / noise
    drafts). Fixes:
      - missing / too-short / generic / too-long ``title``
      - empty ``tags``
      - empty ``experience_type``

    Returns:
        ``(repaired_dict, repaired_fields)`` where ``repaired_fields`` is empty
        if the draft is not eligible or no repair was applied.
    """
    problem = str(structured.get("problem") or structured.get("description") or "").strip()
    solution = str(structured.get("solution", "") or "").strip()

    # Do not fabricate content for empty shells or drafts lacking core substance
    if (
        is_empty_shell_draft(structured)
        or _is_placeholder(problem)
        or len(problem) < MIN_PROBLEM_LEN
        or _is_placeholder(solution)
        or len(solution) < MIN_SOLUTION_LEN
    ):
        return structured, []

    repaired = dict(structured)
    repaired["problem"] = problem
    repaired["description"] = problem
    repaired["solution"] = solution
    repaired_fields: list[str] = []

    title = str(repaired.get("title", "") or "").strip()
    if (
        not title
        or len(title) < MIN_TITLE_LEN
        or title.lower() in _GENERIC_TITLES
        or title.startswith("Draft ")
    ):
        new_title = _synthesize_title(problem, solution, fallback_title=draft_title)
        if MIN_TITLE_LEN <= len(new_title) <= MAX_TITLE_LEN:
            repaired["title"] = new_title
            repaired_fields.append("title")
    elif len(title) > MAX_TITLE_LEN:
        repaired["title"] = title[: MAX_TITLE_LEN - 1].rstrip(" ，,;；:：-—") + "…"
        repaired_fields.append("title")

    exp_type = str(repaired.get("experience_type", "") or "").strip()
    if not exp_type:
        lower_text = f"{problem} {solution}".lower()
        bug_keywords = ("error", "fail", "bug", "fix", "报错", "异常", "修复")
        repaired["experience_type"] = "bugfix" if any(k in lower_text for k in bug_keywords) else "general"
        repaired_fields.append("experience_type")

    raw_tags = repaired.get("tags")
    tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []
    if len(tags) < MIN_TAGS:
        repaired["tags"] = _extract_fallback_tags(
            str(repaired.get("title", "")),
            problem,
            solution,
            str(repaired.get("experience_type", "")),
        )
        repaired_fields.append("tags")

    return repaired, repaired_fields


def classify_discard_category(structured: dict[str, Any], reason: str) -> str:
    """Classify a structural rejection into ``true_noise`` vs ``format_error``.

    - ``true_noise``: empty shell / LLM explicitly returned empty fields /
      both problem and solution lack meaningful substance.
    - ``format_error``: non-trivial content existed, but failed structural
      format rules (e.g. short title/problem/solution or missing tags/type).
    """
    if is_empty_shell_draft(structured):
        return "true_noise"

    problem = str(structured.get("problem") or structured.get("description") or "").strip()
    solution = str(structured.get("solution", "") or "").strip()
    if len(problem) < 20 and len(solution) < 20:
        return "true_noise"

    return "format_error"


class DraftRefiner:
    """Coordinates fact extraction and draft publishing via TMSink.

    Args:
        sink: TMSink instance for calling TM endpoints.
        draft_buffer: ConversationBuffer instance.
    """

    def __init__(self, sink: TMSink, draft_buffer: Any) -> None:
        self._sink = sink
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
        tm_response = await self._sink.draft_save(
            title=title,
            content=content,
            project=project,
            group_key=group_key,
            conversation_id=session_id,
        )

        # Sync to local buffer so we can track pending drafts by session.
        local_id = await self._buf.upsert_draft(
            session_id=session_id,
            title=title,
            content=content,
            project=project,
            group_key=group_key,
        )

        # Write PG experience id back to buffer so refine_and_publish
        # can use the correct id instead of the local buffer id.
        # Even on error, op_draft_save may return an id if the record was
        # partially saved (e.g. embedding degraded but DB row created).
        pg_id = tm_response.get("id")
        if pg_id and local_id:
            await self._buf.set_tm_id(local_id, pg_id)
        elif not pg_id:
            logger.warning(
                "draft_save returned no id for session=%s title=%s — response=%s",
                session_id,
                title,
                tm_response,
            )

        return tm_response

    async def refine_and_publish(self, session_id: str) -> dict | None:
        """Fallback path: refine draft immediately (old behaviour).

        Now used only when LLM refinement is disabled or for session-end flush.
        Extracts keywords and publishes directly without LLM structured extraction.

        Steps:
          1. Get pending drafts from DraftBuffer for this session.
          2. If none → return None.
          3. Extract key facts from accumulated text.
          4. Publish via sink.draft_publish(draft_id, refined_content).
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

        # Use tm_id (PG experience id) if available, fallback to local id.
        first_draft = pending[0]
        draft_id = first_draft.get("tm_id") or first_draft.get("id", first_draft.get("draft_id", ""))

        # Publish via TM.
        await self._sink.draft_publish(draft_id=draft_id, refined_content=refined_content)

        # Mark all pending drafts for this session as published locally.
        await self._buf.mark_published_by_session(session_id)

        return {"draft_id": draft_id, "status": "published"}

    async def mark_for_refinement(self, session_id: str) -> dict | None:
        """Mark a pending draft as needing LLM refinement.

        Called by the pipeline when convergence is detected.
        The RefinementWorker will later pick it up, call LLM, and publish.

        Args:
            session_id: The session identifier.

        Returns:
            Dict with draft_id if successful, None if no draft found.
        """
        pending = await self._buf.get_pending(session_id)
        if not pending:
            logger.warning("No pending draft for session %s to mark for refinement", session_id)
            return None

        first_draft = pending[0]
        draft_id = first_draft.get("id", first_draft.get("draft_id", ""))
        await self._buf.mark_needs_refinement(draft_id)
        logger.info("[REFINE] Draft %s marked as needs_refinement", draft_id)
        return {"draft_id": draft_id, "status": "needs_refinement"}

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
