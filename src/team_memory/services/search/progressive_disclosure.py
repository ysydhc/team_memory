"""Progressive disclosure (L0/L1/L2) & token budget telemetry for search results.

Implements two-tier progressive disclosure on MCP recall payloads:
- Top-K results (default K=2) are returned with full L2 details (`disclosure_level="full"`).
- Lower-ranked results (rank > K) are compacted into L1 summary cards (`disclosure_level="summary"`)
  with a compact summary, truncated solution, collapsed children metadata, and an `expand_hint`.
"""

from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from team_memory.utils.tokenizer import estimate_tokens

_CORE_METADATA_KEYS: tuple[str, ...] = (
    "id",
    "group_id",
    "title",
    "project",
    "tags",
    "score",
    "similarity",
    "confidence",
    "type",
    "status",
    "exp_status",
    "source",
    "matched_Archive_ids",
    "archive_ids",
    "feedback_hint",
    "utility_rate",
    "popularity_boost",
    "is_superseded",
    "superseded_marker",
    "supersedes_id",
    "evolution_marker",
    "valid_from",
    "valid_until",
    "superseded_by_id",
    "cross_project",
)

WORKING_SET_NOTE = "[已在当前会话加载 L2 / Already in Working Set]"


class SessionWorkingSet:
    """Tracks experience IDs already loaded at L2 full detail within a session (Letta/MemGPT paging)."""

    def __init__(
        self,
        *,
        max_sessions: int = 256,
        max_ids_per_session: int = 128,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._max_sessions = max(1, max_sessions)
        self._max_ids_per_session = max(1, max_ids_per_session)
        self._ttl_seconds = ttl_seconds
        self._sessions: OrderedDict[str, tuple[OrderedDict[str, float], float]] = OrderedDict()
        self._injected_personas: set[str] = set()
        self._last_query_entities: set[str] = set()
        self._same_topic_streak: int = 0
        self._last_active: float = time.monotonic()
        self._session_personas: OrderedDict[str, tuple[set[str], float]] = OrderedDict()
        self._session_entities: OrderedDict[str, tuple[set[str], int, float]] = OrderedDict()

    def _maybe_expire(self) -> None:
        now = time.monotonic()
        if (now - self._last_active) > self._ttl_seconds:
            self._injected_personas.clear()
            self._last_query_entities.clear()
            self._same_topic_streak = 0
        self._last_active = now
        self._prune_expired(now)

    def _prune_expired(self, now: float) -> None:
        expired = [
            sid
            for sid, (_, ts) in self._sessions.items()
            if (now - ts) > self._ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        expired_personas = [
            sid
            for sid, (_, ts) in self._session_personas.items()
            if (now - ts) > self._ttl_seconds
        ]
        for sid in expired_personas:
            self._session_personas.pop(sid, None)
        expired_entities = [
            sid
            for sid, (_, _, ts) in self._session_entities.items()
            if (now - ts) > self._ttl_seconds
        ]
        for sid in expired_entities:
            self._session_entities.pop(sid, None)

    def has_persona(self, persona: str = "general", session_id: str | None = None) -> bool:
        """Return True if personal memory for this persona has already been injected in this session."""
        self._maybe_expire()
        key = (persona or "general").strip().lower()
        if session_id is not None:
            sid = session_id.strip()
            if sid in self._session_personas and key in self._session_personas[sid][0]:
                return True
            if persona in self._session_personas and sid.lower() in self._session_personas[persona][0]:
                return True
            return False
        return key in self._injected_personas

    def record_persona(self, persona: str = "general", session_id: str | None = None) -> None:
        """Mark personal memory for this persona as injected in the current session."""
        self._maybe_expire()
        key = (persona or "general").strip().lower()
        self._injected_personas.add(key)
        if session_id is not None:
            sid = session_id.strip()
            if sid:
                personas, _ = self._session_personas.get(sid, (set(), self._last_active))
                personas.add(key)
                self._session_personas[sid] = (personas, self._last_active)
            if persona:
                alt_personas, _ = self._session_personas.get(persona, (set(), self._last_active))
                alt_personas.add(sid.lower())
                self._session_personas[persona] = (alt_personas, self._last_active)

    @property
    def last_query_entities(self) -> set[str]:
        self._maybe_expire()
        return set(self._last_query_entities)

    def get_last_query_entities(self, session_id: str | None = None) -> set[str]:
        """Return a copy of the tracked entities for the current/specified session."""
        self._maybe_expire()
        if session_id and session_id in self._session_entities:
            return set(self._session_entities[session_id][0])
        return set(self._last_query_entities)

    def check_and_update_topic_entities(
        self,
        entities: set[str] | list[str],
        overlap_threshold: float = 0.8,
        max_streak: int = 3,
        *,
        session_id: str | None = None,
    ) -> bool:
        """Check if incoming query entities have high overlap with the last turn's entities.

        Returns True if short-circuit is eligible (overlap >= overlap_threshold and streak < max_streak).
        Updates internal entity tracking and streak counter.
        """
        self._maybe_expire()
        normalized = {str(e).strip().lower() for e in (entities or []) if str(e).strip()}
        if not normalized:
            self._same_topic_streak = 0
            if session_id:
                self._session_entities.pop(session_id, None)
            return False

        prev_entities = self._last_query_entities
        streak = self._same_topic_streak
        if session_id and session_id in self._session_entities:
            prev_entities, streak, _ = self._session_entities[session_id]

        if not prev_entities:
            self._last_query_entities = normalized
            self._same_topic_streak = 0
            if session_id:
                self._session_entities[session_id] = (normalized, 0, self._last_active)
            return False

        overlap = len(normalized & prev_entities) / max(len(normalized), 1)
        if overlap >= overlap_threshold and streak < max_streak:
            streak += 1
            self._same_topic_streak = streak
            merged = prev_entities | normalized
            self._last_query_entities = merged
            if session_id:
                self._session_entities[session_id] = (merged, streak, self._last_active)
            return True

        # Reset streak (either topic shifted or max_streak reached to force fresh full search)
        self._last_query_entities = normalized
        self._same_topic_streak = 0
        if session_id:
            self._session_entities[session_id] = (normalized, 0, self._last_active)
        return False

    def has_l2(self, session_id: str, exp_id: str) -> bool:
        if not session_id or not exp_id:
            return False
        now = time.monotonic()
        self._prune_expired(now)
        entry = self._sessions.get(session_id)
        if entry is None:
            return False
        id_map, _ = entry
        return str(exp_id) in id_map

    def record_l2(self, session_id: str, exp_ids: Iterable[str]) -> None:
        if not session_id:
            return
        now = time.monotonic()
        self._prune_expired(now)
        if session_id in self._sessions:
            id_map, _ = self._sessions.pop(session_id)
        else:
            id_map = OrderedDict()
        for raw_id in exp_ids:
            eid = str(raw_id).strip()
            if not eid:
                continue
            id_map.pop(eid, None)
            id_map[eid] = now
            while len(id_map) > self._max_ids_per_session:
                id_map.popitem(last=False)
        self._sessions[session_id] = (id_map, now)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    def evict(self, session_id: str, exp_id: str) -> None:
        if not session_id or not exp_id:
            return
        entry = self._sessions.get(session_id)
        if entry is not None:
            entry[0].pop(str(exp_id), None)

    def get_loaded_ids(self, session_id: str) -> set[str]:
        if not session_id:
            return set()
        now = time.monotonic()
        self._prune_expired(now)
        entry = self._sessions.get(session_id)
        if entry is None:
            return set()
        return set(entry[0].keys())

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._sessions.clear()
            self._injected_personas.clear()
            self._last_query_entities.clear()
            self._same_topic_streak = 0
            self._session_personas.clear()
            self._session_entities.clear()
        else:
            self._sessions.pop(session_id, None)
            self._session_personas.pop(session_id, None)
            self._session_entities.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        """Clear working set state for a specific session."""
        self.clear(session_id)


_default_working_set = SessionWorkingSet()


def get_session_working_set(session_id: str | None = None) -> SessionWorkingSet:
    """Return the process-wide default SessionWorkingSet."""
    _ = session_id
    return _default_working_set


def estimate_payload_tokens(items: list[dict[str, Any]]) -> int:
    """Estimate the token footprint of serialized search result items."""
    if not items:
        return 0
    return estimate_tokens(json.dumps(items, ensure_ascii=False, default=str))


def _compact_to_l1_summary(
    item: dict[str, Any],
    *,
    max_summary_chars: int = 160,
) -> dict[str, Any]:
    """Compact a full L2 result item into an L1 summary card."""
    card: dict[str, Any] = {}
    for key in _CORE_METADATA_KEYS:
        if key in item:
            card[key] = copy.deepcopy(item[key])

    raw_sol = str(item.get("solution") or "")
    raw_desc = str(item.get("description") or "")
    base_text = raw_sol or raw_desc

    if len(base_text) > max_summary_chars:
        card["summary"] = base_text[:max_summary_chars] + "..."
    else:
        card["summary"] = base_text

    # Compact solution and description without duplicating both long fields
    sol_preview_limit = min(max_summary_chars, 96)
    has_long_solution = (
        "solution" in item and item["solution"] is not None and len(raw_sol) > sol_preview_limit
    )
    if "solution" in item and item["solution"] is not None:
        if len(raw_sol) > sol_preview_limit:
            card["solution"] = raw_sol[:sol_preview_limit] + "... [L1 summary]"
        else:
            card["solution"] = item["solution"]

    if "description" in item and item["description"] is not None and not has_long_solution:
        if len(raw_desc) > sol_preview_limit:
            card["description"] = raw_desc[:sol_preview_limit] + "..."
        else:
            card["description"] = item["description"]

    children = item.get("children")
    if isinstance(children, list) and len(children) > 0:
        card["children_count"] = len(children)
        card["children_titles"] = [
            str(c.get("title", "")) for c in children[:3] if isinstance(c, dict)
        ]

    title = item.get("title", "")
    card["expand_hint"] = f"Call memory_recall(query='{title}', max_results=1) for full L2 details"
    card["disclosure_level"] = "summary"
    return card


def apply_progressive_disclosure(
    results: list[dict[str, Any]],
    *,
    detail_level: str = "progressive",
    full_top_k: int = 2,
    max_summary_chars: int = 160,
    session_id: str | None = None,
    working_set: SessionWorkingSet | None = None,
    force_l2: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply progressive disclosure to search results and compute token telemetry.

    Supports Letta/MemGPT-style SessionWorkingSet paging when ``session_id`` is provided:
    - Experiences already loaded at L2 in the same session are automatically folded
      into L1 summaries with ``working_set_note`` (`[已在当前会话加载 L2 / Already in Working Set]`),
      promoting the next unconsumed experience into the L2 slot.
    - ``force_l2=True`` (or ``detail_level="full"``) forces Page-in of L2 content even if
      previously loaded in the session.

    Args:
        results: Raw search result dictionaries.
        detail_level: One of "progressive" (default), "full", or "summary".
        full_top_k: Number of top results that retain full L2 fields in "progressive" mode.
        max_summary_chars: Maximum character length for L1 summary fields.
        session_id: Optional conversation/session identifier for Working Set L2 deduplication.
        working_set: Optional custom SessionWorkingSet instance (defaults to module singleton).
        force_l2: If True, bypass Working Set folding and force L2 Page-in for top-K items.

    Returns:
        Tuple of (disclosed_results, telemetry_dict).
    """
    active_session_id = (
        session_id.strip()
        if isinstance(session_id, str) and session_id.strip() and session_id.strip() != "unknown"
        else None
    )
    ws = (working_set or _default_working_set) if active_session_id else None

    if not results:
        empty_telemetry: dict[str, Any] = {
            "detail_level": detail_level,
            "full_tokens": 0,
            "returned_tokens": 0,
            "saved_tokens_pct": 0.0,
        }
        if ws is not None and active_session_id is not None:
            empty_telemetry["working_set_paged_out"] = 0
            empty_telemetry["working_set_size"] = len(ws.get_loaded_ids(active_session_id))
        return [], empty_telemetry

    full_tokens = estimate_payload_tokens(results)
    disclosed_results: list[dict[str, Any]] = []
    l2_emitted = 0
    paged_out_count = 0
    newly_loaded_l2_ids: list[str] = []

    for idx, raw_item in enumerate(results):
        exp_id = str(raw_item.get("group_id") or raw_item.get("id") or "").strip()

        if detail_level == "full":
            item_copy = copy.deepcopy(raw_item)
            item_copy["disclosure_level"] = "full"
            if ws is not None and exp_id:
                item_copy["working_set_status"] = "paged_in"
                newly_loaded_l2_ids.append(exp_id)
            disclosed_results.append(item_copy)
            continue

        if detail_level == "summary":
            card = _compact_to_l1_summary(raw_item, max_summary_chars=max_summary_chars)
            disclosed_results.append(card)
            continue

        # detail_level == "progressive"
        already_in_ws = (
            ws is not None
            and active_session_id is not None
            and bool(exp_id)
            and not force_l2
            and ws.has_l2(active_session_id, exp_id)
        )
        if already_in_ws:
            card = _compact_to_l1_summary(raw_item, max_summary_chars=max_summary_chars)
            card["working_set_status"] = "paged_out"
            card["working_set_note"] = WORKING_SET_NOTE
            card["summary"] = f"{WORKING_SET_NOTE} {card.get('summary', '')}".strip()
            disclosed_results.append(card)
            paged_out_count += 1
        elif l2_emitted < full_top_k:
            item_copy = copy.deepcopy(raw_item)
            item_copy["disclosure_level"] = "full"
            if ws is not None and exp_id:
                item_copy["working_set_status"] = "paged_in"
                newly_loaded_l2_ids.append(exp_id)
            disclosed_results.append(item_copy)
            l2_emitted += 1
        else:
            disclosed_results.append(
                _compact_to_l1_summary(raw_item, max_summary_chars=max_summary_chars)
            )

    if ws is not None and active_session_id is not None and newly_loaded_l2_ids:
        ws.record_l2(active_session_id, newly_loaded_l2_ids)

    returned_tokens = estimate_payload_tokens(disclosed_results)
    saved_tokens_pct = (
        max(0.0, round((1.0 - returned_tokens / full_tokens) * 100.0, 2))
        if full_tokens > 0
        else 0.0
    )

    telemetry: dict[str, Any] = {
        "detail_level": detail_level,
        "full_tokens": full_tokens,
        "returned_tokens": returned_tokens,
        "saved_tokens_pct": saved_tokens_pct,
    }
    if ws is not None and active_session_id is not None:
        telemetry["working_set_paged_out"] = paged_out_count
        telemetry["working_set_size"] = len(ws.get_loaded_ids(active_session_id))
    return disclosed_results, telemetry
