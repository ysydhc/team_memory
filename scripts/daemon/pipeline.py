"""Pipeline logic — extracted from hook scripts, runs inside TM Daemon.

Each function corresponds to a hook event and orchestrates the
draft buffer, convergence detection, refinement, and TMSink calls.
"""
from __future__ import annotations

import logging
from typing import Any

from daemon.config import DaemonConfig
from daemon.tm_sink import TMSink

logger = logging.getLogger("tm_daemon.pipeline")


def _resolve_project(workspace_roots: list[str], config: DaemonConfig) -> str | None:
    """Map workspace paths to a TM project name using config patterns."""
    for root in workspace_roots:
        for pm in config.projects:
            for pattern in pm.path_patterns:
                if pattern in root:
                    return pm.name
    return None


async def process_after_response(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any,  # DraftBuffer
    detector: Any,  # ConvergenceDetector
    refiner: Any,  # DraftRefiner
) -> dict[str, Any]:
    """Process agent response: accumulate draft, detect convergence, publish.

    Args:
        input_data: Parsed hook payload with conversation_id, prompt, workspace_roots.
        config: DaemonConfig.
        sink: TMSink for TM storage.
        buf: DraftBuffer instance.
        detector: ConvergenceDetector instance.
        refiner: DraftRefiner instance.

    Returns:
        Dict with action, convergence flag, and draft_id.
    """
    session_id = input_data.get("conversation_id", "") or "unknown"
    # Accept both "response_text" (Claude Code / claude_stop.py) and "prompt"
    # (Hermes / tm_hook.py) so the pipeline stays platform-agnostic.
    response_text = input_data.get("response_text") or input_data.get("prompt", "") or ""
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)

    if project is None:
        logger.debug("No project resolved for workspace: %s", workspace_roots)
        return {"action": "ok", "convergence": False, "draft_id": ""}

    if not response_text.strip():
        logger.debug("Empty response_text, skipping draft pipeline for session=%s", session_id)
        return {"action": "ok", "convergence": False, "draft_id": ""}

    # Get existing pending draft for this session
    existing = await buf.get_pending(session_id)

    # Accumulate text
    if existing:
        accumulated = existing[0].get("content", "") + "\n" + response_text
    else:
        accumulated = response_text

    # Detect convergence
    converged = detector.detect_convergence(
        response_text,
        recent_tools=[],
        current_path=project,
        previous_path=None,
    )

    if converged and existing:
        # Converged with existing draft → update buffer + mark for LLM refinement
        draft_id = existing[0].get("id", "")
        await buf.update_draft(draft_id, accumulated)

        result = await refiner.mark_for_refinement(session_id)
        if result is not None:
            return {
                "action": "needs_refinement",
                "convergence": True,
                "draft_id": result.get("draft_id", draft_id),
            }
        return {"action": "ok", "convergence": True, "draft_id": draft_id}

    elif converged and not existing:
        # Converged on first response — save draft then mark for refinement
        title = f"Session {session_id[:8]} draft"
        tm_resp = await refiner.save_draft(
            session_id, title, accumulated, project=project,
        )
        result = await refiner.mark_for_refinement(session_id)
        if result is not None:
            return {
                "action": "needs_refinement",
                "convergence": True,
                "draft_id": result.get("draft_id", tm_resp.get("id", "")),
            }
        return {
            "action": "draft_saved",
            "convergence": True,
            "draft_id": tm_resp.get("id", ""),
        }

    else:
        # Not converged → save/update draft
        title = f"Session {session_id[:8]} draft"
        tm_resp = await refiner.save_draft(
            session_id, title, accumulated, project=project,
        )
        return {
            "action": "draft_saved",
            "convergence": False,
            "draft_id": tm_resp.get("id", ""),
        }


async def process_session_start(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
) -> dict[str, Any]:
    """Inject project context for new session.

    Retrieves relevant memories from TM and returns them as additional_context.
    """
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)

    if project is None:
        return {"additional_context": "", "project": None}

    try:
        result = await sink.context(project=project)
        return {"additional_context": result, "project": project}
    except Exception:
        logger.exception("Failed to get context for project=%s", project)
        return {"additional_context": "", "project": project}


async def process_before_prompt(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
) -> dict[str, Any]:
    """Retrieve relevant memories for user prompt.

    Checks for keyword triggers in the prompt and recalls matching experiences.
    """
    query = input_data.get("prompt", "")
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)

    # Check keyword triggers
    has_trigger = any(kw in query for kw in config.retrieval.keyword_triggers)
    if not has_trigger and not query:
        return {"results": [], "project": project}

    try:
        results = await sink.recall(
            query=query,
            project=project,
            max_results=config.retrieval.session_start_top_k,
        )
        return {"results": results, "project": project}
    except Exception:
        logger.exception("Failed to recall for query=%s", query[:50])
        return {"results": [], "project": project}


async def process_session_end(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any,
    refiner: Any,
) -> dict[str, Any]:
    """Flush remaining drafts on session end.

    If there's a pending draft for this session, refine and publish it.
    """
    session_id = input_data.get("conversation_id", "") or "unknown"
    existing = await buf.get_pending(session_id)

    if not existing:
        return {"action": "ok", "flushed": False}

    try:
        result = await refiner.refine_and_publish(session_id)
        if result:
            return {
                "action": "published",
                "flushed": True,
                "draft_id": result.get("draft_id", ""),
            }
    except Exception:
        logger.exception("Failed to flush draft for session=%s", session_id)

    return {"action": "ok", "flushed": False}
