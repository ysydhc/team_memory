"""Pipeline logic — extracted from hook scripts, runs inside TM Daemon.

Each function corresponds to a hook event and orchestrates the
conversation buffer, convergence detection, refinement, and TMSink calls.
"""
from __future__ import annotations

import logging
import re
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


async def _try_extract_personal_memory(
    session_id: str, buf: Any, config: DaemonConfig
) -> None:
    """Extract personal memory from user messages in a session.

    Non-blocking: logs errors but does not raise.
    """
    try:
        user_messages = await buf.get_user_messages(session_id)
        if not user_messages:
            logger.debug("No user messages for session=%s, skipping personal memory", session_id[:8])
            return

        from daemon.personal_memory_extractor import PersonalMemoryExtractor
        extractor = PersonalMemoryExtractor(config)
        saved = await extractor.extract_and_save(session_id, user_messages)
        if saved:
            logger.info("[PROFILE] Saved %d personal memories from session=%s", saved, session_id[:8])
        else:
            logger.debug("[PROFILE] No personal memories extracted from session=%s", session_id[:8])
    except Exception:
        logger.exception("Personal memory extraction failed for session=%s", session_id[:8])


async def process_session_start(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
) -> dict[str, Any]:
    """Process session start: resolve project and return context.

    Returns:
        Dict with project and optional additional_context.
    """
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)
    return {"project": project or "", "additional_context": ""}


async def process_before_prompt(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any = None,  # ConversationBuffer
) -> dict[str, Any]:
    """Process before-prompt: retrieve memories and store user message.

    Args:
        input_data: Parsed hook payload with prompt, workspace_roots.
        config: DaemonConfig.
        sink: TMSink for retrieval.
        buf: ConversationBuffer instance (optional, for storing user prompt).

    Returns:
        Dict with results and project.
    """
    prompt = input_data.get("prompt", "")
    workspace_roots = input_data.get("workspace_roots", [])
    session_id = input_data.get("conversation_id", "") or "unknown"
    project = _resolve_project(workspace_roots, config)

    # Store user message in conversation buffer
    if buf and prompt.strip() and project:
        await buf.add_message(session_id, "user", prompt, project=project)

    # Retrieve memories (existing logic)
    if not prompt.strip():
        return {"results": [], "project": project}

    results = await sink.recall(
        query=prompt,
        project=project,
        max_results=getattr(config, "max_results", 5),
    )

    return {"results": results, "project": project or ""}


async def process_after_response(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any,  # ConversationBuffer
    detector: Any,  # ConvergenceDetector
    refiner: Any,  # DraftRefiner
) -> dict[str, Any]:
    """Process agent response: store message, detect convergence, publish.

    Args:
        input_data: Parsed hook payload with conversation_id, response_text/prompt, workspace_roots.
        config: DaemonConfig.
        sink: TMSink for TM storage.
        buf: ConversationBuffer instance.
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

    # Store assistant message in conversation buffer
    await buf.add_message(session_id, "assistant", response_text, project=project)

    # Detect convergence
    converged = detector.detect_convergence(
        response_text,
        recent_tools=[],
        current_path=project,
        previous_path=None,
    )

    if converged:
        # Mark session for LLM refinement
        result = await refiner.mark_for_refinement(session_id)
        # Extract personal memory from user messages
        await _try_extract_personal_memory(session_id, buf, config)
        if result is not None:
            return {
                "action": "needs_refinement",
                "convergence": True,
                "draft_id": result.get("draft_id", session_id),
            }
        return {"action": "ok", "convergence": True, "draft_id": session_id}

    return {"action": "draft_updated", "convergence": False, "draft_id": session_id}


async def process_session_end(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any,
    refiner: Any,
) -> dict[str, Any]:
    """Flush remaining conversation on session end.

    If there's an active session with messages, refine and publish it.
    Also extract personal memory from user messages.
    """
    session_id = input_data.get("conversation_id", "") or "unknown"
    session = await buf.get_session(session_id)

    if not session or session.get("status") != "active":
        return {"action": "ok", "flushed": False}

    # Always try personal memory extraction on session end
    await _try_extract_personal_memory(session_id, buf, config)

    # Check if there are any assistant messages
    messages = await buf.get_assistant_messages(session_id)
    if not messages:
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
        logger.exception("Failed to flush session=%s", session_id)

    return {"action": "ok", "flushed": False}
