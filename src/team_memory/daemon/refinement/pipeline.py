"""Pipeline logic — extracted from hook scripts, runs inside TM Daemon.

Each function corresponds to a hook event and orchestrates the
conversation buffer, convergence detection, refinement, and TMSink calls.
"""

from __future__ import annotations

import logging
from typing import Any

from team_memory.daemon.integration.tm_sink import TMSink
from team_memory.daemon.runtime.config import DaemonConfig

logger = logging.getLogger("tm_daemon.pipeline")


def _resolve_project(workspace_roots: list[str], config: DaemonConfig) -> str | None:
    """Map workspace paths to a TM project name.

    Priority:
      1. .tm.toml in workspace root (explicit declaration)
      2. Git repo basename (auto-inferred)
      3. config.yaml path_patterns (legacy fallback)
    """
    from team_memory.project_resolver import infer_project_name, read_tm_toml

    for root in workspace_roots:
        # 1. .tm.toml
        tm_config = read_tm_toml(root)
        if tm_config and tm_config.get("name"):
            return tm_config["name"]

        # 2. Git repo name (via infer_project_name)
        inferred = infer_project_name(root)
        if inferred and inferred != "default":
            return inferred

    # 3. Legacy: config.yaml path_patterns
    for root in workspace_roots:
        for pm in config.projects:
            for pattern in pm.path_patterns:
                if pattern in root:
                    return pm.name
    return None


async def _try_extract_personal_memory(session_id: str, buf: Any, config: DaemonConfig) -> None:
    """Extract personal memory from user messages in a session.

    Non-blocking: logs errors but does not raise.
    """
    try:
        user_messages = await buf.get_user_messages(session_id)
        if not user_messages:
            logger.debug("No user messages for session=%s, skipping personal memory", session_id[:8])
            return

        from team_memory.daemon.integration.personal_memory_extractor import PersonalMemoryExtractor

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
    buf: Any = None,  # ConversationBuffer
) -> dict[str, Any]:
    """Process session start: resolve project, persist to buffer, return context."""
    workspace_roots = input_data.get("workspace_roots", [])
    conversation_id = input_data.get("conversation_id", "")
    project = _resolve_project(workspace_roots, config)

    # Persist project mapping so later hooks (which may lack workspace) can look it up
    if buf is not None and conversation_id and project:
        await buf.ensure_session(conversation_id, project)
        logger.info("[READ]  session_start → project=%s conversation=%s", project, conversation_id[:8])

    return {"project": project or "", "additional_context": ""}


def _detect_source_pipeline(data: dict) -> str:
    """Detect hook source from payload fields / 从 payload 字段检测来源."""
    if data.get("agent_response") or data.get("generation_id"):
        return "cursor"
    if data.get("last_assistant_message"):
        return "claude"
    if data.get("hook_event_name"):
        return "hermes"
    return "unknown"


async def process_before_prompt(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any = None,
    wiki: Any = None,
    metrics: Any = None,
) -> dict[str, Any]:
    """Process before-prompt: retrieve memories and store user message.

    Args:
        input_data: Parsed hook payload with prompt, workspace_roots.
        config: DaemonConfig.
        sink: TMSink for retrieval.
        buf: ConversationBuffer instance (optional, for storing user prompt).
        wiki: LLMWiki instance (optional, for local wiki search).

    Returns:
        Dict with results and project.
    """
    prompt = input_data.get("prompt", "")
    workspace_roots = input_data.get("workspace_roots", [])
    session_id = input_data.get("conversation_id", "") or "unknown"
    project = _resolve_project(workspace_roots, config)

    # Fallback: if no workspace provided, look up project from existing session record
    if project is None and buf and session_id != "unknown":
        session = await buf.get_session(session_id)
        if session:
            project = session.get("project")

    # Store user message in conversation buffer (always, not just on trigger keywords)
    if buf and prompt.strip():
        await buf.add_message(session_id, "user", prompt, project=project or "default")

    # Retrieve memories (existing logic)
    if not prompt.strip():
        return {"results": [], "project": project}

    results = await sink.recall(
        query=prompt,
        project=project,
        max_results=getattr(config, "max_results", 5),
    )

    # Normalize results to list (some sinks return dict with "results" key)
    if isinstance(results, dict):
        results = results.get("results", [])
    if not isinstance(results, list):
        results = []

    # Dual retrieval: also search local wiki
    wiki_results: list[dict] = []
    if wiki is not None:
        try:
            wiki_results = await wiki.search(query=prompt, project=project, limit=3)
        except Exception:
            logger.debug("Wiki search failed for query=%s", prompt[:60], exc_info=True)

    # Tag results with source for downstream formatting
    for r in results:
        if "_source" not in r:
            r["_source"] = "mem"
    for r in wiki_results:
        r["_source"] = "wiki"

    # Combine: experiences first, then wiki entries (avoid duplicates by title)
    seen_titles = {r.get("title", "").lower() for r in results}
    for wr in wiki_results:
        if wr.get("title", "").lower() not in seen_titles:
            results.append(wr)
            seen_titles.add(wr.get("title", "").lower())

    # Persona profile: classify intent and load personal memories
    persona_context = ""
    persona = "general"
    try:
        from team_memory.daemon.integration.intent_classifier import classify_persona

        persona = classify_persona(prompt)
        user_id = getattr(config.tm, "user", "admin") or "admin"

        from team_memory.config import load_settings as _load_tm_settings
        from team_memory.embedding.factory import create_embedding_provider
        from team_memory.services.memory.personal_memory import PersonalMemoryService

        _settings = _load_tm_settings()
        embedding = create_embedding_provider(_settings)
        pms = PersonalMemoryService(embedding_provider=embedding, db_url=str(_settings.database.url))
        profile = await pms.build_profile_for_user(user_id=user_id, persona=persona)

        # Format persona context as compact string
        all_memories = profile.get("static", []) + profile.get("dynamic", [])
        if all_memories:
            lines = [f"[PERSONA: {persona}]"]
            for mem in all_memories:
                lines.append(f"- {mem}")
            persona_context = "\n".join(lines)
            logger.info("[READ]  persona profile → persona=%s memories=%d", persona, len(all_memories))
        else:
            logger.debug("No persona memories for user=%s persona=%s", user_id, persona)
    except Exception:
        logger.debug("Persona profile loading failed (non-critical)", exc_info=True)

    # Build immediate-feedback context (recall + persona)
    context_parts = []
    if results:
        titles = []
        for r in results[:5]:
            t = r.get("title", "") or r.get("content", "")[:60]
            if t:
                titles.append(f"- {t}")
        if titles:
            context_parts.append(f"TeamMemory found {len(results)} relevant experience(s):\n" + "\n".join(titles))
    if persona_context:
        context_parts.append(persona_context)
    context = "\n\n".join(context_parts)[:2000]

    if metrics:
        source = _detect_source_pipeline(input_data)
        metrics.record(
            "hook_calls_total",
            1,
            {"source": source, "event": "before_prompt", "project": project or "unknown"},
        )

    return {
        "results": results,
        "project": project or "",
        "persona": persona,
        "persona_context": persona_context,
        "context": context,
    }


async def process_after_response(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any,
    detector: Any,
    refiner: Any,
    metrics: Any = None,
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
    response_text = (
        input_data.get("response_text")
        or input_data.get("prompt", "")
        or input_data.get("agent_response", "")
        or input_data.get("content", "")
        or ""
    )
    workspace_roots = input_data.get("workspace_roots", [])
    project = _resolve_project(workspace_roots, config)

    # Fallback: look up project from session record if workspace not provided
    if project is None and session_id != "unknown":
        session = await buf.get_session(session_id)
        if session:
            project = session.get("project")

    if project is None:
        logger.debug("No project resolved for workspace: %s — falling back to 'default'", workspace_roots)
        project = "default"

    if not response_text.strip():
        logger.debug("Empty response_text, skipping draft pipeline for session=%s", session_id)
        return {"action": "ok", "convergence": False, "draft_id": ""}

    # Store assistant message in conversation buffer
    await buf.add_message(session_id, "assistant", response_text, project=project)

    # Detect convergence — three signals (first match wins)
    # 检测收敛 — 三个信号（先匹配先赢）
    converged = False

    # Signal 1: Explicit keyword convergence (original logic)
    # 信号 1: 显式关键词收敛（原有逻辑）
    converged = detector.detect_convergence(
        response_text,
        recent_tools=[],
        current_path=project,
        previous_path=None,
    )

    # Signal 2: Message-count threshold — auto-converge after N assistant messages
    # 信号 2: 消息累积阈值 — 累积 N 条 assistant 消息后自动收敛
    if not converged:
        msg_count = await buf.count_messages(session_id, role="assistant")
        msg_threshold = 3  # 3+ assistant messages → auto-converge
        if msg_count >= msg_threshold:
            converged = True
            logger.info(
                "[CONVERGE] session=%s msg_count=%d >= %d (auto-converge)",
                session_id[:8],
                msg_count,
                msg_threshold,
            )

    if converged:
        # Check session status first — prevent re-trigger on already processed sessions
        # 检查 session 状态 — 防止对已处理的 session 重复触发
        session = await buf.get_session(session_id)
        if session and session.get("status") != "active":
            logger.debug(
                "[CONVERGE] session=%s already in status=%s, skipping",
                session_id[:8],
                session.get("status"),
            )
            return {"action": "ok", "convergence": False, "draft_id": session_id}

        # Mark session for LLM refinement / 标记会话待精炼
        result = await refiner.mark_for_refinement(session_id)
        # Extract personal memory from user messages / 从用户消息提取个人记忆
        await _try_extract_personal_memory(session_id, buf, config)
        if result is not None:
            return {
                "action": "needs_refinement",
                "convergence": True,
                "draft_id": result.get("draft_id", session_id),
            }
        return {"action": "ok", "convergence": True, "draft_id": session_id}

    if metrics:
        source = _detect_source_pipeline(input_data)
        metrics.record(
            "hook_calls_total", 1, {"source": source, "event": "after_response", "project": project or "unknown"}
        )
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
        # Delegate to RefinementWorker for LLM refinement + quality gate.
        # Mark session as needs_refinement so the worker picks it up
        # on its next tick (within ~30 seconds).
        await buf.mark_needs_refinement(session_id)
        logger.info(
            "[WRITE] session_end → conversation=%s marked for refinement",
            session_id[:8],
        )
        return {
            "action": "needs_refinement",
            "flushed": True,
            "draft_id": session_id,
        }
    except Exception:
        logger.exception("Failed to mark session=%s for refinement", session_id)
        return {"action": "ok", "flushed": False}
