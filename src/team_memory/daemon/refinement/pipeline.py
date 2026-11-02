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

_cached_pms: Any = None


def _get_personal_memory_service() -> Any:
    """Return a process-cached PersonalMemoryService instance."""
    global _cached_pms
    from team_memory.services.memory.personal_memory import PersonalMemoryService

    # If PersonalMemoryService is patched in tests (e.g. unittest.mock), always call it
    if hasattr(PersonalMemoryService, "_mock_name") or getattr(
        PersonalMemoryService, "__module__", ""
    ).startswith("unittest.mock"):
        from team_memory.config import load_settings as _load_tm_settings
        from team_memory.embedding.factory import create_embedding_provider

        _settings = _load_tm_settings()
        embedding = create_embedding_provider(_settings)
        return PersonalMemoryService(
            embedding_provider=embedding, db_url=str(_settings.database.url)
        )

    if _cached_pms is None:
        from team_memory.config import load_settings as _load_tm_settings
        from team_memory.embedding.factory import create_embedding_provider

        _settings = _load_tm_settings()
        embedding = create_embedding_provider(_settings)
        _cached_pms = PersonalMemoryService(
            embedding_provider=embedding, db_url=str(_settings.database.url)
        )
    return _cached_pms


def _resolve_single_root(root: str, config: DaemonConfig) -> str | None:
    """Resolve a single workspace root path to a TM project name (or None)."""
    from team_memory.project_resolver import infer_project_name, read_tm_toml

    if not root or not str(root).strip():
        return None
    # 1. Explicit DaemonConfig.projects path_patterns (when configured)
    for pm in getattr(config, "projects", []) or []:
        for pattern in getattr(pm, "path_patterns", []) or []:
            if pattern in root:
                return pm.name

    # 2. .tm.toml
    tm_config = read_tm_toml(root)
    if tm_config and tm_config.get("name"):
        return str(tm_config["name"])

    # 3. Git repo name (via infer_project_name)
    inferred = infer_project_name(root)
    if inferred and inferred != "default":
        return inferred
    return None


def _resolve_project_with_meta(
    workspace_roots: list[str],
    config: DaemonConfig,
    *,
    prompt: str = "",
    target_files: list[str] | None = None,
    sticky_project: str = "",
) -> tuple[str | None, bool]:
    """Map workspace paths + prompt/target_files + session sticky to `(project, is_explicit)`.

    Priority:
      1. Explicit file path match in `target_files` against candidate roots
      2. Explicit project/alias/path match in `prompt`
      3. Single workspace root (`len(workspace_roots) == 1`)
      4. Session sticky project (`sticky_project` from ConversationBuffer)
      5. Multi-root fallback to `valid_root_projects[0]`
    """
    from pathlib import Path

    from team_memory.project_resolver import (
        _BUILTIN_SHORTHAND_ALIASES,
        _aliases,
        _prompt_matches_keyword,
        normalize_project_name,
    )

    valid_root_projects: list[tuple[str, str]] = []
    for r in workspace_roots or []:
        if not r or not str(r).strip():
            continue
        norm_r = str(r).strip().rstrip("/")
        proj = _resolve_single_root(norm_r, config)
        if proj:
            valid_root_projects.append((norm_r, proj))

    # 1. Explicit target_files match
    if target_files:
        for tf in target_files:
            if not tf or not str(tf).strip():
                continue
            tf_str = str(tf).strip()
            for r, proj in valid_root_projects:
                if tf_str == r or tf_str.startswith(r + "/"):
                    return (proj, True)
            for r, proj in valid_root_projects:
                basename = Path(r).name
                if basename and f"/{basename}/" in tf_str:
                    return (proj, True)

    # 2. Explicit prompt match (path or project/alias keyword)
    prompt_lower = (prompt or "").strip().lower()
    if prompt_lower:
        for r, proj in valid_root_projects:
            r_lower = r.lower()
            basename_lower = Path(r).name.lower()
            if r_lower in prompt_lower or (basename_lower and f"/{basename_lower}/" in prompt_lower):
                return (proj, True)

        candidate_projs: list[str] = []
        for _r, proj in valid_root_projects:
            if proj not in candidate_projs:
                candidate_projs.append(proj)
        for pm in getattr(config, "projects", []) or []:
            if pm.name and pm.name not in candidate_projs:
                candidate_projs.append(pm.name)

        proj_keywords: dict[str, set[str]] = {p: {p.lower()} for p in candidate_projs}
        for r, proj in valid_root_projects:
            bname = Path(r).name.lower()
            if bname:
                proj_keywords.setdefault(proj, set()).add(bname)
        for canonical_key, alias_list in _BUILTIN_SHORTHAND_ALIASES.items():
            all_names = {
                canonical_key.lower(),
                normalize_project_name(canonical_key).lower(),
                *(a.lower() for a in alias_list),
                *(normalize_project_name(a).lower() for a in alias_list),
            }
            for p in candidate_projs:
                if p.lower() in all_names or normalize_project_name(p).lower() in all_names:
                    proj_keywords[p].add(canonical_key.lower())
                    for a in alias_list:
                        proj_keywords[p].add(a.lower())
        for alias_key, target_canonical in _aliases.items():
            target_norm = normalize_project_name(target_canonical).lower()
            for p in candidate_projs:
                if p.lower() == target_canonical.lower() or normalize_project_name(p).lower() == target_norm:
                    proj_keywords[p].add(alias_key.lower())

        scored_matches: list[tuple[int, str]] = []
        for p in candidate_projs:
            best_len = 0
            for kw in proj_keywords.get(p, set()):
                if _prompt_matches_keyword(prompt_lower, kw):
                    if len(kw) > best_len:
                        best_len = len(kw)
            if best_len > 0:
                scored_matches.append((best_len, p))
        if scored_matches:
            scored_matches.sort(key=lambda item: item[0], reverse=True)
            return (scored_matches[0][1], True)

    # 3. Single workspace root is unambiguous
    non_empty_roots = [r for r in (workspace_roots or []) if r and str(r).strip()]
    if len(non_empty_roots) == 1 and len(valid_root_projects) == 1:
        return (valid_root_projects[0][1], True)

    # 4. Session sticky project (prevents multi-root fallback to workspace_roots[0])
    sticky_clean = (sticky_project or "").strip()
    if sticky_clean and sticky_clean != "default":
        return (sticky_clean, False)

    # 5. Multi-root fallback to first valid root
    if valid_root_projects:
        return (valid_root_projects[0][1], False)

    return (None, False)


def _resolve_project(
    workspace_roots: list[str],
    config: DaemonConfig,
    *,
    prompt: str = "",
    target_files: list[str] | None = None,
    sticky_project: str = "",
) -> str | None:
    """Map workspace paths to a TM project name.

    Priority:
      1. Explicit file path / prompt project or alias match
      2. Single workspace root (.tm.toml -> git repo basename -> config.yaml path_patterns)
      3. Session sticky project (when provided)
      4. First matching root in workspace_roots
    """
    proj, _ = _resolve_project_with_meta(
        workspace_roots,
        config,
        prompt=prompt,
        target_files=target_files,
        sticky_project=sticky_project,
    )
    return proj


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


def _format_l1_item(r: dict[str, Any], *, session_id: str | None = None) -> str:
    """Format a single recall/wiki result into a compact L1 summary line (~60-80 tokens)."""
    full_id = str(r.get("group_id") or r.get("id", "") or "").strip()
    exp_id = str(r.get("id", "") or "")[:8]
    source = r.get("_source", "mem")
    uri = f"tm://wiki/{exp_id or 'doc'}" if source == "wiki" else (f"tm://exp/{exp_id}" if exp_id else "tm://exp")
    title = str(r.get("title") or "Untitled").strip()

    ws_in_memory = False
    if session_id and session_id != "unknown" and full_id:
        try:
            from team_memory.services.search.progressive_disclosure import (
                WORKING_SET_NOTE,
                get_session_working_set,
            )

            if get_session_working_set().has_l2(session_id, full_id):
                ws_in_memory = True
                r["working_set_status"] = "paged_out"
                r["working_set_note"] = WORKING_SET_NOTE
        except Exception:
            pass

    raw_body = (
        r.get("summary")
        or r.get("solution")
        or r.get("description")
        or r.get("problem")
        or r.get("content")
        or ""
    )
    cleaned_body = " ".join(str(raw_body).split())
    if ws_in_memory:
        summary_text = "[已在当前会话加载 L2 / Already in Working Set]"
    else:
        summary_text = cleaned_body[:180] + ("..." if len(cleaned_body) > 180 else "")

    tags = r.get("tags") or []
    tag_str = f" [{', '.join(str(t) for t in tags[:3])}]" if isinstance(tags, list) and tags else ""

    r["uri"] = uri
    if summary_text and summary_text.lower() != title.lower():
        line = f"- [{uri}]{tag_str} {title}: {summary_text}"
    else:
        line = f"- [{uri}]{tag_str} {title}"
    r["l1_summary"] = line
    return line


def _build_l0_manifest(project: str, experiences: list[dict[str, Any]]) -> str:
    """Build an L0 directory manifest (~100 tokens) for SessionStart injection."""
    tags_seen: list[str] = []
    titles: list[str] = []
    for exp in experiences[:5]:
        if not isinstance(exp, dict):
            continue
        for t in exp.get("tags") or []:
            ts = str(t).strip()
            if ts and ts not in tags_seen:
                tags_seen.append(ts)
        exp_id = str(exp.get("id", "") or "")[:8]
        title = str(exp.get("title") or "").strip()
        if title:
            prefix = f"[tm://exp/{exp_id}] " if exp_id else ""
            titles.append(f"  * {prefix}{title}")

    parts = [f"[TM L0 Manifest | project={project}]"]
    if tags_seen:
        parts.append(f"Modules/Tags: {', '.join(tags_seen[:8])}")
    if titles:
        parts.append("Key Experiences:\n" + "\n".join(titles[:3]))
    parts.append("(Progressive disclosure: L1 summaries auto-inject on prompt; use memory_get for L2 full details)")
    return "\n".join(parts)


def _build_l1_context(
    results: list[dict[str, Any]],
    persona_context: str = "",
    max_items: int = 3,
    *,
    session_id: str | None = None,
) -> str:
    """Build an L1 progressive disclosure context block (< 600 tokens)."""
    context_parts: list[str] = []
    if results:
        lines: list[str] = []
        for r in results[:max_items]:
            if isinstance(r, dict):
                lines.append(_format_l1_item(r, session_id=session_id))
        # Also populate l1_summary on remaining results
        for r in results[max_items:]:
            if isinstance(r, dict):
                _format_l1_item(r, session_id=session_id)
        if lines:
            header = f"[TM L1 Context — Top {len(lines)} of {len(results)} experience(s)]"
            footer = "(Use memory_get(id) or .tm/wiki/ for L2 full details)"
            context_parts.append(header + "\n" + "\n".join(lines) + "\n" + footer)
    if persona_context:
        context_parts.append(persona_context)
    return "\n\n".join(context_parts)[:1500]


async def process_session_start(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any = None,  # ConversationBuffer
) -> dict[str, Any]:
    """Process session start: resolve project, persist to buffer, return L0 context."""
    workspace_roots = input_data.get("workspace_roots", [])
    conversation_id = input_data.get("conversation_id", "")
    sticky_project = ""
    if buf is not None and conversation_id:
        session = await buf.get_session(conversation_id)
        if session:
            sticky_project = str(session.get("project") or "")

    resolved, is_explicit = _resolve_project_with_meta(
        workspace_roots,
        config,
        prompt=str(input_data.get("prompt") or ""),
        sticky_project=sticky_project,
    )
    project = resolved or input_data.get("project", "") or None

    # Persist project mapping so later hooks (which may lack workspace) can look it up
    if buf is not None and conversation_id and project:
        if is_explicit and sticky_project and sticky_project != project and hasattr(buf, "update_session_project"):
            await buf.update_session_project(conversation_id, project)
        else:
            await buf.ensure_session(conversation_id, project)
        logger.info("[READ]  session_start → project=%s conversation=%s", project, conversation_id[:8])

    if not project:
        return {"project": "", "additional_context": "", "l0_manifest": ""}

    experiences: list[dict[str, Any]] = []
    try:
        top_k = getattr(getattr(config, "retrieval", None), "session_start_top_k", 3) or 3
        if not isinstance(top_k, int):
            top_k = 3
        raw = await sink.recall(query=project, project=project, max_results=top_k)
        if isinstance(raw, dict):
            experiences = raw.get("results", []) or []
        elif isinstance(raw, list):
            experiences = raw
    except Exception:
        logger.debug("session_start recall failed (non-critical)", exc_info=True)

    if experiences:
        l0_manifest = _build_l0_manifest(project, experiences)
        return {
            "project": project,
            "l0_manifest": l0_manifest,
            "additional_context": {
                "relevant_experiences": experiences,
                "l0_manifest": l0_manifest,
                "profile": {"static": [], "dynamic": []},
            },
        }

    return {"project": project, "additional_context": "", "l0_manifest": ""}


def _detect_source_pipeline(data: dict) -> str:
    """Detect hook source from payload fields / 从 payload 字段检测来源."""
    if data.get("agent_response") or data.get("generation_id"):
        return "cursor"
    if data.get("last_assistant_message"):
        return "claude"
    if data.get("hook_event_name"):
        return "hermes"
    if data.get("conversationId") or data.get("transcriptPath") or data.get("terminationReason"):
        return "antigravity"
    if data.get("source_platform"):
        return str(data["source_platform"])
    return "unknown"


async def process_before_prompt(
    input_data: dict,
    config: DaemonConfig,
    sink: TMSink,
    buf: Any = None,
    wiki: Any = None,
    metrics: Any = None,
    intent_gate: Any = None,
) -> dict[str, Any]:
    """Process before-prompt: retrieve memories and store user message.

    Args:
        input_data: Parsed hook payload with prompt, workspace_roots.
        config: DaemonConfig.
        sink: TMSink for retrieval.
        buf: ConversationBuffer instance (optional, for storing user prompt).
        wiki: LLMWiki instance (optional, for local wiki search).
        intent_gate: IntentGate instance (optional, for LLM-based intent classification).

    Returns:
        Dict with results and project.
    """
    prompt = input_data.get("prompt", "")
    workspace_roots = input_data.get("workspace_roots", [])
    session_id = input_data.get("conversation_id", "") or "unknown"

    sticky_project = ""
    prior_user_messages: list[str] = []
    if buf and session_id != "unknown":
        session = await buf.get_session(session_id)
        if session:
            sticky_project = str(session.get("project") or "")
        if hasattr(buf, "get_user_messages"):
            try:
                fetched_msgs = await buf.get_user_messages(session_id)
                if isinstance(fetched_msgs, list):
                    prior_user_messages = [str(m) for m in fetched_msgs if m]
            except Exception:
                prior_user_messages = []

    raw_files = input_data.get("target_files")
    target_files = [str(f) for f in raw_files if f] if isinstance(raw_files, list) else None

    project, is_explicit = _resolve_project_with_meta(
        workspace_roots,
        config,
        prompt=prompt,
        target_files=target_files,
        sticky_project=sticky_project,
    )

    # Fallback: if no workspace matched, use sticky_project if present
    if project is None and sticky_project:
        project = sticky_project

    # Update session sticky project when explicitly resolved to a new project
    if buf and session_id != "unknown" and project and is_explicit:
        if sticky_project and sticky_project != project and hasattr(buf, "update_session_project"):
            await buf.update_session_project(session_id, project)

    # Store user message in conversation buffer (always, not just on trigger keywords)
    if buf and prompt.strip():
        await buf.add_message(session_id, "user", prompt, project=project or "default")

    # --- Noise gate (two layers) ---
    # Layer 1 (rules, zero-cost): flag high-confidence noise (test commands,
    # confirmations, multi-step instructions, short fillers). Conservative —
    # only obvious noise is caught here; ambiguous prompts defer to the LLM.
    from team_memory.daemon.integration.intent_gate import IntentResult
    from team_memory.daemon.refinement.query_noise import NoiseReason, classify_noise_rule

    rule_verdict = classify_noise_rule(prompt)
    if rule_verdict is not None:
        return {
            "results": [],
            "project": project or "",
            "is_noise": True,
            "noise_reason": rule_verdict.value,
        }

    # Layer 2 (LLM, context-aware): summarize the true intent of this turn
    # using recent dialogue, then decide whether retrieval is needed.
    intent = IntentResult.BOTH  # default: search everything
    if intent_gate is not None:
        history: list[dict] = []
        if buf and session_id != "unknown":
            history = await buf.get_messages(session_id)
        intent = await intent_gate.classify_with_history(prompt, history=history)
        logger.debug("[INTENT_GATE] prompt=%s → %s", prompt[:60], intent.value)

    if intent == IntentResult.NONE:
        return {
            "results": [],
            "project": project or "",
            "intent": "none",
            "is_noise": True,
            "noise_reason": NoiseReason.LLM.value,
        }

    # --- Short follow-up context enrichment & same-topic short-circuit gate ---
    from team_memory.services.entity.entity_search import extract_entities_from_query
    from team_memory.services.search.progressive_disclosure import get_session_working_set

    ws = get_session_working_set()
    raw_entities = extract_entities_from_query(prompt)
    search_query = prompt
    short_circuited = False
    borrowed_entities: list[str] = []

    if session_id and session_id != "unknown":
        if len(raw_entities) >= 2 and ws.check_and_update_topic_entities(
            raw_entities, overlap_threshold=0.8, session_id=session_id
        ):
            short_circuited = True
            logger.info(
                "[SHORT_CIRCUIT] session=%s entities=%s overlapped >=80%% with working set, skipping recall",
                session_id[:8],
                raw_entities,
            )
        elif len(prompt.strip()) < 25 or len(raw_entities) == 0:
            for prev_msg in reversed(prior_user_messages):
                prev_clean = prev_msg.strip()
                if len(prev_clean) >= 6 and prev_clean != prompt.strip():
                    prev_ents = extract_entities_from_query(prev_clean)
                    if prev_ents:
                        borrowed_entities = prev_ents[:5]
                        search_query = f"{prompt.strip()} ({' '.join(borrowed_entities)})"
                        ws.check_and_update_topic_entities(
                            borrowed_entities, overlap_threshold=1.01, session_id=session_id
                        )
                        logger.info(
                            "[ENRICH] session=%s short follow-up enriched query='%s'",
                            session_id[:8],
                            search_query[:80],
                        )
                        break
            if not borrowed_entities and raw_entities:
                ws.check_and_update_topic_entities(
                    raw_entities, overlap_threshold=0.8, session_id=session_id
                )
        elif raw_entities:
            ws.check_and_update_topic_entities(
                raw_entities, overlap_threshold=1.01, session_id=session_id
            )

    # Retrieve experiences (skip if intent is wiki-only or short-circuited)
    results: list[dict] = []
    if not short_circuited and intent in (IntentResult.EXPERIENCE, IntentResult.BOTH):
        raw_results = await sink.recall(
            query=search_query,
            project=project,
            max_results=getattr(config, "max_results", 5),
        )
        if isinstance(raw_results, dict):
            results = raw_results.get("results", [])
        elif isinstance(raw_results, list):
            results = raw_results
        else:
            results = []

    # Retrieve wiki (skip if intent is experience-only or short-circuited)
    wiki_results: list[dict] = []
    if not short_circuited and intent in (IntentResult.WIKI, IntentResult.BOTH) and wiki is not None:
        try:
            wiki_results = await wiki.search(query=search_query, project=project, limit=3)
        except Exception:
            logger.debug("Wiki search failed for query=%s", search_query[:60], exc_info=True)

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

    # Persona profile: classify intent and load personal memories (dedup per session + Top-5 cap)
    persona_context = ""
    persona = "general"
    try:
        from team_memory.daemon.integration.intent_classifier import classify_persona

        persona = classify_persona(prompt)
        user_id = getattr(config.tm, "user", "admin") or "admin"

        already_injected = (
            session_id != "unknown"
            and bool(session_id)
            and ws.has_persona(persona, session_id=session_id)
        )
        if already_injected:
            logger.debug(
                "[READ]  persona profile → persona=%s already in working set for session=%s, skipping",
                persona,
                session_id[:8],
            )
        else:
            pms = _get_personal_memory_service()
            try:
                profile = await pms.build_profile_for_user(
                    user_id=user_id,
                    current_context=prompt,
                    persona=persona,
                    max_items=5,
                    query_hint=search_query,
                )
            except TypeError:
                profile = await pms.build_profile_for_user(
                    user_id=user_id,
                    current_context=prompt,
                    persona=persona,
                )

            # Format persona context as compact string (capped at Top-5 items)
            all_memories = (profile.get("static", []) + profile.get("dynamic", []))[:5]
            if all_memories:
                lines = [f"[PERSONA: {persona}]"]
                for mem in all_memories:
                    lines.append(f"- {mem}")
                persona_context = "\n".join(lines)
                if session_id and session_id != "unknown":
                    ws.record_persona(persona, session_id=session_id)
                logger.info("[READ]  persona profile → persona=%s memories=%d", persona, len(all_memories))
            else:
                logger.debug("No persona memories for user=%s persona=%s", user_id, persona)
    except Exception:
        logger.debug("Persona profile loading failed (non-critical)", exc_info=True)

    # Build L1 progressive disclosure context (recall + persona, < 600 tokens)
    context = _build_l1_context(
        results, persona_context=persona_context, max_items=3, session_id=session_id
    )

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
        "l1_context": context,
        "intent": intent.value,
        "search_query": search_query,
        "short_circuited": short_circuited,
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
    sticky_project = ""
    if session_id != "unknown" and buf is not None:
        session = await buf.get_session(session_id)
        if session:
            sticky_project = str(session.get("project") or "")

    raw_files = input_data.get("target_files")
    target_files = [str(f) for f in raw_files if f] if isinstance(raw_files, list) else None
    user_prompt = input_data.get("user_prompt")
    prompt_hint = str(user_prompt) if isinstance(user_prompt, str) else ""

    project, is_explicit = _resolve_project_with_meta(
        workspace_roots,
        config,
        prompt=prompt_hint,
        target_files=target_files,
        sticky_project=sticky_project,
    )

    if project is None and sticky_project:
        project = sticky_project

    if session_id != "unknown" and buf is not None and project and is_explicit:
        if sticky_project and sticky_project != project and hasattr(buf, "update_session_project"):
            await buf.update_session_project(session_id, project)

    if project is None:
        logger.debug("No project resolved for workspace: %s — falling back to 'default'", workspace_roots)
        project = "default"

    resolved_project_out = project if project != "default" else ""

    if not response_text.strip():
        logger.debug("Empty response_text, skipping draft pipeline for session=%s", session_id)
        return {"action": "ok", "convergence": False, "draft_id": "", "project": resolved_project_out}

    # Store user_prompt if provided in payload (e.g. Antigravity stop hook)
    if isinstance(user_prompt, str) and user_prompt.strip():
        await buf.add_message(session_id, "user", user_prompt.strip(), project=project)

    # Store assistant message in conversation buffer
    await buf.add_message(session_id, "assistant", response_text, project=project)

    # Gather tool observations from buffer + payload
    tool_obs: list[dict[str, Any]] = []
    if hasattr(buf, "get_tool_observations"):
        try:
            fetched_obs = await buf.get_tool_observations(session_id)
            if isinstance(fetched_obs, list):
                tool_obs.extend(fetched_obs)
        except Exception:
            logger.debug("Failed to load tool_observations for session=%s", session_id, exc_info=True)

    payload_tools = input_data.get("recent_tools")
    if isinstance(payload_tools, list):
        for pt in payload_tools:
            if isinstance(pt, dict):
                tool_obs.append(pt)

    # Detect convergence — multi-signal (first match wins)
    # 检测收敛 — 多信号融合（先匹配先赢）
    converged = False

    # Signal 1: Explicit keyword / tool pattern / error-to-pass resolution
    converged = detector.detect_convergence(
        response_text,
        recent_tools=tool_obs,
        current_path=project,
        previous_path=None,
    )

    # Signal 2: Message-count threshold — auto-converge after N assistant messages
    # 信号 2: 消息累积阈值 — 累积 N 条 assistant 消息后自动收敛
    if not converged:
        msg_count = await buf.count_messages(session_id, role="assistant")
        if not isinstance(msg_count, int):
            msg_count = 0
        msg_threshold = 3  # 3+ assistant messages → auto-converge
        if msg_count >= msg_threshold:
            converged = True
            logger.info(
                "[CONVERGE] session=%s msg_count=%d >= %d (auto-converge)",
                session_id[:8],
                msg_count,
                msg_threshold,
            )

    # Signal 3: Antigravity NO_TOOL_CALL with substantive tool trajectory
    if not converged and input_data.get("termination_reason") == "NO_TOOL_CALL" and len(tool_obs) >= 2:
        converged = True
        logger.info(
            "[CONVERGE] session=%s termination_reason=NO_TOOL_CALL with %d tool_obs",
            session_id[:8],
            len(tool_obs),
        )

    if converged:
        # Check session status — handle already-processed sessions
        session = await buf.get_session(session_id)
        if session and session.get("status") not in ("active", None):
            if session.get("status") == "published":
                # Session was published before but conversation continued —
                # reset to active so new content can produce fresh experiences
                await buf.mark_session(session_id, "active")
                logger.info(
                    "[CONVERGE] session=%s was published, resetting to active for new content",
                    session_id[:8],
                )
            else:
                # In-flight states (needs_refinement, refining, publishing) — skip
                logger.debug(
                    "[CONVERGE] session=%s in status=%s, skipping",
                    session_id[:8],
                    session.get("status"),
                )
                return {
                    "action": "ok",
                    "convergence": False,
                    "draft_id": session_id,
                    "project": resolved_project_out,
                    "tool_observations": tool_obs,
                }

        # Mark session for LLM refinement / 标记会话待精炼
        result = await refiner.mark_for_refinement(session_id)
        # Extract personal memory from user messages / 从用户消息提取个人记忆
        await _try_extract_personal_memory(session_id, buf, config)
        if result is not None:
            return {
                "action": "needs_refinement",
                "convergence": True,
                "draft_id": result.get("draft_id", session_id),
                "project": resolved_project_out,
                "tool_observations": tool_obs,
            }
        return {
            "action": "ok",
            "convergence": True,
            "draft_id": session_id,
            "project": resolved_project_out,
            "tool_observations": tool_obs,
        }

    if metrics:
        source = _detect_source_pipeline(input_data)
        metrics.record(
            "hook_calls_total", 1, {"source": source, "event": "after_response", "project": project or "unknown"}
        )
    return {
        "action": "draft_updated",
        "convergence": False,
        "draft_id": session_id,
        "project": resolved_project_out,
        "tool_observations": tool_obs,
    }


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


async def process_tool_observation(
    input_data: dict[str, Any],
    config: DaemonConfig,
    buf: Any,
) -> dict[str, Any]:
    """Record a coding/terminal tool execution observation in ConversationBuffer."""
    session_id = (
        input_data.get("conversation_id")
        or input_data.get("session_id")
        or "unknown"
    )
    tool_name = str(input_data.get("tool_name") or "").strip()
    if not tool_name or session_id == "unknown" or buf is None:
        return {"action": "skipped", "session_id": session_id}

    raw_files = input_data.get("target_files")
    if isinstance(raw_files, list):
        target_files = [str(f) for f in raw_files if f]
    elif input_data.get("target_file"):
        target_files = [str(input_data.get("target_file"))]
    else:
        target_files = []

    sticky_project = ""
    if hasattr(buf, "get_session"):
        try:
            sess = await buf.get_session(session_id)
            if isinstance(sess, dict):
                sticky_project = str(sess.get("project") or "")
        except Exception:
            sticky_project = ""

    workspace_roots = input_data.get("workspace_roots", [])
    resolved_proj, is_explicit = _resolve_project_with_meta(
        workspace_roots,
        config,
        target_files=target_files,
        sticky_project=sticky_project,
    )
    project = resolved_proj or input_data.get("project")
    if project:
        if is_explicit and sticky_project and sticky_project != project and hasattr(buf, "update_session_project"):
            await buf.update_session_project(session_id, project)
        elif hasattr(buf, "ensure_session"):
            await buf.ensure_session(session_id, project)

    exit_code = input_data.get("exit_code")
    if isinstance(exit_code, str) and exit_code.lstrip("-").isdigit():
        exit_code = int(exit_code)
    elif not isinstance(exit_code, int):
        exit_code = None

    obs_id = 0
    if hasattr(buf, "add_tool_observation"):
        obs_id = await buf.add_tool_observation(
            session_id=session_id,
            tool_name=tool_name,
            command_summary=str(
                input_data.get("command_summary")
                or input_data.get("command")
                or input_data.get("cmd")
                or ""
            ),
            target_files=target_files,
            exit_code=exit_code,
            succeeded=bool(input_data.get("succeeded", True)),
            error_excerpt=str(input_data.get("error_excerpt") or ""),
            project=str(project or "default"),
        )
    return {
        "action": "recorded",
        "session_id": session_id,
        "observation_id": obs_id,
    }
