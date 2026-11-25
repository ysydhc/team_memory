from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from team_memory.services.search.progressive_disclosure import apply_progressive_disclosure
from team_memory.utils.project import (
    normalize_project_name,
)
from team_memory.utils.project import (
    resolve_project as _resolve_project,
)

logger = logging.getLogger("team_memory")

_CROSS_PROJECT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "adx_sdk_ios",
        (
            "adx_sdk_ios",
            "adx-sdk-ios",
            "admfunnel",
            "admslottemplate",
            "admadrequestinfo",
            "admrewardedvideo",
            "admfullscreen",
            "admsplash",
            "admbanner",
            "adminterstitial",
            "admnative",
            "admwebview",
        ),
    ),
    (
        "adx_sdk_android",
        (
            "adx_sdk_android",
            "adx-sdk-android",
            "cc.admaster.android",
            "admaster.android",
        ),
    ),
    (
        "phone_pilot",
        (
            "phone_pilot",
            "phone-pilot",
            "phonepilot",
            "mcp_phone_pilot",
        ),
    ),
    (
        "mediago_ads_skills",
        (
            "mediago_ads_skills",
            "mediago-ads-skills",
            "mediago_ads",
        ),
    ),
    (
        "inapp_game_crystal_mist",
        (
            "inapp_game_crystal_mist",
            "crystal_mist",
            "crystal-mist",
        ),
    ),
    (
        "ad_learning",
        (
            "ad_learning",
            "ad-learning",
        ),
    ),
)

_TM_SELF_HINTS: tuple[str, ...] = (
    "team_memory",
    "team_doc",
    "tm-daemon",
    "tm_daemon",
    "tm-cli",
    "faithfulness",
    "search_log",
    "response_buffer",
)


def _infer_recall_project(
    explicit_project: str | None,
    resolved_project: str,
    *,
    file_path: str | None = None,
    query_text: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Infer target project when MCP server defaulted to 'team_memory' or 'default'."""
    if explicit_project and explicit_project.strip():
        return resolved_project
    if resolved_project not in ("default", "team_memory", ""):
        return resolved_project

    if file_path and os.path.isabs(file_path):
        try:
            from team_memory.project_resolver import infer_project_name

            target_dir = file_path if os.path.isdir(file_path) else str(Path(file_path).parent)
            inferred = normalize_project_name(infer_project_name(target_dir))
            if inferred and inferred not in ("default", "team_memory", "antigravity", "brain", "scratch"):
                return inferred
        except Exception:
            pass

    combined_parts: list[str] = []
    if query_text:
        combined_parts.append(query_text)
    if file_path:
        combined_parts.append(file_path)
    if tags:
        combined_parts.extend(str(t) for t in tags if t)
    haystack = " ".join(combined_parts).lower()
    if haystack:
        for proj_slug, hints in _CROSS_PROJECT_HINTS:
            if any(h in haystack for h in hints):
                return proj_slug

    return resolved_project


def build_feedback_instruction(results: list[dict], *, max_ids: int = 3) -> dict | None:
    """Build a compact structured feedback instruction for top recalled experiences."""
    top_ids: list[str] = []
    for r in results:
        if not isinstance(r, dict) or r.get("type") == "archive":
            continue
        eid = str(r.get("group_id") or r.get("id") or "").strip()
        if eid and eid not in top_ids:
            top_ids.append(eid)
        if len(top_ids) >= max_ids:
            break
    if not top_ids:
        return None
    return {
        "action": "memory_feedback",
        "top_experience_ids": top_ids,
        "hint": (
            f"Applied an experience? Call memory_feedback(experience_id='{top_ids[0]}', rating=5); "
            "outdated/misleading? Call rating=1."
        ),
    }


GROUNDING_CONTRACT = (
    "Memory is a lead, not absolute ground truth. Current source code and runtime outputs "
    "always override historical memory. If you verify a recalled item is outdated or misleading, "
    "call memory_feedback(experience_id=..., rating=1) or overwrite it via memory_save."
)


def _attach_stage_telemetry(payload: dict, search_obj: object) -> dict:
    """Attach transient stage metrics/counts, CRAG confidence, and grounding contract."""
    stage_metrics = getattr(search_obj, "stage_metrics", None)
    stage_counts = getattr(search_obj, "stage_counts", None)
    if stage_metrics is not None:
        payload["_stage_metrics"] = dict(stage_metrics) if isinstance(stage_metrics, dict) else stage_metrics
    if stage_counts is not None:
        payload["_stage_counts"] = dict(stage_counts) if isinstance(stage_counts, dict) else stage_counts
    if getattr(search_obj, "degraded", False):
        payload["degraded"] = True
        reason = getattr(search_obj, "degraded_reason", None)
        if reason:
            payload["degraded_reason"] = reason
    if getattr(search_obj, "low_confidence", False):
        payload["low_confidence"] = True
    if payload.get("results"):
        payload["grounding_contract"] = GROUNDING_CONTRACT
    return payload


async def _fetch_user_profile_timed(
    *,
    user: str,
    context_text: str | None,
    include_user_profile: bool,
    settings: object,
    embedding_provider: object,
    db_url: str,
    project: str | None = None,
    embedding_cache: object | None = None,
) -> tuple[dict | None, int]:
    """Fetch user profile with timing, shared embedding cache, and timeout protection."""
    if not include_user_profile:
        return None, 0
    t0 = time.monotonic()
    mcp_cfg = getattr(settings, "mcp", None)
    max_side = getattr(mcp_cfg, "profile_max_strings_per_side", None) or 8
    max_items = getattr(mcp_cfg, "profile_max_items", None) or max_side
    max_tokens = getattr(mcp_cfg, "profile_max_tokens", None) or 400
    emb_timeout = float(getattr(getattr(settings, "search", None), "embedding_timeout_seconds", 2.5))
    try:
        from team_memory.services.memory.personal_memory import PersonalMemoryService

        pm_svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url=db_url)
        build_kwargs: dict = {
            "user_id": user,
            "current_context": context_text,
            "max_per_side": max_side,
            "max_items": max_items,
            "max_tokens": max_tokens,
            "project": project,
        }
        if not hasattr(pm_svc.build_profile_for_user, "_mock_name"):
            build_kwargs["embedding_timeout"] = emb_timeout
            if embedding_cache is not None:
                build_kwargs["embedding_cache"] = embedding_cache
        profile = await pm_svc.build_profile_for_user(**build_kwargs)
    except Exception:
        logger.debug("recall: attach profile failed", exc_info=True)
        profile = {"static": [], "dynamic": []}
    return profile, int((time.monotonic() - t0) * 1000)


async def _append_user_profile_to_recall_json(
    payload: dict,
    *,
    user: str,
    context_text: str | None,
    include_user_profile: bool,
    settings: object,
    embedding_provider: object,
    db_url: str,
    project: str | None = None,
    embedding_cache: object | None = None,
) -> dict:
    data = {**payload}
    profile, pm_ms = await _fetch_user_profile_timed(
        user=user,
        context_text=context_text,
        include_user_profile=include_user_profile,
        settings=settings,
        embedding_provider=embedding_provider,
        db_url=db_url,
        project=project,
        embedding_cache=embedding_cache,
    )
    data["profile"] = profile
    if isinstance(data.get("_stage_metrics"), dict):
        data["_stage_metrics"]["personal_memory_ms"] = pm_ms
    return data


async def _recall_solve(
    *,
    problem: str,
    file_path: str | None,
    language: str | None,
    framework: str | None,
    tags: list[str] | None,
    max_results: int,
    search_orchestrator: object,
    user: str,
    project: str,
    include_archives: bool,
    detail_level: str = "progressive",
    include_superseded: bool = False,
    session_id: str | None = None,
    force_l2: bool = False,
) -> dict:
    """Solve mode: enhanced query + implicit feedback via search."""
    query_parts = [problem]
    if language:
        query_parts.append(f"language: {language}")
    if framework:
        query_parts.append(f"framework: {framework}")
    if file_path:
        p = PurePosixPath(file_path)
        query_parts.append(f"file: {p.name}")
    enhanced_query = " | ".join(query_parts)

    combined_tags = list(tags) if tags else []
    if language and language.lower() not in combined_tags:
        combined_tags.append(language.lower())
    if framework and framework.lower() not in combined_tags:
        combined_tags.append(framework.lower())

    extra_search_kw: dict = {"include_superseded": True} if include_superseded else {}
    osearch = await search_orchestrator.search(
        query=enhanced_query,
        tags=combined_tags or None,
        max_results=max_results,
        min_similarity=0.5,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=2,
        project=project,
        include_archives=include_archives,
        include_promoted=True,
        **extra_search_kw,
    )
    results = osearch.results

    if not results:
        return _attach_stage_telemetry(
            {
                "message": (
                    "No matching experiences found. After solving this problem, call memory_save to share the solution."
                ),
                "results": [],
                "reranked": False,
                "intent_type": osearch.intent_type,
                "returned_tokens": 0,
                "saved_tokens_pct": 0.0,
                "detail_level": detail_level,
            },
            osearch,
        )

    best_id = results[0].get("group_id") or results[0].get("id", "")
    fb_inst = build_feedback_instruction(results)
    results, pd_telemetry = apply_progressive_disclosure(
        results,
        detail_level=detail_level,
        session_id=session_id,
        force_l2=force_l2,
    )
    payload = {
        "message": f"Found {len(results)} solution(s).",
        "results": results,
        "reranked": osearch.reranked,
        "intent_type": osearch.intent_type,
        "returned_tokens": pd_telemetry["returned_tokens"],
        "saved_tokens_pct": pd_telemetry["saved_tokens_pct"],
        "detail_level": detail_level,
        "feedback_hint": (f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"),
    }
    if fb_inst:
        payload["feedback_instruction"] = fb_inst
    if "working_set_paged_out" in pd_telemetry:
        payload["working_set_paged_out"] = pd_telemetry["working_set_paged_out"]
        payload["working_set_size"] = pd_telemetry["working_set_size"]
    return _attach_stage_telemetry(payload, osearch)


async def _recall_search(
    *,
    query: str,
    tags: list[str] | None,
    max_results: int,
    search_orchestrator: object,
    user: str,
    project: str,
    include_archives: bool,
    detail_level: str = "progressive",
    include_superseded: bool = False,
    session_id: str | None = None,
    force_l2: bool = False,
) -> dict:
    """Search mode: direct query."""
    extra_search_kw: dict = {"include_superseded": True} if include_superseded else {}
    osearch = await search_orchestrator.search(
        query=query,
        tags=tags,
        max_results=max_results,
        min_similarity=0.6,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=3,
        project=project,
        include_archives=include_archives,
        include_promoted=True,
        **extra_search_kw,
    )
    results = osearch.results

    if not results:
        return _attach_stage_telemetry(
            {
                "message": "No matching experiences found.",
                "results": [],
                "reranked": False,
                "intent_type": osearch.intent_type,
                "returned_tokens": 0,
                "saved_tokens_pct": 0.0,
                "detail_level": detail_level,
            },
            osearch,
        )

    best_id = results[0].get("group_id") or results[0].get("id", "")
    fb_inst = build_feedback_instruction(results)
    results, pd_telemetry = apply_progressive_disclosure(
        results,
        detail_level=detail_level,
        session_id=session_id,
        force_l2=force_l2,
    )
    payload = {
        "message": f"Found {len(results)} result(s).",
        "results": results,
        "reranked": osearch.reranked,
        "intent_type": osearch.intent_type,
        "returned_tokens": pd_telemetry["returned_tokens"],
        "saved_tokens_pct": pd_telemetry["saved_tokens_pct"],
        "detail_level": detail_level,
        "feedback_hint": (f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"),
    }
    if fb_inst:
        payload["feedback_instruction"] = fb_inst
    if "working_set_paged_out" in pd_telemetry:
        payload["working_set_paged_out"] = pd_telemetry["working_set_paged_out"]
        payload["working_set_size"] = pd_telemetry["working_set_size"]
    return _attach_stage_telemetry(payload, osearch)


async def _recall_adaptive(
    *,
    query: str,
    tags: list[str] | None,
    max_results: int,
    adaptive_retriever: object,
    user: str,
    project: str,
    include_archives: bool,
    detail_level: str = "progressive",
    session_id: str | None = None,
    force_l2: bool = False,
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
        return _attach_stage_telemetry(
            {
                "message": "No matching experiences found.",
                "results": [],
                "reranked": False,
                "intent_type": result.intent_type,
                "returned_tokens": 0,
                "saved_tokens_pct": 0.0,
                "detail_level": detail_level,
                "adaptive": {"hops": result.hops, "queries_used": result.queries_used},
            },
            result,
        )

    best_id = result.results[0].get("group_id") or result.results[0].get("id", "")
    fb_inst = build_feedback_instruction(result.results)
    results, pd_telemetry = apply_progressive_disclosure(
        result.results,
        detail_level=detail_level,
        session_id=session_id,
        force_l2=force_l2,
    )
    payload = {
        "message": f"Found {len(results)} result(s) across {result.hops} hop(s).",
        "results": results,
        "reranked": result.reranked,
        "intent_type": result.intent_type,
        "returned_tokens": pd_telemetry["returned_tokens"],
        "saved_tokens_pct": pd_telemetry["saved_tokens_pct"],
        "detail_level": detail_level,
        "adaptive": {
            "hops": result.hops,
            "queries_used": result.queries_used,
            "hop_log": result.hop_log,
        },
        "feedback_hint": f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)",
    }
    if fb_inst:
        payload["feedback_instruction"] = fb_inst
    if "working_set_paged_out" in pd_telemetry:
        payload["working_set_paged_out"] = pd_telemetry["working_set_paged_out"]
        payload["working_set_size"] = pd_telemetry["working_set_size"]
    return _attach_stage_telemetry(payload, result)


async def _recall_suggest(
    *,
    file_path: str | None,
    language: str | None,
    framework: str | None,
    max_results: int,
    search_orchestrator: object,
    user: str,
    project: str,
) -> dict:
    """Suggest mode: build query from file context."""
    query_parts: list[str] = []

    if file_path:
        p = PurePosixPath(file_path)
        parts_lower = [part.lower() for part in p.parts]
        for hint in (
            "test",
            "tests",
            "migration",
            "migrations",
            "api",
            "auth",
            "config",
            "deploy",
            "docker",
            "ci",
            "scripts",
        ):
            if hint in parts_lower:
                query_parts.append(hint)
        if p.suffix:
            query_parts.append(f"file type: {p.suffix}")
        query_parts.append(p.name)

    if language:
        query_parts.append(language)
    if framework:
        query_parts.append(framework)

    if not query_parts:
        return {"message": "No context to build suggestions from.", "results": []}

    query = " ".join(query_parts)
    filter_tags: list[str] = []
    if language:
        filter_tags.append(language.lower())
    if framework:
        filter_tags.append(framework.lower())

    osearch = await search_orchestrator.search(
        query=query,
        tags=filter_tags or None,
        max_results=max_results,
        min_similarity=0.4,
        user_name=user,
        source="mcp",
        grouped=True,
        top_k_children=1,
        project=project,
        include_promoted=True,
    )
    results = osearch.results

    if not results:
        return {
            "message": "No relevant experiences for this context.",
            "results": [],
            "reranked": False,
            "intent_type": osearch.intent_type,
        }

    suggestions = []
    for r in results:
        parent = r.get("parent", r)
        suggestions.append(
            {
                "id": r.get("group_id") or parent.get("id", ""),
                "title": parent.get("title", "Untitled"),
                "tags": parent.get("tags", []),
                "score": r.get("score", r.get("similarity", 0)),
                "confidence": r.get("confidence", "medium"),
            }
        )

    payload = {
        "message": f"Found {len(suggestions)} suggestion(s) for your context.",
        "results": suggestions,
        "reranked": osearch.reranked,
        "intent_type": osearch.intent_type,
    }
    fb_inst = build_feedback_instruction(suggestions)
    if fb_inst:
        payload["feedback_instruction"] = fb_inst
    return payload


async def _maybe_write_search_log(
    recall_result: dict,
    db_url: str,
    project: str,
    query: str = "",
) -> None:
    """Best-effort write of a SearchLog entry after recall completes.

    Extracts intent_type and result_ids from the recall result dict.
    On success, injects ``search_log_id`` into *recall_result* so that
    PostToolUse hooks can correlate the MCP response with the log entry.
    Failures are logged and do not propagate.
    """
    results = recall_result.get("results", [])
    intent_type = recall_result.get("intent_type", "unknown")
    stage_metrics = recall_result.pop("_stage_metrics", None)
    stage_counts = recall_result.pop("_stage_counts", None)
    returned_tokens = recall_result.get("returned_tokens")

    result_ids: list[dict] | None = None
    if results:
        result_ids = []
        for r in results:
            rid = r.get("group_id") or r.get("id", "")
            score = r.get("score", r.get("similarity", 0))
            result_ids.append({"id": str(rid), "score": score})

    extra_kwargs: dict = {}
    if stage_metrics is not None:
        extra_kwargs["stage_metrics"] = stage_metrics
    if stage_counts is not None:
        extra_kwargs["stage_counts"] = stage_counts
    if returned_tokens is not None:
        extra_kwargs["returned_tokens"] = returned_tokens

    effective_project = _infer_recall_project(None, project, query_text=query)
    if results and effective_project in ("default", "team_memory"):
        top_r = results[0] if isinstance(results[0], dict) else {}
        top_proj = top_r.get("project") or (
            top_r.get("parent", {}).get("project") if isinstance(top_r.get("parent"), dict) else None
        )
        if top_proj and top_proj not in ("default", "*", "all", "team_memory"):
            effective_project = str(top_proj)
        else:
            q_lower = (query or "").lower()
            if not any(h in q_lower for h in _TM_SELF_HINTS):
                other_projs: Counter[str] = Counter()
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    rp = r.get("project") or (
                        r.get("parent", {}).get("project") if isinstance(r.get("parent"), dict) else None
                    )
                    if rp and rp not in ("default", "*", "all", "team_memory"):
                        other_projs[str(rp)] += 1
                if other_projs:
                    effective_project = other_projs.most_common(1)[0][0]
            elif top_proj and top_proj not in ("default", "*", "all"):
                effective_project = str(top_proj)

    # Late import: tests patch write_search_log on the facade module
    import team_memory.services.memory.memory_operations as _ops_facade

    log_id = await _ops_facade.write_search_log(
        db_url=db_url,
        query=query or "(recall)",
        intent_type=intent_type,
        project=effective_project,
        source="mcp",
        result_ids=result_ids,
        **extra_kwargs,
    )
    if log_id:
        recall_result["search_log_id"] = log_id


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
    detail_level: str = "progressive",
    include_superseded: bool = False,
    session_id: str | None = None,
    force_l2: bool = False,
) -> dict:
    """Unified read: route to solve/search/suggest mode based on input."""
    if include_archives is None:
        from team_memory.config import _default_include_archives

        include_archives = _default_include_archives()

    resolved_project = _infer_recall_project(
        project,
        _resolve_project(project),
        file_path=file_path,
        query_text=problem or query,
        tags=tags,
    )

    ctx_hint: str | None = None
    if problem:
        ctx_hint = problem
    elif query:
        ctx_hint = query
    elif file_path:
        ctx_hint = PurePosixPath(file_path).name

    shared_cache = getattr(getattr(search_orchestrator, "_search_pipeline", None), "_cache", None)

    async def _run_with_parallel_profile(recall_coro, log_query: str) -> dict:
        if include_user_profile:
            raw, (profile, pm_ms) = await asyncio.gather(
                recall_coro,
                _fetch_user_profile_timed(
                    user=user,
                    context_text=ctx_hint,
                    include_user_profile=True,
                    settings=settings,
                    embedding_provider=embedding_provider,
                    db_url=db_url,
                    project=resolved_project,
                    embedding_cache=shared_cache,
                ),
            )
            result = {**raw, "profile": profile}
            if isinstance(result.get("_stage_metrics"), dict):
                result["_stage_metrics"]["personal_memory_ms"] = pm_ms
        else:
            raw = await recall_coro
            result = {**raw, "profile": None}
        await _maybe_write_search_log(result, db_url, resolved_project, query=log_query)
        result.pop("_stage_metrics", None)
        result.pop("_stage_counts", None)
        return result

    if problem:
        return await _run_with_parallel_profile(
            _recall_solve(
                problem=problem,
                file_path=file_path,
                language=language,
                framework=framework,
                tags=tags,
                max_results=max_results,
                search_orchestrator=search_orchestrator,
                user=user,
                project=resolved_project,
                include_archives=include_archives,
                detail_level=detail_level,
                include_superseded=include_superseded,
                session_id=session_id,
                force_l2=force_l2,
            ),
            log_query=problem or "",
        )

    if query:
        if adaptive and adaptive_retriever is not None:
            recall_coro = _recall_adaptive(
                query=query,
                tags=tags,
                max_results=max_results,
                adaptive_retriever=adaptive_retriever,
                user=user,
                project=resolved_project,
                include_archives=include_archives,
                detail_level=detail_level,
                session_id=session_id,
                force_l2=force_l2,
            )
        else:
            recall_coro = _recall_search(
                query=query,
                tags=tags,
                max_results=max_results,
                search_orchestrator=search_orchestrator,
                user=user,
                project=resolved_project,
                include_archives=include_archives,
                detail_level=detail_level,
                include_superseded=include_superseded,
                session_id=session_id,
                force_l2=force_l2,
            )
        return await _run_with_parallel_profile(recall_coro, log_query=query or "")

    if file_path or language or framework:
        return await _run_with_parallel_profile(
            _recall_suggest(
                file_path=file_path,
                language=language,
                framework=framework,
                max_results=max_results,
                search_orchestrator=search_orchestrator,
                user=user,
                project=resolved_project,
            ),
            log_query=ctx_hint or "",
        )

    raw = {
        "error": True,
        "message": ("Provide at least one of: problem, query, file_path, language, or framework."),
        "code": "validation_error",
    }
    return await _append_user_profile_to_recall_json(
        raw,
        user=user,
        context_text=ctx_hint,
        include_user_profile=include_user_profile,
        settings=settings,
        embedding_provider=embedding_provider,
        db_url=db_url,
        project=resolved_project,
    )
