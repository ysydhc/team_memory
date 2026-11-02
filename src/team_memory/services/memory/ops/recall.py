from __future__ import annotations

import logging
import os
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


def _attach_stage_telemetry(payload: dict, search_obj: object) -> dict:
    """Attach transient stage metrics/counts to recall payload for SearchLog persistence."""
    stage_metrics = getattr(search_obj, "stage_metrics", None)
    stage_counts = getattr(search_obj, "stage_counts", None)
    if stage_metrics is not None:
        payload["_stage_metrics"] = stage_metrics
    if stage_counts is not None:
        payload["_stage_counts"] = stage_counts
    return payload



async def _append_user_profile_to_recall_json(
    payload: dict,
    *,
    user: str,
    context_text: str | None,
    include_user_profile: bool,
    settings: object,
    embedding_provider: object,
    db_url: str,
) -> dict:
    data = {**payload}
    if not include_user_profile:
        data["profile"] = None
        return data
    max_side = getattr(settings.mcp, "profile_max_strings_per_side", None) or 20
    try:
        from team_memory.services.memory.personal_memory import PersonalMemoryService

        pm_svc = PersonalMemoryService(embedding_provider=embedding_provider, db_url=db_url)
        data["profile"] = await pm_svc.build_profile_for_user(
            user_id=user,
            current_context=context_text,
            max_per_side=max_side,
        )
    except Exception:
        logger.debug("recall: attach profile failed", exc_info=True)
        data["profile"] = {"static": [], "dynamic": []}
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

    return {
        "message": f"Found {len(suggestions)} suggestion(s) for your context.",
        "results": suggestions,
        "reranked": osearch.reranked,
        "intent_type": osearch.intent_type,
    }


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
    include_user_profile: bool = True,
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

    if problem:
        raw = await _recall_solve(
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
        )
        result = await _append_user_profile_to_recall_json(
            raw,
            user=user,
            context_text=ctx_hint,
            include_user_profile=include_user_profile,
            settings=settings,
            embedding_provider=embedding_provider,
            db_url=db_url,
        )
        await _maybe_write_search_log(result, db_url, resolved_project, query=problem or "")
        result.pop("_stage_metrics", None)
        result.pop("_stage_counts", None)
        return result

    if query:
        if adaptive and adaptive_retriever is not None:
            raw = await _recall_adaptive(
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
            raw = await _recall_search(
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
        result = await _append_user_profile_to_recall_json(
            raw,
            user=user,
            context_text=ctx_hint,
            include_user_profile=include_user_profile,
            settings=settings,
            embedding_provider=embedding_provider,
            db_url=db_url,
        )
        await _maybe_write_search_log(result, db_url, resolved_project, query=query or "")
        result.pop("_stage_metrics", None)
        result.pop("_stage_counts", None)
        return result

    if file_path or language or framework:
        raw = await _recall_suggest(
            file_path=file_path,
            language=language,
            framework=framework,
            max_results=max_results,
            search_orchestrator=search_orchestrator,
            user=user,
            project=resolved_project,
        )
        result = await _append_user_profile_to_recall_json(
            raw,
            user=user,
            context_text=ctx_hint,
            include_user_profile=include_user_profile,
            settings=settings,
            embedding_provider=embedding_provider,
            db_url=db_url,
        )
        await _maybe_write_search_log(result, db_url, resolved_project, query=ctx_hint or "")
        return result

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
    )
