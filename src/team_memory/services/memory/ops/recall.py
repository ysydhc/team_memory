from __future__ import annotations

import logging
from pathlib import PurePosixPath

from team_memory.utils.project import resolve_project as _resolve_project

logger = logging.getLogger("team_memory")


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
    )
    results = osearch.results

    if not results:
        return {
            "message": (
                "No matching experiences found. After solving this problem, call memory_save to share the solution."
            ),
            "results": [],
            "reranked": False,
            "intent_type": osearch.intent_type,
        }

    best_id = results[0].get("group_id") or results[0].get("id", "")
    return {
        "message": f"Found {len(results)} solution(s).",
        "results": results,
        "reranked": osearch.reranked,
        "intent_type": osearch.intent_type,
        "feedback_hint": (f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"),
    }


async def _recall_search(
    *,
    query: str,
    tags: list[str] | None,
    max_results: int,
    search_orchestrator: object,
    user: str,
    project: str,
    include_archives: bool,
) -> dict:
    """Search mode: direct query."""
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
    )
    results = osearch.results

    if not results:
        return {
            "message": "No matching experiences found.",
            "results": [],
            "reranked": False,
            "intent_type": osearch.intent_type,
        }

    best_id = results[0].get("group_id") or results[0].get("id", "")
    return {
        "message": f"Found {len(results)} result(s).",
        "results": results,
        "reranked": osearch.reranked,
        "intent_type": osearch.intent_type,
        "feedback_hint": (f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)"),
    }


async def _recall_adaptive(
    *,
    query: str,
    tags: list[str] | None,
    max_results: int,
    adaptive_retriever: object,
    user: str,
    project: str,
    include_archives: bool,
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
        return {
            "message": "No matching experiences found.",
            "results": [],
            "reranked": False,
            "intent_type": result.intent_type,
            "adaptive": {"hops": result.hops, "queries_used": result.queries_used},
        }

    best_id = result.results[0].get("group_id") or result.results[0].get("id", "")
    return {
        "message": f"Found {len(result.results)} result(s) across {result.hops} hop(s).",
        "results": result.results,
        "reranked": result.reranked,
        "intent_type": result.intent_type,
        "adaptive": {
            "hops": result.hops,
            "queries_used": result.queries_used,
            "hop_log": result.hop_log,
        },
        "feedback_hint": f"If helpful, call memory_feedback(experience_id='{best_id}', rating=5)",
    }


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

    result_ids: list[dict] | None = None
    if results:
        result_ids = []
        for r in results:
            rid = r.get("group_id") or r.get("id", "")
            score = r.get("score", r.get("similarity", 0))
            result_ids.append({"id": str(rid), "score": score})

    # Late import: tests patch write_search_log on the facade module
    import team_memory.services.memory.memory_operations as _ops_facade

    log_id = await _ops_facade.write_search_log(
        db_url=db_url,
        query=query or "(recall)",
        intent_type=intent_type,
        project=project,
        source="mcp",
        result_ids=result_ids,
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
) -> dict:
    """Unified read: route to solve/search/suggest mode based on input."""
    if include_archives is None:
        from team_memory.config import _default_include_archives

        include_archives = _default_include_archives()

    resolved_project = _resolve_project(project)

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
