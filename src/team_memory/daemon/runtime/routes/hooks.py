"""Hook callback endpoints for the TM Daemon.

POST /hooks/after_response
POST /hooks/session_start
POST /hooks/before_prompt
POST /hooks/session_end
POST /hooks/mcp_callback
POST /hooks/session_timeout_check
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

from team_memory.daemon.refinement.pipeline import (
    _resolve_project,
    process_after_response,
    process_before_prompt,
    process_session_end,
    process_session_start,
    process_tool_observation,
)
from team_memory.daemon.runtime.app import _detect_source, send_daemon_alert
from team_memory.daemon.runtime.models import HookPayload, MCPCallbackPayload
from team_memory.project_resolver import find_project_root
from team_memory.services.search.search_log_writer import (
    SearchLogWriter,
    evaluate_rule_based_adoption,
)

logger = logging.getLogger("daemon.app")

router = APIRouter()

# ---------------------------------------------------------------------------
# Payload health probe — detect broken hook pipelines early
# ---------------------------------------------------------------------------
_EMPTY_RESPONSE_COUNTS: dict[str, int] = {}  # key: model or source
_EMPTY_RESPONSE_ALERT_THRESHOLD = 5


async def _check_payload_health(raw: dict) -> None:
    """Alert if a source keeps sending empty response_text (broken pipeline)."""
    response_text = raw.get("response_text", "") or raw.get("agent_response", "") or raw.get("content", "") or ""
    source_key = raw.get("model", "") or _detect_source(raw) or "unknown"

    if response_text.strip():
        _EMPTY_RESPONSE_COUNTS[source_key] = 0
        return

    _EMPTY_RESPONSE_COUNTS[source_key] = _EMPTY_RESPONSE_COUNTS.get(source_key, 0) + 1
    count = _EMPTY_RESPONSE_COUNTS[source_key]

    if count == _EMPTY_RESPONSE_ALERT_THRESHOLD:
        logger.error(
            "[ALERT] after_response received %d consecutive empty response_text from source=%s",
            count,
            source_key,
        )
        await send_daemon_alert(
            title="Hook 链路断裂",
            body=(
                f"after_response 连续 {count} 次收到空 response_text。\n"
                f"来源: {source_key}\n"
                f"可能原因: hook 客户端 payload 格式变更，response_text 未正确传递。"
            ),
            severity="critical",
        )


@router.post("/hooks/after_response")
async def hook_after_response(request: Request) -> dict[str, Any]:
    """Process agent response (draft pipeline)."""
    app = request.app
    raw = await request.json()
    await _check_payload_health(raw)
    capture = getattr(app.state, "payload_capture", None)
    if capture:
        await capture.capture("/hooks/after_response", raw, source=_detect_source(raw))
    payload = HookPayload.model_validate(raw)
    result = await process_after_response(
        input_data=payload.model_dump(),
        config=app.state.config,
        sink=app.state.sink,
        buf=app.state.buf,
        detector=app.state.detector,
        refiner=app.state.refiner,
        metrics=getattr(app.state, "metrics", None),
    )
    # Resolve project from pipeline result (includes sticky session / prompt match) or workspace_roots
    resolved_project = (
        result.get("project")
        or _resolve_project(payload.workspace_roots, app.state.config)
        or ""
    )
    action = result.get("action", "")
    convergence = result.get("convergence", False)
    result["project"] = resolved_project
    logger.info(
        "[WRITE] after_response → project=%s action=%s convergence=%s",
        resolved_project,
        action,
        convergence,
    )
    # Evaluate usage: rule-based fast adoption + buffer response for faithfulness judgment.
    response_text = payload.response_text or payload.prompt
    if response_text:
        search_log: SearchLogWriter = app.state.search_log
        tool_obs = (
            result.get("tool_observations")
            or getattr(payload, "recent_tools", None)
            or raw.get("recent_tools")
            or []
        )

        # Look up cached search from before_prompt
        recent_searches: dict = app.state.recent_searches
        key = payload.conversation_id or resolved_project or "default"
        cached_search = recent_searches.pop(key, None)

        # Auto-buffer response for faithfulness evaluation + rule-based adoption
        try:
            response_buffer = getattr(app.state, "response_buffer", None)
            if cached_search and cached_search.get("log_id"):
                recent_log = {
                    "id": cached_search["log_id"],
                    "query": cached_search["query"],
                    "result_ids": cached_search["result_ids"],
                }
            elif not (cached_search and cached_search.get("mcp_calls")) and hasattr(
                search_log, "get_recent_with_results"
            ):
                # No cached daemon recall or explicit mcp_callback — last resort: most recent log with results.
                recent_log = await search_log.get_recent_with_results(
                    project=resolved_project or None,
                    hours_window=1,
                )
            else:
                recent_log = None

            if recent_log and recent_log.get("id"):
                was_adopted, matched_exp_ids = evaluate_rule_based_adoption(
                    recent_log.get("result_ids") or [],
                    response_text,
                    tool_obs,
                )
                if was_adopted and hasattr(search_log, "record_rule_adoption"):
                    await search_log.record_rule_adoption(
                        str(recent_log["id"]),
                        matched_exp_ids,
                    )
                    result["rule_adopted"] = True
                    result["adopted_experience_ids"] = matched_exp_ids

            if response_buffer is not None:
                if recent_log and recent_log.get("id"):
                    await response_buffer.add(
                        query=recent_log["query"],
                        agent_response=response_text[:10000],
                        result_ids=recent_log["result_ids"],
                        search_log_id=recent_log["id"],
                    )

                # --- Always scan for unjudged MCP recall logs in this turn's window ---
                # MCP recalls (agent-invoked memory_recall) are logged by the server
                # with source='mcp'. They used to be captured only via the fragile
                # mcp_callback PostToolUse hook, so ~85% were missed. Scanning the
                # [last_user_message, now] window here captures them independent of
                # that hook; has_search_log() dedups against the paths above/below.
                t_end = datetime.now(timezone.utc)
                t_start = None
                conv_buf = getattr(app.state, "buf", None) or getattr(app.state, "conversation_buffer", None)
                if conv_buf and payload.conversation_id:
                    try:
                        t_start = await conv_buf.get_last_user_message_time(payload.conversation_id)
                    except Exception:
                        pass
                if not t_start:
                    from datetime import timedelta

                    t_start = t_end - timedelta(minutes=5)

                mcp_fallback = await search_log.find_unjudged_in_window(
                    t_start=t_start,
                    t_end=t_end,
                    project=resolved_project or None,
                    source="mcp",
                )
                if not mcp_fallback and resolved_project:
                    mcp_fallback = await search_log.find_unjudged_in_window(
                        t_start=t_start,
                        t_end=t_end,
                        project=None,
                        source="mcp",
                    )
                for mcp_log in mcp_fallback:
                    mcp_adopted, mcp_matched_ids = evaluate_rule_based_adoption(
                        mcp_log.get("result_ids") or [],
                        response_text,
                        tool_obs,
                    )
                    if mcp_adopted and hasattr(search_log, "record_rule_adoption"):
                        await search_log.record_rule_adoption(
                            str(mcp_log["id"]),
                            mcp_matched_ids,
                        )
                        result["rule_adopted"] = True
                    if not await response_buffer.has_search_log(mcp_log["id"]):
                        await response_buffer.add(
                            query=mcp_log["query"] or "(mcp recall)",
                            agent_response=response_text[:10000],
                            result_ids=mcp_log["result_ids"],
                            search_log_id=mcp_log["id"],
                        )
                        logger.info(
                            "[EVAL]  buffered MCP search_log %s (window %s~%s)",
                            mcp_log["id"][:8],
                            t_start.strftime("%H:%M:%S"),
                            t_end.strftime("%H:%M:%S"),
                        )

                # Process MCP recall calls registered by PostToolUse hooks
                mcp_calls = (cached_search or {}).get("mcp_calls", [])
                for mcp in mcp_calls:
                    if not mcp.get("search_log_id"):
                        continue
                    mcp_adopted, mcp_matched_ids = evaluate_rule_based_adoption(
                        mcp.get("result_ids", []) or [],
                        response_text,
                        tool_obs,
                    )
                    if mcp_adopted and hasattr(search_log, "record_rule_adoption"):
                        await search_log.record_rule_adoption(
                            str(mcp["search_log_id"]),
                            mcp_matched_ids,
                        )
                        result["rule_adopted"] = True
                    # Skip if this search_log_id was already buffered (dedup)
                    if await response_buffer.has_search_log(mcp["search_log_id"]):
                        logger.debug(
                            "[EVAL]  mcp search_log %s already buffered, skipping",
                            mcp["search_log_id"][:8],
                        )
                        continue
                    await response_buffer.add(
                        query="(mcp recall)",
                        agent_response=response_text[:10000],
                        result_ids=mcp.get("result_ids", []),
                        search_log_id=mcp["search_log_id"],
                    )
                    logger.info(
                        "[EVAL]  buffered MCP search_log %s for faithfulness",
                        mcp["search_log_id"][:8],
                    )

                # Check if batch threshold reached
                if await response_buffer.should_batch():
                    logger.info("[EVAL]  faithfulness batch threshold reached, triggering evaluation")
                    import subprocess
                    import sys as _sys

                    # Get DB URL from config (same as how response_buffer connects)
                    from team_memory.config import load_settings as _load_tm2

                    _tm_settings2 = _load_tm2()
                    _db_url_val = str(_tm_settings2.database.url).replace("+asyncpg", "")
                    subprocess.Popen(
                        [
                            _sys.executable,
                            "-m",
                            "team_memory.daemon.quality.faithfulness_batch",
                            "--force",
                            "--db-url",
                            _db_url_val,
                            "--checkpoint",
                            "/tmp/tm-faithfulness-checkpoint.txt",
                        ],
                        cwd=find_project_root(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        except Exception as e:
            logger.info("Faithfulness buffer capture failed: %s", e, exc_info=True)
            await send_daemon_alert(
                title="Faithfulness Buffer Error",
                body=f"Faithfulness buffer capture failed: {e}",
                severity="warning",
            )
    return result


@router.post("/hooks/session_start")
async def hook_session_start(request: Request) -> dict[str, Any]:
    """Inject project context at session start."""
    app = request.app
    raw = await request.json()
    capture = getattr(app.state, "payload_capture", None)
    if capture:
        await capture.capture("/hooks/session_start", raw, source=_detect_source(raw))
    payload = HookPayload.model_validate(raw)
    result = await process_session_start(
        input_data=payload.model_dump(),
        config=app.state.config,
        sink=app.state.sink,
        buf=app.state.buf,
    )
    project = result.get("project", "")
    has_ctx = bool(result.get("additional_context", ""))
    logger.info(
        "[READ]  session_start → project=%s context=%s",
        project,
        "yes" if has_ctx else "none",
    )
    return result


@router.post("/hooks/before_prompt")
async def hook_before_prompt(request: Request) -> dict[str, Any]:
    """Retrieve relevant memories before prompt."""
    app = request.app
    raw = await request.json()
    capture = getattr(app.state, "payload_capture", None)
    if capture:
        await capture.capture("/hooks/before_prompt", raw, source=_detect_source(raw))
    payload = HookPayload.model_validate(raw)
    result = await process_before_prompt(
        input_data=payload.model_dump(),
        config=app.state.config,
        sink=app.state.sink,
        buf=app.state.buf,
        wiki=getattr(app.state, "llm_wiki", None),
        metrics=getattr(app.state, "metrics", None),
        intent_gate=getattr(app.state, "intent_gate", None),
    )
    project = result.get("project", "")
    all_results = result.get("results", [])
    n_results = len(all_results)
    query_preview = payload.prompt[:60] if payload.prompt else ""
    # Log result titles and ids for debugging
    result_lines = []
    for i, r in enumerate(all_results[:5]):
        tid = r.get("id", "?")
        title = r.get("title", "?")[:50]
        result_lines.append(f'  #{i + 1} "{title}" [{tid[:8]}]')
    extra = "\n" + "\n".join(result_lines) if result_lines else ""
    logger.info(
        '[READ]  before_prompt → project=%s query="%s" results=%d%s',
        project,
        query_preview,
        n_results,
        extra,
    )
    # Write search log for evaluation (skip trivially short queries)
    min_query_len = 5
    if query_preview and len(query_preview.strip()) >= min_query_len:
        recent_searches: dict = app.state.recent_searches
        key = payload.conversation_id or project or "default"
        existing = recent_searches.get(key)
        now_mono = time.monotonic()
        # Dedup repeated before_prompt calls with the exact same query in the active turn
        if (
            existing
            and existing.get("log_id")
            and existing.get("query") == query_preview
            and (now_mono - float(existing.get("T1") or 0.0)) < 300.0
        ):
            logger.debug(
                "[READ]  before_prompt duplicate query suppressed for key=%s log_id=%s",
                key[:8],
                str(existing["log_id"])[:8],
            )
            return result

        search_log: SearchLogWriter = app.state.search_log
        log_result_ids = [
            {
                "id": r.get("id", ""),
                "score": r.get("score", 0),
                "source": str(r.get("_source", "mem")),
                "title": str(r.get("title") or "")[:120],
            }
            for r in all_results[:5]
        ]
        # Noise flags from the entry gate — written but excluded from stats.
        is_noise = result.get("is_noise")
        noise_reason = result.get("noise_reason")
        log_id = await search_log.log_search(
            query=query_preview,
            project=project or "default",
            source="daemon",
            intent_type=str(result.get("intent") or "general"),
            result_ids=log_result_ids if log_result_ids else None,
            is_noise=True if is_noise else None,
            noise_reason=noise_reason,
        )
        # Cache for after_response correlation
        if log_id:
            prev_mcp_calls = list((existing or {}).get("mcp_calls") or [])
            recent_searches[key] = {
                "log_id": log_id,
                "query": query_preview,
                "result_ids": log_result_ids,
                "project": project,
                "T1": now_mono,  # monotonic timestamp for window matching
                "T1_iso": datetime.now(timezone.utc).isoformat(),
                "mcp_calls": prev_mcp_calls,  # populated by /hooks/mcp_callback
            }
    return result


@router.post("/hooks/session_end")
async def hook_session_end(request: Request) -> dict[str, Any]:
    """Flush remaining drafts at session end."""
    app = request.app
    raw = await request.json()
    capture = getattr(app.state, "payload_capture", None)
    if capture:
        await capture.capture("/hooks/session_end", raw, source=_detect_source(raw))
    payload = HookPayload.model_validate(raw)
    result = await process_session_end(
        input_data=payload.model_dump(),
        config=app.state.config,
        sink=app.state.sink,
        buf=app.state.buf,
        refiner=app.state.refiner,
    )
    flushed = result.get("flushed", False)
    logger.info(
        "[WRITE] session_end → conversation=%s flushed=%s",
        payload.conversation_id[:8] if payload.conversation_id else "?",
        flushed,
    )
    return result


@router.post("/hooks/mcp_callback")
async def hook_mcp_callback(request: Request) -> dict[str, Any]:
    """Register an MCP recall result from a PostToolUse hook.

    PostToolUse / afterMCPExecution / post_tool_call hooks call this
    after the AI successfully invokes a TM recall tool.  The daemon
    stores the search_log_id in ``recent_searches[session_id].mcp_calls``
    so that ``after_response`` can correlate it with the AI's answer.
    """
    app = request.app
    raw = await request.json()
    capture = getattr(app.state, "payload_capture", None)
    if capture:
        await capture.capture("/hooks/mcp_callback", raw, source=_detect_source(raw))
    payload = MCPCallbackPayload.model_validate(raw)
    recent_searches: dict = app.state.recent_searches
    key = payload.session_id

    if key in recent_searches:
        # Session already has a before_prompt entry — append MCP call
        recent_searches[key]["mcp_calls"].append(
            {
                "tool_use_id": payload.tool_use_id,
                "search_log_id": payload.search_log_id,
                "result_ids": payload.result_ids,
                "tool_name": payload.tool_name,
            }
        )
        logger.info(
            "[MCP]   mcp_callback → session=%s tool=%s log_id=%s..%s",
            key[:8],
            payload.tool_name,
            payload.search_log_id[:8] if payload.search_log_id else "?",
            payload.tool_use_id[:8] if payload.tool_use_id else "?",
        )
    else:
        # No before_prompt entry yet — create a stub entry
        recent_searches[key] = {
            "log_id": None,
            "query": "",
            "result_ids": [],
            "project": "",
            "T1": time.monotonic(),
            "T1_iso": datetime.now(timezone.utc).isoformat(),
            "mcp_calls": [
                {
                    "tool_use_id": payload.tool_use_id,
                    "search_log_id": payload.search_log_id,
                    "result_ids": payload.result_ids,
                    "tool_name": payload.tool_name,
                }
            ],
        }
        logger.info(
            "[MCP]   mcp_callback → session=%s (no before_prompt) tool=%s",
            key[:8],
            payload.tool_name,
        )

    return {"status": "ok", "mcp_calls": len(recent_searches[key]["mcp_calls"])}


@router.post("/hooks/session_timeout_check")
async def hook_session_timeout_check(request: Request) -> dict[str, Any]:
    """Return active sessions that have been idle for >= 30 minutes."""
    from datetime import timedelta

    app = request.app
    buf_obj = app.state.buf
    sessions = await buf_obj.get_active_sessions()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    timed_out = []
    for s in sessions:
        updated_at = s.get("updated_at")
        if not updated_at:
            continue
        try:
            dt = datetime.fromisoformat(str(updated_at))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt <= cutoff:
                timed_out.append(s["id"])
        except (ValueError, TypeError):
            continue
    return {"timed_out_sessions": timed_out}


@router.post("/hooks/tool_observation")
async def hook_tool_observation(request: Request) -> dict[str, Any]:
    """Record a coding/terminal tool execution observation from PostToolUse."""
    app = request.app
    raw = await request.json()
    capture = getattr(app.state, "payload_capture", None)
    if capture:
        await capture.capture("/hooks/tool_observation", raw, source=_detect_source(raw))
    result = await process_tool_observation(
        input_data=raw,
        config=app.state.config,
        buf=app.state.buf,
    )
    logger.debug(
        "[OBS]   tool_observation → session=%s tool=%s action=%s",
        str(raw.get("conversation_id", ""))[:8],
        raw.get("tool_name", "?"),
        result.get("action", ""),
    )
    return result


@router.post("/ingest/session")
async def ingest_session(request: Request) -> dict[str, Any]:
    """Ingest a single conversation session (e.g. from ChatGPT Userscript or AGY)."""
    from team_memory.daemon.refinement.chatgpt_importer import import_chatgpt_sessions

    app = request.app
    raw = await request.json()
    project = str(raw.get("project") or "")
    dry_run = bool(raw.get("dry_run", False))
    conversations = raw.get("conversations") if isinstance(raw.get("conversations"), list) else [raw]
    result = await import_chatgpt_sessions(
        buf=app.state.buf,
        raw_conversations=conversations,
        project=project,
        dry_run=dry_run,
    )
    return result


@router.post("/ingest/batch")
async def ingest_batch(request: Request) -> dict[str, Any]:
    """Ingest a batch of sessions from ChatGPT export or Antigravity brain directory."""
    from datetime import datetime

    from team_memory.daemon.observe.agy_transcript_watcher import import_agy_sessions
    from team_memory.daemon.refinement.chatgpt_importer import import_chatgpt_sessions

    app = request.app
    raw = await request.json()
    source = str(raw.get("source") or "chatgpt").lower()
    project = str(raw.get("project") or "")
    dry_run = bool(raw.get("dry_run", False))

    since_val = raw.get("since_ts") if raw.get("since_ts") is not None else raw.get("since")
    since_ts: float | None = None
    if isinstance(since_val, (int, float)):
        since_ts = float(since_val)
    elif isinstance(since_val, str) and since_val.strip():
        try:
            since_ts = float(since_val)
        except ValueError:
            try:
                since_ts = datetime.fromisoformat(since_val).timestamp()
            except ValueError:
                since_ts = None

    if source in ("antigravity", "agy"):
        return await import_agy_sessions(
            buf=app.state.buf,
            brain_dir=raw.get("brain_dir"),
            file_path=raw.get("file_path"),
            project=project,
            since_ts=since_ts,
            dry_run=dry_run,
        )

    conversations = raw.get("conversations") if isinstance(raw.get("conversations"), list) else None
    return await import_chatgpt_sessions(
        buf=app.state.buf,
        file_path=raw.get("file_path"),
        raw_conversations=conversations,
        project=project,
        since_ts=since_ts,
        dry_run=dry_run,
    )
