"""Lightweight non-blocking multi-tenant LLM / Embedding Usage Ledger.

Records LLM and Embedding invocations into ``llm_usage_logs`` without ever
blocking or breaking the main business path. Captures the active tenant_id
from ``tenant_context`` synchronously before dispatching async persistence.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text as sa_text

from team_memory.storage.database import get_session
from team_memory.storage.tenant_context import get_current_tenant_id, parse_tenant_uuid

logger = logging.getLogger("team_memory.usage_ledger")

_configured_db_url: str | None = None
_pending_tasks: set[asyncio.Task[Any]] = set()
_MAX_PENDING_TASKS = 256


def configure_usage_ledger(db_url: str | None) -> None:
    """Explicitly configure the database URL used by the usage ledger."""
    global _configured_db_url
    _configured_db_url = db_url.strip() if isinstance(db_url, str) and db_url.strip() else None


def resolve_usage_db_url() -> str | None:
    """Resolve DB URL from explicit config, active bootstrap context, or env."""
    if _configured_db_url:
        return _configured_db_url
    try:
        from team_memory import bootstrap

        inst = getattr(bootstrap, "_instance", None)
        if inst is not None and getattr(inst, "db_url", None):
            return str(inst.db_url)
    except Exception:
        pass
    for env_key in ("TEAM_MEMORY_DB_URL", "DATABASE_URL"):
        val = (os.environ.get(env_key) or "").strip()
        if val:
            return val
    return None


def estimate_tokens(payload: str | list[str] | list[dict[str, Any]] | None) -> int:
    """Estimate token count for text, string lists, or chat message lists.

    Uses a lightweight heuristic that handles both CJK (~0.7 tokens/char) and
    ASCII/Latin (~0.25 tokens/char) without external tokenizer dependencies.
    """
    if not payload:
        return 0
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
        text = "\n".join(parts)
    else:
        text = str(payload)

    stripped = text.strip()
    if not stripped:
        return 0

    cjk_chars = sum(1 for ch in stripped if "\u4e00" <= ch <= "\u9fff")
    other_chars = len(stripped) - cjk_chars
    est = int(cjk_chars * 0.7 + other_chars / 4.0)
    return max(1, est)


async def write_usage_log(
    db_url: str,
    *,
    caller: str,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    tenant_id: uuid.UUID | str | None = None,
    created_at: datetime | None = None,
) -> bool:
    """Persist a single usage log row into ``llm_usage_logs``.

    Never raises: returns True on success, False on any database error.
    """
    tid = parse_tenant_uuid(tenant_id) or get_current_tenant_id()
    ts = created_at or datetime.now(timezone.utc)
    try:
        async with get_session(db_url, tenant_id=tid) as session:
            await session.execute(
                sa_text(
                    """
                    INSERT INTO llm_usage_logs (
                        id, tenant_id, caller, provider, model,
                        prompt_tokens, completion_tokens, latency_ms, success, created_at
                    ) VALUES (
                        :id, :tenant_id, :caller, :provider, :model,
                        :prompt_tokens, :completion_tokens, :latency_ms, :success, :created_at
                    )
                    """
                ),
                {
                    "id": uuid.uuid4(),
                    "tenant_id": tid,
                    "caller": str(caller or "unknown")[:100],
                    "provider": str(provider or "unknown")[:64],
                    "model": str(model or "unknown")[:128],
                    "prompt_tokens": max(0, int(prompt_tokens or 0)),
                    "completion_tokens": max(0, int(completion_tokens or 0)),
                    "latency_ms": max(0, int(latency_ms or 0)),
                    "success": bool(success),
                    "created_at": ts,
                },
            )
        return True
    except Exception as exc:
        logger.debug("Usage ledger write failed (non-fatal): %s", exc)
        return False


def record_usage_async(
    *,
    caller: str,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    tenant_id: uuid.UUID | str | None = None,
    db_url: str | None = None,
) -> asyncio.Task[bool] | None:
    """Schedule a non-blocking fire-and-forget usage log write.

    Captures the active tenant_id synchronously before scheduling so RLS
    tenant context is preserved across the background task. Any exception is
    swallowed so billing/telemetry never affects the main business path.
    """
    try:
        target_db_url = db_url or resolve_usage_db_url()
        if not target_db_url:
            return None

        tid = parse_tenant_uuid(tenant_id) or get_current_tenant_id()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None

        if len(_pending_tasks) >= _MAX_PENDING_TASKS:
            logger.debug("Usage ledger queue saturated (%d pending); dropping telemetry record", len(_pending_tasks))
            return None

        task = loop.create_task(
            write_usage_log(
                target_db_url,
                caller=caller,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=success,
                tenant_id=tid,
            )
        )
        _pending_tasks.add(task)
        task.add_done_callback(_pending_tasks.discard)
        return task
    except Exception as exc:
        logger.debug("Usage ledger dispatch failed (non-fatal): %s", exc)
        return None


async def flush_usage_ledger(timeout: float = 5.0) -> None:
    """Await all currently pending background usage writes (useful for tests/shutdown)."""
    if not _pending_tasks:
        return
    tasks = list(_pending_tasks)
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except Exception:
        pass


def _empty_window_summary(window_days: int) -> dict[str, Any]:
    return {
        "window_days": window_days,
        "total_calls": 0,
        "success_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_caller": [],
    }


async def aggregate_usage_windows(
    session: Any,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate 7d and 30d LLM/Embedding usage by caller using an existing session."""
    ref_now = now_utc or datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "window_7d": _empty_window_summary(7),
        "window_30d": _empty_window_summary(30),
    }

    for key, window_days in (("window_7d", 7), ("window_30d", 30)):
        since_ts = ref_now - timedelta(days=window_days)
        rows = (
            await session.execute(
                sa_text(
                    """
                    SELECT
                        caller,
                        COUNT(*) AS calls,
                        COUNT(*) FILTER (WHERE success = true) AS success_calls,
                        COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                        COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                        COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                    FROM llm_usage_logs
                    WHERE created_at >= :since_ts
                    GROUP BY caller
                    ORDER BY (COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0)) DESC,
                             COUNT(*) DESC
                    """
                ),
                {"since_ts": since_ts},
            )
        ).fetchall()

        by_caller: list[dict[str, Any]] = []
        total_calls = 0
        success_calls = 0
        total_prompt = 0
        total_completion = 0

        for caller, calls, ok_calls, p_tok, c_tok, avg_lat in rows:
            calls_i = int(calls or 0)
            ok_i = int(ok_calls or 0)
            p_i = int(p_tok or 0)
            c_i = int(c_tok or 0)
            total_calls += calls_i
            success_calls += ok_i
            total_prompt += p_i
            total_completion += c_i
            by_caller.append(
                {
                    "caller": str(caller or "unknown"),
                    "calls": calls_i,
                    "success_calls": ok_i,
                    "failed_calls": max(0, calls_i - ok_i),
                    "prompt_tokens": p_i,
                    "completion_tokens": c_i,
                    "total_tokens": p_i + c_i,
                    "avg_latency_ms": round(float(avg_lat or 0.0), 1),
                }
            )

        result[key] = {
            "window_days": window_days,
            "total_calls": total_calls,
            "success_calls": success_calls,
            "failed_calls": max(0, total_calls - success_calls),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "by_caller": by_caller,
        }

    return result
