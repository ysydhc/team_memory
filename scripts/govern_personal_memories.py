#!/usr/bin/env python3
"""Governance script for personal_memories: classify, prune ephemeral/expired, and cluster-dedup.

Usage:
    # Preview only (default, zero DB writes):
    .venv/bin/python scripts/govern_personal_memories.py --dry-run

    # Apply changes to PostgreSQL in a single transaction:
    .venv/bin/python scripts/govern_personal_memories.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure src/ is importable when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from sqlalchemy import delete, select, update  # noqa: E402

from team_memory.daemon.integration.personal_memory_extractor import (  # noqa: E402
    _is_ephemeral_task_memory,
)
from team_memory.storage.database import close_db, get_session  # noqa: E402
from team_memory.storage.models import PersonalMemory  # noqa: E402
from team_memory.utils.tokenizer import estimate_tokens  # noqa: E402

DEFAULT_DB_URL = os.environ.get(
    "TEAM_MEMORY_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5433/team_memory",
)

# High-precision keyword -> project classification rules
PROJECT_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "adx-sdk-android",
        (
            "adx-sdk-android",
            "logcat",
            "gradle",
            "aar",
            "dexdump",
            "proguard",
            "android",
            "apk",
            "adb ",
            "funnel 埋点",
            "funnel埋点",
            "androidx",
            "activity",
        ),
    ),
    (
        "adx-sdk-ios",
        (
            "adx-sdk-ios",
            "xcode",
            "xcframework",
            "cocoapods",
            "podfile",
            "objective-c",
            "oc_format",
            "otool",
            "ios",
            "privacyinfo",
            "uikit",
            "nsstring",
        ),
    ),
    (
        "team_doc",
        (
            "team_doc",
            "teammemory",
            "team_memory",
            "tm-cli",
            "tm-daemon",
            "pgvector",
            "memory_recall",
            "memory_save",
            "memory_context",
            "janitor",
            "alembic",
            "refinement_worker",
        ),
    ),
    (
        "phone_pilot",
        (
            "phone_pilot",
            "phone-pilot",
            "uiautomator",
            "webdriveragent",
            "wda",
            "mini-game-pilot",
        ),
    ),
    (
        "inapp-adx",
        (
            "inapp-adx",
            "adx-neo",
            "xsdk-proto",
            "inapp-dsp",
            "bidrequest",
            "bidresponse",
            "inapp-protocol",
            "gke",
            "rawdata-",
        ),
    ),
    (
        "inapp-jump",
        (
            "inapp-jump",
            "deeplink",
            "dp跳转",
        ),
    ),
    (
        "ad_learning",
        (
            "ad_learning",
            "obsidian",
            "知识卡片",
        ),
    ),
]

_EXTRA_EPHEMERAL_PATTERNS = (
    "临时跳过",
    "本次任务",
    "这次需求",
    "当前页面",
    "这个按钮",
    "刚才那个",
    "adx-sdk-",
    "修复这个",
    "改成红色",
    "改成绿色",
)

_LOCAL_USER_ALIASES = frozenset({"default", "developer", "admin", "yeshouyou"})


def is_noise_or_ephemeral(content: str) -> bool:
    """Return True if a personal memory looks ephemeral, too short, or one-off."""
    text = (content or "").strip()
    if _is_ephemeral_task_memory(text):
        return True
    low = text.lower()
    if any(p in low for p in _EXTRA_EPHEMERAL_PATTERNS):
        return True
    if re.search(r"(?:202\d[-/]\d{1,2}[-/]\d{1,2})", low):
        return True
    return False


def classify_scope_and_project(content: str, current_project: str | None = None) -> tuple[str, str | None]:
    """Classify a personal memory into ('global', None) or ('project', '<project>')."""
    if current_project and current_project.strip():
        return "project", current_project.strip()
    low = (content or "").lower()
    scores: Counter[str] = Counter()
    for proj, keywords in PROJECT_RULES:
        for kw in keywords:
            if kw in low:
                scores[proj] += 1
    if not scores:
        return "global", None
    best_proj, best_hits = scores.most_common(1)[0]
    # If content mentions multiple projects equally (cross-project tool/workflow), keep global
    top_projects = [p for p, c in scores.items() if c == best_hits]
    if len(top_projects) > 1:
        return "global", None
    return "project", best_proj


def _to_unit_vector(vec: Any) -> Any:
    """Pre-convert an embedding to a normalized numpy 1-D array (or None)."""
    if vec is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return None
        return arr / norm
    except Exception:
        return None


def cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if a is None or b is None:
        return 0.0
    ua = _to_unit_vector(a)
    ub = _to_unit_vector(b)
    if ua is not None and ub is not None and ua.shape == ub.shape:
        return float(ua @ ub)
    try:
        va = [float(x) for x in a]
        vb = [float(x) for x in b]
    except (TypeError, ValueError):
        return 0.0
    if not va or len(va) != len(vb):
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class GovernanceRecord:
    id: Any
    tenant_id: str
    user_id: str
    content: str
    scope: str
    project: str | None
    profile_kind: str
    persona: str
    priority: int
    embedding: Any
    created_at: datetime | None
    updated_at: datetime | None


@dataclass
class ClusterMerge:
    primary_id: Any
    primary_content: str
    scope: str
    project: str | None
    new_priority: int
    dropped_ids: list[Any] = field(default_factory=list)
    dropped_contents: list[str] = field(default_factory=list)


@dataclass
class GovernancePlan:
    total_before: int
    total_after: int
    expired_dynamic_ids: list[Any]
    ephemeral_noise_ids: list[Any]
    ephemeral_samples: list[str]
    tail_pruned_ids: list[Any]
    updates: list[dict[str, Any]]  # {id, scope, project, priority, content}
    delete_ids: list[Any]
    merges: list[ClusterMerge]
    scope_before: dict[str, int]
    scope_after: dict[str, int]
    project_after: dict[str, int]
    persona_after: dict[str, int]
    before_recall_items: int
    before_recall_tokens: int
    after_context_items: int
    after_context_tokens: int
    after_recall_default_tokens: int


def build_governance_plan(
    records: list[GovernanceRecord],
    *,
    dedup_threshold: float = 0.78,
    dynamic_ttl_days: int = 90,
    max_global_per_persona: int = 0,
    now: datetime | None = None,
) -> GovernancePlan:
    """Compute a deterministic governance plan over personal_memories records."""
    now_dt = now or datetime.now(timezone.utc)
    cutoff = now_dt - timedelta(days=dynamic_ttl_days)

    scope_before = Counter(r.scope for r in records)

    # Simulate legacy memory_recall (top-20 static by priority, updated_at)
    legacy_static = sorted(
        [r for r in records if r.profile_kind == "static"],
        key=lambda r: (r.priority, r.updated_at or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )[:20]
    before_recall_items = len(legacy_static)
    before_recall_tokens = sum(estimate_tokens(r.content) for r in legacy_static)

    expired_dynamic_ids: list[Any] = []
    ephemeral_noise_ids: list[Any] = []
    ephemeral_samples: list[str] = []
    survivors: list[tuple[GovernanceRecord, str, str | None]] = []

    for r in records:
        created = r.created_at
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if r.profile_kind == "dynamic" and created is not None and created < cutoff:
            expired_dynamic_ids.append(r.id)
            continue
        if is_noise_or_ephemeral(r.content):
            ephemeral_noise_ids.append(r.id)
            if len(ephemeral_samples) < 15:
                ephemeral_samples.append(r.content)
            continue
        new_scope, new_proj = classify_scope_and_project(r.content, r.project)
        survivors.append((r, new_scope, new_proj))

    # Bucket by (tenant_id, normalized_user, scope, project, profile_kind)
    buckets: dict[tuple[str, str, str, str, str], list[GovernanceRecord]] = defaultdict(list)
    for r, new_scope, new_proj in survivors:
        uid_key = "local_user" if r.user_id in _LOCAL_USER_ALIASES else r.user_id
        buckets[(r.tenant_id, uid_key, new_scope, new_proj or "", r.profile_kind)].append(r)

    updates: list[dict[str, Any]] = []
    dup_delete_ids: list[Any] = []
    merges: list[ClusterMerge] = []
    kept_records: list[tuple[GovernanceRecord, str, str | None, int, str]] = []

    for (_tid, _uid, new_scope, proj_str, _pk), items in buckets.items():
        new_proj = proj_str or None
        # Highest priority, richest content, newest first
        items.sort(
            key=lambda x: (
                int(x.priority or 50),
                len((x.content or "").strip()),
                x.updated_at or x.created_at or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )
        clusters: list[tuple[GovernanceRecord, Any, list[GovernanceRecord]]] = []
        for cand in items:
            cand_norm = " ".join(cand.content.strip().lower().split())
            cand_uvec = _to_unit_vector(cand.embedding)
            matched_idx: int | None = None
            for idx, (prim, prim_uvec, _dups) in enumerate(clusters):
                prim_norm = " ".join(prim.content.strip().lower().split())
                if cand_norm == prim_norm:
                    matched_idx = idx
                    break
                if cand_uvec is not None and prim_uvec is not None and cand_uvec.shape == prim_uvec.shape:
                    sim = float(prim_uvec @ cand_uvec)
                else:
                    sim = cosine_similarity(prim.embedding, cand.embedding)
                if sim >= dedup_threshold:
                    matched_idx = idx
                    break
            if matched_idx is not None:
                clusters[matched_idx][2].append(cand)
            else:
                clusters.append((cand, cand_uvec, []))

        for prim, _prim_uvec, dups in clusters:
            max_pri = max([int(prim.priority or 50)] + [int(d.priority or 50) for d in dups])
            boost = min(20, 5 * len(dups)) if dups else 0
            new_pri = min(100, max_pri + boost)
            best_content = prim.content.strip()
            for d in dups:
                if len(d.content.strip()) > int(len(best_content) * 1.15) and len(d.content.strip()) <= 160:
                    best_content = d.content.strip()

            kept_records.append((prim, new_scope, new_proj, new_pri, best_content))
            if (
                prim.scope != new_scope
                or prim.project != new_proj
                or prim.priority != new_pri
                or prim.content != best_content
            ):
                updates.append(
                    {
                        "id": prim.id,
                        "scope": new_scope,
                        "project": new_proj,
                        "priority": new_pri,
                        "content": best_content,
                    }
                )
            if dups:
                dropped_ids = [d.id for d in dups]
                dup_delete_ids.extend(dropped_ids)
                merges.append(
                    ClusterMerge(
                        primary_id=prim.id,
                        primary_content=best_content,
                        scope=new_scope,
                        project=new_proj,
                        new_priority=new_pri,
                        dropped_ids=dropped_ids,
                        dropped_contents=[d.content.strip() for d in dups],
                    )
                )

    tail_pruned_ids: list[Any] = []
    if max_global_per_persona > 0:
        # Enforce per-(tenant, normalized_user, persona) Top-K cap for global memories;
        # keep all project-scoped memories untouched.
        global_by_persona: dict[tuple[str, str, str], list[tuple[GovernanceRecord, str, str | None, int, str]]] = (
            defaultdict(list)
        )
        project_kept: list[tuple[GovernanceRecord, str, str | None, int, str]] = []
        for entry in kept_records:
            rec, sc, _proj, _pri, _content = entry
            if sc == "global":
                uid_key = "local_user" if rec.user_id in _LOCAL_USER_ALIASES else rec.user_id
                global_by_persona[(rec.tenant_id, uid_key, rec.persona)].append(entry)
            else:
                project_kept.append(entry)

        final_kept: list[tuple[GovernanceRecord, str, str | None, int, str]] = list(project_kept)
        pruned_id_set: set[Any] = set()
        for _key, p_entries in global_by_persona.items():
            p_entries.sort(
                key=lambda e: (
                    0 if e[0].profile_kind == "static" else 1,
                    -e[3],
                    -len(e[4]),
                    -(
                        (e[0].updated_at or e[0].created_at or datetime.min.replace(tzinfo=timezone.utc)).timestamp()
                    ),
                )
            )
            final_kept.extend(p_entries[:max_global_per_persona])
            for dropped in p_entries[max_global_per_persona:]:
                tail_pruned_ids.append(dropped[0].id)
                pruned_id_set.add(dropped[0].id)

        kept_records = final_kept
        updates = [u for u in updates if u["id"] not in pruned_id_set]

    all_delete_ids = expired_dynamic_ids + ephemeral_noise_ids + dup_delete_ids + tail_pruned_ids
    scope_after = Counter(sc for _r, sc, _p, _pri, _c in kept_records)
    project_after = Counter(p for _r, sc, p, _pri, _c in kept_records if sc == "project" and p)
    persona_after = Counter(r.persona for r, _sc, _p, _pri, _c in kept_records)

    # Simulate new budgeted memory_context (global + top-8, <= 400 tokens)
    after_global_static = sorted(
        [(pri, c) for r, sc, _p, pri, c in kept_records if sc == "global" and r.profile_kind == "static"],
        key=lambda x: x[0],
        reverse=True,
    )
    selected_after: list[str] = []
    used_toks = 0
    for _pri, content in after_global_static:
        if len(selected_after) >= 8:
            break
        t = estimate_tokens(content)
        if used_toks + t > 400:
            continue
        selected_after.append(content)
        used_toks += t

    return GovernancePlan(
        total_before=len(records),
        total_after=len(kept_records),
        expired_dynamic_ids=expired_dynamic_ids,
        ephemeral_noise_ids=ephemeral_noise_ids,
        ephemeral_samples=ephemeral_samples,
        tail_pruned_ids=tail_pruned_ids,
        updates=updates,
        delete_ids=all_delete_ids,
        merges=sorted(merges, key=lambda m: len(m.dropped_ids), reverse=True),
        scope_before=dict(scope_before),
        scope_after=dict(scope_after),
        project_after=dict(project_after),
        persona_after=dict(persona_after),
        before_recall_items=before_recall_items,
        before_recall_tokens=before_recall_tokens,
        after_context_items=len(selected_after),
        after_context_tokens=used_toks,
        after_recall_default_tokens=0,
    )


def format_report(plan: GovernancePlan, *, applied: bool) -> str:
    mode = "APPLIED" if applied else "DRY-RUN PREVIEW"
    reduction_pct = (
        round((1 - plan.total_after / plan.total_before) * 100, 1) if plan.total_before else 0.0
    )
    dup_count = (
        len(plan.delete_ids)
        - len(plan.expired_dynamic_ids)
        - len(plan.ephemeral_noise_ids)
        - len(plan.tail_pruned_ids)
    )
    lines = [
        f"=== Personal Memory Governance Report ({mode}) ===",
        (
            f"Total records: {plan.total_before} -> {plan.total_after} "
            f"(-{plan.total_before - plan.total_after}, -{reduction_pct}%)"
        ),
        f"  - Expired dynamic (>TTL): {len(plan.expired_dynamic_ids)}",
        f"  - Ephemeral/noise dropped: {len(plan.ephemeral_noise_ids)}",
        f"  - Semantic duplicates merged: {dup_count} (across {len(plan.merges)} clusters)",
        f"  - Low-priority tail pruned (per-persona cap): {len(plan.tail_pruned_ids)}",
        f"  - Kept records updated (scope/project/priority): {len(plan.updates)}",
        "",
        f"Scope distribution BEFORE: {plan.scope_before}",
        f"Scope distribution AFTER:  {plan.scope_after}",
        f"Project breakdown AFTER:   {plan.project_after}",
        f"Persona breakdown AFTER:   {plan.persona_after}",
        "",
        "=== Simulated Context/Recall Injection Token Comparison ===",
        (
            f"  - BEFORE memory_recall (include_user_profile=True, top-20): "
            f"{plan.before_recall_items} items, ~{plan.before_recall_tokens} tokens"
        ),
        "  - AFTER  memory_recall (default include_user_profile=False): 0 items, 0 tokens (-100%)",
        (
            f"  - AFTER  memory_context (budgeted top-8, <=400 tokens):      "
            f"{plan.after_context_items} items, ~{plan.after_context_tokens} tokens"
        ),
    ]
    if plan.ephemeral_samples:
        lines.append("")
        lines.append("=== Sample Ephemeral/Noise Dropped ===")
        for s in plan.ephemeral_samples[:8]:
            lines.append(f"  [DROP] {s[:90]}")
    if plan.merges:
        lines.append("")
        lines.append("=== Top Merged Semantic Clusters ===")
        for m in plan.merges[:10]:
            proj_tag = f"/{m.project}" if m.project else ""
            header = f"  [KEEP {m.scope}{proj_tag} pri={m.new_priority} merged={len(m.dropped_ids)}]"
            lines.append(f"{header} {m.primary_content[:80]}")
            for d in m.dropped_contents[:2]:
                lines.append(f"      <- merged: {d[:80]}")
            if len(m.dropped_contents) > 2:
                lines.append(f"      <- ... and {len(m.dropped_contents) - 2} more")
    return "\n".join(lines)


async def run_governance(
    *,
    db_url: str,
    apply: bool,
    dedup_threshold: float,
    dynamic_ttl_days: int,
    max_global_per_persona: int = 0,
    report_json: str | None = None,
) -> GovernancePlan:
    async with get_session(db_url, bypass_rls=True) as session:
        rows = (await session.execute(select(PersonalMemory))).scalars().all()
        records = [
            GovernanceRecord(
                id=r.id,
                tenant_id=str(r.tenant_id),
                user_id=r.user_id,
                content=r.content,
                scope=r.scope,
                project=getattr(r, "project", None),
                profile_kind=r.profile_kind,
                persona=r.persona,
                priority=int(r.priority or 50),
                embedding=r.embedding,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in rows
        ]
        plan = build_governance_plan(
            records,
            dedup_threshold=dedup_threshold,
            dynamic_ttl_days=dynamic_ttl_days,
            max_global_per_persona=max_global_per_persona,
        )

        if apply:
            for upd in plan.updates:
                await session.execute(
                    update(PersonalMemory)
                    .where(PersonalMemory.id == upd["id"])
                    .values(
                        scope=upd["scope"],
                        project=upd["project"],
                        priority=upd["priority"],
                        content=upd["content"],
                        updated_at=datetime.now(timezone.utc),
                    )
                )
            if plan.delete_ids:
                await session.execute(
                    delete(PersonalMemory).where(PersonalMemory.id.in_(plan.delete_ids))
                )

    await close_db()

    if report_json:
        Path(report_json).write_text(
            json.dumps(
                {
                    "applied": apply,
                    "total_before": plan.total_before,
                    "total_after": plan.total_after,
                    "scope_before": plan.scope_before,
                    "scope_after": plan.scope_after,
                    "project_after": plan.project_after,
                    "persona_after": plan.persona_after,
                    "before_recall_items": plan.before_recall_items,
                    "before_recall_tokens": plan.before_recall_tokens,
                    "after_context_items": plan.after_context_items,
                    "after_context_tokens": plan.after_context_tokens,
                    "after_recall_default_tokens": plan.after_recall_default_tokens,
                    "expired_dynamic": len(plan.expired_dynamic_ids),
                    "ephemeral_noise": len(plan.ephemeral_noise_ids),
                    "clusters_merged": len(plan.merges),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Govern personal_memories table.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Preview only (default)")
    mode.add_argument("--apply", action="store_true", help="Apply changes to PostgreSQL")
    parser.add_argument("--dedup-threshold", type=float, default=0.78, help="Cosine dedup threshold")
    parser.add_argument("--dynamic-ttl-days", type=int, default=90, help="Dynamic memory TTL days")
    parser.add_argument(
        "--max-global-per-persona",
        type=int,
        default=0,
        help="Optional Top-K cap per persona for global memories (0 = unlimited)",
    )
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="Database URL")
    parser.add_argument("--report-json", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    plan = asyncio.run(
        run_governance(
            db_url=args.db_url,
            apply=bool(args.apply),
            dedup_threshold=args.dedup_threshold,
            dynamic_ttl_days=args.dynamic_ttl_days,
            max_global_per_persona=args.max_global_per_persona,
            report_json=args.report_json,
        )
    )
    print(format_report(plan, applied=bool(args.apply)))


if __name__ == "__main__":
    main()
