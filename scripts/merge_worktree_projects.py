#!/usr/bin/env python3
"""Merge historical Git Worktree orphan projects back into canonical main projects.

Scans PostgreSQL tables (`experiences`, `archives`, `entities`, `search_logs`,
`personal_memories`) for:
  1. Worktree branch-prefixed orphan projects (e.g. `adx_sdk_android_adx_sdk_535`
     -> `adx_sdk_android`, `adx_sdk_ios_e2e_verify` -> `adx_sdk_ios`).
  2. DSH Ticket Worktree temporary directory projects ending with a 12-char hex
     hash (`_[0-9a-f]{12}$`), routed semantically (Option B1) or uniformly to
     `inapp_all` (Option B2 via `--hex-mode hub`).
  3. Non-canonical hyphenated project names (e.g. `adx-sdk-android` ->
     `adx_sdk_android`).

Safe & idempotent:
  - Defaults to `--dry-run` (read-only preview).
  - With `--apply`, runs all updates and entity/relationship merges inside a
    single PostgreSQL transaction, properly handling:
      * `entities` UNIQUE(tenant_id, name, project)
      * `experience_entities` PK(experience_id, entity_id)
      * `relationships` UNIQUE(source_entity_id, target_entity_id, relation_type)
        and self-loop elimination.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/merge_worktree_projects.py --dry-run
    PYTHONPATH=src .venv/bin/python scripts/merge_worktree_projects.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from team_memory.config import load_settings
from team_memory.project_resolver import canonicalize_project_name, load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_TABLES = (
    "experiences",
    "archives",
    "entities",
    "search_logs",
    "personal_memories",
)

# Known canonical prefixes that spawned branch-suffixed worktree dirs
WORKTREE_PREFIX_TARGETS: tuple[tuple[str, str], ...] = (
    ("adx_sdk_android_", "adx_sdk_android"),
    ("adx_sdk_ios_", "adx_sdk_ios"),
    ("inapp_adx_", "inapp_adx"),
    ("inapp_dsp_", "inapp_dsp"),
    ("inapp_jump_", "inapp_jump"),
    ("adx_neo_", "adx_neo"),
    ("agent_common_", "agent_common"),
    ("phone_pilot_", "phone_pilot"),
    ("team_memory_", "team_memory"),
    ("team_doc_", "team_doc"),
)

HEX12_RE = re.compile(r"_[0-9a-f]{12}$")

# Semantic routing rules (Option B1) for DSH Ticket Worktree 12-hex projects.
# Checked in order against the slug (with trailing _<12hex> stripped).
_HEX_SEMANTIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "adx_sdk_ios",
        (
            "ios",
            "biz9",
            "biz8",
            "endcard",
            "spec_copilot_icafe_commit",
        ),
    ),
    (
        "adx_sdk_android",
        (
            "android",
        ),
    ),
    (
        "phone_pilot",
        (
            "phone_pilot",
            "topology",
            "keyboard",
            "emoji_page_map",
        ),
    ),
    (
        "agent_common",
        (
            "onetool",
            "skills",
            "common_changes",
            "sdk_parity",
            "agy_dsh_dual_mode",
            "spec_icafe_cr_flow",
            "slim_spec_copilot",
        ),
    ),
)


@dataclass
class MappingPlan:
    source: str
    target: str
    category: str  # "worktree_prefix" | "dsh_ticket_hex" | "hyphenated"
    counts: dict[str, int]


def classify_hex_ticket_project(project: str, hex_mode: str = "semantic") -> str:
    """Classify a `*_12hex` DSH Ticket Worktree project name into a target project."""
    if hex_mode == "hub":
        return "inapp_all"
    slug = HEX12_RE.sub("", project).lower()
    # Explicit disambiguation: `pp_android_report_trial` is a phone_pilot task
    if slug.startswith("pp_") or "_pp_" in slug or "phone_pilot" in slug:
        return "phone_pilot"
    for target, tokens in _HEX_SEMANTIC_RULES:
        for tok in tokens:
            if tok in slug:
                return target
    return "inapp_all"


def detect_project_mapping(
    project: str,
    existing_projects: set[str],
    *,
    hex_mode: str = "semantic",
    include_hyphenated: bool = True,
) -> tuple[str, str] | None:
    """Return ``(target_project, category)`` if *project* should be merged, else None."""
    if not project:
        return None

    # 1. Explicit branch-prefixed worktree directories
    for prefix, target in WORKTREE_PREFIX_TARGETS:
        if project.startswith(prefix) and project != target:
            return (target, "worktree_prefix")

    # 2. DSH Ticket Worktree 12-char hex suffix
    if HEX12_RE.search(project):
        target = classify_hex_ticket_project(project, hex_mode=hex_mode)
        if target != project:
            return (target, "dsh_ticket_hex")

    # 3. Non-canonical hyphenated names (e.g. adx-sdk-android -> adx_sdk_android)
    if include_hyphenated and "-" in project:
        canon = canonicalize_project_name(project)
        if canon and canon != project and canon in existing_projects:
            return (canon, "hyphenated")

    return None


async def scan_projects(
    conn: AsyncConnection,
    *,
    hex_mode: str = "semantic",
    include_hyphenated: bool = True,
) -> list[MappingPlan]:
    """Scan all 5 tables and build deterministic MappingPlan list."""
    per_project_counts: dict[str, dict[str, int]] = defaultdict(lambda: {t: 0 for t in PROJECT_TABLES})
    existing_projects: set[str] = set()

    for tbl in PROJECT_TABLES:
        res = await conn.execute(
            text(f"SELECT project, count(*) FROM {tbl} WHERE project IS NOT NULL GROUP BY project")
        )
        for proj, cnt in res.fetchall():
            existing_projects.add(proj)
            per_project_counts[proj][tbl] = int(cnt)

    plans: list[MappingPlan] = []
    for proj in sorted(existing_projects):
        detected = detect_project_mapping(
            proj,
            existing_projects,
            hex_mode=hex_mode,
            include_hyphenated=include_hyphenated,
        )
        if detected is None:
            continue
        target, category = detected
        plans.append(
            MappingPlan(
                source=proj,
                target=target,
                category=category,
                counts=dict(per_project_counts[proj]),
            )
        )
    return plans


async def preview_entity_conflicts(conn: AsyncConnection, source: str, target: str) -> tuple[int, int]:
    """Return (conflict_merge_count, direct_move_count) for entities from source -> target."""
    res = await conn.execute(
        text(
            """
            SELECT
                count(*) FILTER (WHERE e2.id IS NOT NULL) AS conflicts,
                count(*) FILTER (WHERE e2.id IS NULL) AS moves
            FROM entities e1
            LEFT JOIN entities e2
              ON e2.tenant_id = e1.tenant_id
             AND e2.name = e1.name
             AND e2.project = :target
            WHERE e1.project = :source
            """
        ),
        {"source": source, "target": target},
    )
    row = res.fetchone()
    return (int(row[0] or 0), int(row[1] or 0)) if row else (0, 0)


async def _merge_single_entity(conn: AsyncConnection, src_id: str, dst_id: str) -> None:
    """Merge entity *src_id* into *dst_id* within an active transaction."""
    # 1. Re-point non-duplicate experience_entities rows, then remove remaining duplicates
    await conn.execute(
        text(
            """
            UPDATE experience_entities
            SET entity_id = :dst_id
            WHERE entity_id = :src_id
              AND NOT EXISTS (
                  SELECT 1 FROM experience_entities ee2
                  WHERE ee2.experience_id = experience_entities.experience_id
                    AND ee2.entity_id = :dst_id
              )
            """
        ),
        {"src_id": src_id, "dst_id": dst_id},
    )
    await conn.execute(
        text("DELETE FROM experience_entities WHERE entity_id = :src_id"),
        {"src_id": src_id},
    )

    # 2. Delete direct edges between src_id and dst_id (would become self-loops)
    await conn.execute(
        text(
            """
            DELETE FROM relationships
            WHERE (source_entity_id = :src_id AND target_entity_id = :dst_id)
               OR (source_entity_id = :dst_id AND target_entity_id = :src_id)
            """
        ),
        {"src_id": src_id, "dst_id": dst_id},
    )

    # 3. Merge outgoing relationships (source_entity_id = src_id)
    out_res = await conn.execute(
        text(
            """
            SELECT id, target_entity_id, relation_type, weight, evidence
            FROM relationships
            WHERE source_entity_id = :src_id
            """
        ),
        {"src_id": src_id},
    )
    for rel_id, tgt_id, rel_type, weight, evidence in out_res.fetchall():
        existing = await conn.execute(
            text(
                """
                SELECT id, weight, evidence
                FROM relationships
                WHERE source_entity_id = :dst_id
                  AND target_entity_id = :tgt_id
                  AND relation_type = :rel_type
                """
            ),
            {"dst_id": dst_id, "tgt_id": tgt_id, "rel_type": rel_type},
        )
        ex_row = existing.fetchone()
        if ex_row is not None:
            ex_id, ex_weight, ex_evidence = ex_row
            merged_ev = list(dict.fromkeys([*(ex_evidence or []), *(evidence or [])]))
            merged_weight = max(float(ex_weight or 1.0), float(weight or 1.0))
            await conn.execute(
                text(
                    """
                    UPDATE relationships
                    SET weight = :weight, evidence = CAST(:evidence AS jsonb)
                    WHERE id = :ex_id
                    """
                ),
                {"weight": merged_weight, "evidence": json.dumps(merged_ev), "ex_id": ex_id},
            )
            await conn.execute(text("DELETE FROM relationships WHERE id = :rel_id"), {"rel_id": rel_id})
        else:
            await conn.execute(
                text("UPDATE relationships SET source_entity_id = :dst_id WHERE id = :rel_id"),
                {"dst_id": dst_id, "rel_id": rel_id},
            )

    # 4. Merge incoming relationships (target_entity_id = src_id)
    in_res = await conn.execute(
        text(
            """
            SELECT id, source_entity_id, relation_type, weight, evidence
            FROM relationships
            WHERE target_entity_id = :src_id
            """
        ),
        {"src_id": src_id},
    )
    for rel_id, orig_src_id, rel_type, weight, evidence in in_res.fetchall():
        existing = await conn.execute(
            text(
                """
                SELECT id, weight, evidence
                FROM relationships
                WHERE source_entity_id = :orig_src_id
                  AND target_entity_id = :dst_id
                  AND relation_type = :rel_type
                """
            ),
            {"orig_src_id": orig_src_id, "dst_id": dst_id, "rel_type": rel_type},
        )
        ex_row = existing.fetchone()
        if ex_row is not None:
            ex_id, ex_weight, ex_evidence = ex_row
            merged_ev = list(dict.fromkeys([*(ex_evidence or []), *(evidence or [])]))
            merged_weight = max(float(ex_weight or 1.0), float(weight or 1.0))
            await conn.execute(
                text(
                    """
                    UPDATE relationships
                    SET weight = :weight, evidence = CAST(:evidence AS jsonb)
                    WHERE id = :ex_id
                    """
                ),
                {"weight": merged_weight, "evidence": json.dumps(merged_ev), "ex_id": ex_id},
            )
            await conn.execute(text("DELETE FROM relationships WHERE id = :rel_id"), {"rel_id": rel_id})
        else:
            await conn.execute(
                text("UPDATE relationships SET target_entity_id = :dst_id WHERE id = :rel_id"),
                {"dst_id": dst_id, "rel_id": rel_id},
            )

    # 5. Merge entity metadata (source_count, aliases, description) and delete src_id
    src_meta = (
        await conn.execute(
            text("SELECT source_count, aliases, description FROM entities WHERE id = :id"),
            {"id": src_id},
        )
    ).fetchone()
    dst_meta = (
        await conn.execute(
            text("SELECT source_count, aliases, description FROM entities WHERE id = :id"),
            {"id": dst_id},
        )
    ).fetchone()
    if src_meta and dst_meta:
        new_count = int(dst_meta[0] or 0) + int(src_meta[0] or 0)
        merged_aliases = list(dict.fromkeys([*(dst_meta[1] or []), *(src_meta[1] or [])]))
        new_desc = dst_meta[2] if (dst_meta[2] and dst_meta[2].strip()) else src_meta[2]
        await conn.execute(
            text(
                """
                UPDATE entities
                SET source_count = :cnt,
                    aliases = CAST(:aliases AS jsonb),
                    description = :desc,
                    updated_at = NOW()
                WHERE id = :dst_id
                """
            ),
            {
                "cnt": new_count,
                "aliases": json.dumps(merged_aliases),
                "desc": new_desc,
                "dst_id": dst_id,
            },
        )

    await conn.execute(text("DELETE FROM entities WHERE id = :src_id"), {"src_id": src_id})


async def merge_entities_for_project(conn: AsyncConnection, source: str, target: str) -> tuple[int, int]:
    """Merge all entities from *source* into *target*. Returns (merged_conflicts, moved)."""
    res = await conn.execute(
        text(
            """
            SELECT e1.id AS src_id, e2.id AS dst_id
            FROM entities e1
            LEFT JOIN entities e2
              ON e2.tenant_id = e1.tenant_id
             AND e2.name = e1.name
             AND e2.project = :target
            WHERE e1.project = :source
            ORDER BY e1.name
            """
        ),
        {"source": source, "target": target},
    )
    rows = res.fetchall()
    merged_conflicts = 0
    moved = 0
    for src_id, dst_id in rows:
        if dst_id is not None:
            await _merge_single_entity(conn, str(src_id), str(dst_id))
            merged_conflicts += 1
        else:
            await conn.execute(
                text("UPDATE entities SET project = :target, updated_at = NOW() WHERE id = :src_id"),
                {"target": target, "src_id": src_id},
            )
            moved += 1
    return merged_conflicts, moved


async def apply_plans(conn: AsyncConnection, plans: list[MappingPlan]) -> dict[str, int]:
    """Execute all MappingPlans inside the caller's transaction."""
    totals: dict[str, int] = {
        "experiences": 0,
        "archives": 0,
        "entities_merged": 0,
        "entities_moved": 0,
        "search_logs": 0,
        "personal_memories": 0,
    }
    for plan in plans:
        for tbl in ("experiences", "archives", "search_logs", "personal_memories"):
            if plan.counts.get(tbl, 0) > 0:
                r = await conn.execute(
                    text(f"UPDATE {tbl} SET project = :target WHERE project = :source"),
                    {"target": plan.target, "source": plan.source},
                )
                totals[tbl] += int(r.rowcount or 0)
        if plan.counts.get("entities", 0) > 0:
            merged, moved = await merge_entities_for_project(conn, plan.source, plan.target)
            totals["entities_merged"] += merged
            totals["entities_moved"] += moved
    return totals


async def run(apply: bool, hex_mode: str, include_hyphenated: bool) -> None:
    settings = load_settings()
    engine = create_async_engine(str(settings.database.url))
    try:
        async with engine.connect() as conn:
            plans = await scan_projects(
                conn,
                hex_mode=hex_mode,
                include_hyphenated=include_hyphenated,
            )
            if not plans:
                print("No orphan worktree or non-canonical projects found. Database is clean.")
                return

            print(f"Found {len(plans)} project mapping(s) (hex_mode={hex_mode}):\n")
            by_cat: dict[str, list[MappingPlan]] = defaultdict(list)
            for p in plans:
                by_cat[p.category].append(p)

            for cat in ("worktree_prefix", "dsh_ticket_hex", "hyphenated"):
                cat_plans = by_cat.get(cat, [])
                if not cat_plans:
                    continue
                print(f"=== Category: {cat} ({len(cat_plans)} projects) ===")
                for p in cat_plans:
                    extra = ""
                    if p.counts["entities"] > 0:
                        conflicts, moves = await preview_entity_conflicts(conn, p.source, p.target)
                        extra = f" [entities: {conflicts} merge, {moves} move]"
                    nonzero = ", ".join(f"{t}={c}" for t, c in p.counts.items() if c > 0)
                    print(f"  {p.source} -> {p.target}  ({nonzero}){extra}")
                print()

        if not apply:
            print("DRY RUN complete. Re-run with --apply to execute in a single transaction.")
            return

        async with engine.begin() as conn:
            totals = await apply_plans(conn, plans)
        print("APPLY complete (committed in single transaction):")
        for k, v in totals.items():
            print(f"  {k}: {v}")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge historical Git Worktree orphan projects back into canonical main projects."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview mappings and entity conflict merges without modifying data (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Execute the merge inside a single PostgreSQL transaction.",
    )
    parser.add_argument(
        "--hex-mode",
        choices=("semantic", "hub"),
        default="semantic",
        help="Routing for 12-hex DSH ticket projects: 'semantic' (Option B1, default) or 'hub' (all -> inapp_all).",
    )
    parser.add_argument(
        "--no-hyphenated",
        action="store_true",
        help="Skip normalizing hyphenated project names (e.g. adx-sdk-android -> adx_sdk_android).",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            apply=args.apply,
            hex_mode=args.hex_mode,
            include_hyphenated=not args.no_hyphenated,
        )
    )


if __name__ == "__main__":
    main()
