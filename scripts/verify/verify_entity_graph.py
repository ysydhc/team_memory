#!/usr/bin/env python3
"""Entity graph verification script (L2.5).

Usage:
    PYTHONPATH=src:scripts python scripts/verify/verify_entity_graph.py [--step N]

Steps (all run by default):
  1. DB tables exist and are queryable
  2. Rule-based entity extraction from queries
  3. Seed sample entities + relationships
  4. Search entities + graph traversal

Environment: TEAM_MEMORY_DB_URL required (loaded from .env by caller).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

from sqlalchemy import text


# ── helpers ────────────────────────────────────────────────────────────── #

def _ok(msg: str) -> None:
    print(f"  ✔ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✘ {msg}")


def _section(n: int, title: str) -> None:
    print(f"\nStep {n}: {title}")


# ── step 1: tables ────────────────────────────────────────────────────── #

async def step1_check_tables(db_url: str) -> bool:
    """Verify entity graph tables exist and are queryable."""
    _section(1, "DB tables exist and are queryable")
    from team_memory.storage.database import get_session

    ok = True
    async with get_session(db_url) as s:
        for t in ("entities", "relationships", "experience_entities"):
            try:
                r = await s.execute(text(f"SELECT count(*) FROM {t}"))
                count = r.scalar()
                _ok(f"{t}: {count} rows")
            except Exception as exc:
                _fail(f"{t}: {exc}")
                ok = False
    return ok


# ── step 2: rule-based extraction ────────────────────────────────────── #

async def step2_rule_extraction() -> bool:
    """Test rule-based entity extraction from query strings."""
    _section(2, "Rule-based entity extraction")
    from team_memory.services.entity_search import extract_entities_from_query

    tests = [
        ("LiteLLM 和 Clash 有什么关系？", ["LiteLLM", "Clash"]),
        ("how to configure litellm-proxy with Docker", ["litellm-proxy", "Docker"]),
        ("MCP 工具调用失败怎么办", ["MCP"]),
        ('"ClaudeCode" integration', ["ClaudeCode"]),
    ]
    ok = True
    for query, expected in tests:
        result = extract_entities_from_query(query)
        # Check at least one expected entity is found (lenient)
        found = any(e in result for e in expected)
        if found:
            _ok(f"{query!r} → {result}")
        else:
            _fail(f"{query!r} → {result} (expected one of {expected})")
            ok = False
    return ok


# ── step 3: seed sample data ─────────────────────────────────────────── #

_SEED_EID1: uuid.UUID | None = None
_SEED_EID2: uuid.UUID | None = None


async def step3_seed_data(db_url: str) -> bool:
    """Insert sample entities and a relationship, return IDs for step 4."""
    global _SEED_EID1, _SEED_EID2
    _section(3, "Seed sample entities + relationship")
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Entity, Relationship
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    eid1 = uuid.uuid4()
    eid2 = uuid.uuid4()

    async with get_session(db_url) as s:
        try:
            # Upsert (idempotent if name already exists)
            await s.execute(
                pg_insert(Entity)
                .values(id=eid1, name="LiteLLM", entity_type="tool", project="default")
                .on_conflict_do_nothing(constraint="uq_entity_name_project")
            )
            await s.execute(
                pg_insert(Entity)
                .values(id=eid2, name="Clash", entity_type="tool", project="default")
                .on_conflict_do_nothing(constraint="uq_entity_name_project")
            )
            # Fetch actual IDs in case of conflict
            r1 = await s.execute(
                text("SELECT id FROM entities WHERE name = 'LiteLLM' AND project = 'default'")
            )
            row1 = r1.fetchone()
            r2 = await s.execute(
                text("SELECT id FROM entities WHERE name = 'Clash' AND project = 'default'")
            )
            row2 = r2.fetchone()
            eid1 = row1[0]
            eid2 = row2[0]

            # Upsert relationship
            await s.execute(
                pg_insert(Relationship)
                .values(
                    id=uuid.uuid4(),
                    source_entity_id=eid1,
                    target_entity_id=eid2,
                    relation_type="configures",
                    weight=1.0,
                )
                .on_conflict_do_nothing(constraint="uq_relationship_src_tgt_type")
            )
            await s.commit()
            _SEED_EID1 = eid1
            _SEED_EID2 = eid2
            _ok(f"LiteLLM → {eid1}")
            _ok(f"Clash   → {eid2}")
            _ok("relationship: LiteLLM --configures--> Clash")
            return True
        except Exception as exc:
            _fail(str(exc))
            return False


# ── step 4: search + graph traversal ─────────────────────────────────── #

async def step4_search_and_graph(db_url: str) -> bool:
    """Test entity search and graph traversal APIs."""
    _section(4, "Search entities + graph traversal")
    from team_memory.services.entity_search import get_entity_graph, search_entities

    ok = True

    # Search
    try:
        results = await search_entities(db_url, "Lite", ["default"])
        names = [e["name"] for e in results]
        if "LiteLLM" in names:
            _ok(f"search('Lite') → {names}")
        else:
            _fail(f"search('Lite') → {names} (expected LiteLLM)")
            ok = False
    except Exception as exc:
        _fail(f"search failed: {exc}")
        ok = False

    # Graph traversal (use seeded entity)
    if _SEED_EID1 is None:
        _fail("No seed entity ID from step 3")
        return False

    try:
        graph = await get_entity_graph(db_url, str(_SEED_EID1), max_depth=2)
        node_names = [n["name"] for n in graph["nodes"]]
        edge_types = [e["relation_type"] for e in graph["edges"]]
        if "Clash" in node_names and "configures" in edge_types:
            _ok(f"graph(LiteLLM, depth=2): nodes={node_names}, edges={edge_types}")
        else:
            _fail(f"graph(LiteLLM, depth=2): nodes={node_names}, edges={edge_types}")
            ok = False
    except Exception as exc:
        _fail(f"graph traversal failed: {exc}")
        ok = False

    return ok


# ── main ──────────────────────────────────────────────────────────────── #

async def main(steps: set[int] | None = None) -> int:
    db_url = os.environ.get("TEAM_MEMORY_DB_URL")
    if not db_url:
        print("ERROR: TEAM_MEMORY_DB_URL not set. Run with .env sourced.")
        return 1

    if steps is None:
        steps = {1, 2, 3, 4}

    results: dict[int, bool] = {}
    if 1 in steps:
        results[1] = await step1_check_tables(db_url)
    if 2 in steps:
        results[2] = await step2_rule_extraction()
    if 3 in steps:
        results[3] = await step3_seed_data(db_url)
    if 4 in steps:
        results[4] = await step4_search_and_graph(db_url)

    print("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    status = "PASS" if all(results.values()) else "FAIL"
    print(f"  {status}: {passed}/{total} steps passed")
    print("=" * 50)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify entity graph (L2.5)")
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        help="Run only specific step(s) (1-4). Default: all.",
    )
    args = parser.parse_args()
    steps = set(args.step) if args.step else None
    sys.exit(asyncio.run(main(steps)))
