"""Entity search service (L2.5).

Two responsibilities:
1. Rule-based entity detection from a query string (no LLM, low latency).
2. PostgreSQL recursive CTE graph traversal to find related experience IDs.

These are used by SearchPipeline to enrich search results with entity-graph
context — surfaces experiences that are related via the entity graph even if
they don't match the query directly by vector/FTS.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import text

from team_memory.storage.database import get_session

# ---------------------------------------------------------------------------
# Rule-based entity extractor (used on the query path, no LLM)
# ---------------------------------------------------------------------------

# Minimum length for an extracted entity name
_MIN_ENTITY_LEN = 2

# Patterns that look like tool/project names: CamelCase, ALLCAPS (≥2 chars),
# hyphenated-words, or known keyword lists.
_CAMEL_PATTERN = re.compile(r'\b[A-Z][a-z]+[A-Z][A-Za-z]*\b')  # CamelCase
_UPPER_PATTERN = re.compile(r'\b[A-Z]{2,}\b')                    # ALLCAPS
_HYPHEN_PATTERN = re.compile(r'\b[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)+\b')  # kebab-case


def extract_entities_from_query(query: str) -> list[str]:
    """Extract candidate entity names from a search query using rules only.

    Returns a deduplicated list of candidate entity names (lowered canonical).
    Intentionally lenient — false positives are cheaper than false negatives here.
    """
    candidates: set[str] = set()

    # CamelCase tokens (LiteLLM, ClaudeCode, etc.)
    for m in _CAMEL_PATTERN.finditer(query):
        w = m.group()
        if len(w) >= _MIN_ENTITY_LEN:
            candidates.add(w)

    # ALLCAPS tokens (MCP, API, SDK, etc.)
    for m in _UPPER_PATTERN.finditer(query):
        w = m.group()
        if len(w) >= _MIN_ENTITY_LEN:
            candidates.add(w)

    # kebab-case tokens (litellm-proxy, claude-code, etc.)
    for m in _HYPHEN_PATTERN.finditer(query):
        w = m.group()
        if len(w) >= _MIN_ENTITY_LEN:
            candidates.add(w)

    # Also include multi-word quoted phrases if present: "LiteLLM proxy"
    for phrase in re.findall(r'"([^"]{2,50})"', query):
        candidates.add(phrase.strip())

    return [c for c in candidates if c]


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------

_GRAPH_CTE_SQL = text("""
WITH RECURSIVE entity_graph(entity_id, depth) AS (
    -- anchor: entities matching any of the given names (case-insensitive)
    SELECT e.id, 0
    FROM entities e
    WHERE lower(e.name) = ANY(:names)
      AND e.project = ANY(:projects)

    UNION

    -- expansion: follow outgoing + incoming edges up to :max_depth hops
    SELECT
        CASE
            WHEN r.source_entity_id = g.entity_id THEN r.target_entity_id
            ELSE r.source_entity_id
        END,
        g.depth + 1
    FROM relationships r
    JOIN entity_graph g ON (
        r.source_entity_id = g.entity_id OR r.target_entity_id = g.entity_id
    )
    WHERE g.depth < :max_depth
)
SELECT DISTINCT ee.experience_id
FROM entity_graph g
JOIN experience_entities ee ON ee.entity_id = g.entity_id
""")


async def find_experience_ids_by_entities(
    db_url: str,
    entity_names: list[str],
    projects: list[str],
    max_depth: int = 2,
) -> list[str]:
    """Return experience IDs reachable from the given entity names within max_depth hops.

    Uses a bidirectional recursive CTE so both directions of relationships
    are explored.  Returns at most 50 IDs to keep overhead bounded.
    """
    if not entity_names or not projects:
        return []

    names_lower = [n.lower() for n in entity_names if n]

    async with get_session(db_url) as session:
        result = await session.execute(
            _GRAPH_CTE_SQL,
            {
                "names": names_lower,
                "projects": projects,
                "max_depth": max_depth,
            },
        )
        rows = result.fetchall()

    return [str(row[0]) for row in rows[:50]]


# ---------------------------------------------------------------------------
# Entity repository helpers (used by Web API)
# ---------------------------------------------------------------------------

_ENTITY_SEARCH_SQL = text("""
SELECT id, name, entity_type, description, aliases, source_count, project,
       created_at, updated_at
FROM entities
WHERE project = ANY(:projects)
  AND (
    lower(name) LIKE :q
    OR aliases::text ILIKE :q_raw
  )
ORDER BY source_count DESC, name
LIMIT :limit
""")

_ENTITY_GRAPH_SQL = text("""
WITH RECURSIVE entity_graph(entity_id, depth, path) AS (
    SELECT e.id, 0, ARRAY[e.id]
    FROM entities e
    WHERE e.id = :entity_id

    UNION

    SELECT
        CASE
            WHEN r.source_entity_id = g.entity_id THEN r.target_entity_id
            ELSE r.source_entity_id
        END,
        g.depth + 1,
        g.path || CASE
            WHEN r.source_entity_id = g.entity_id THEN r.target_entity_id
            ELSE r.source_entity_id
        END
    FROM relationships r
    JOIN entity_graph g ON (
        r.source_entity_id = g.entity_id OR r.target_entity_id = g.entity_id
    )
    WHERE g.depth < :max_depth
      AND NOT (
        CASE
            WHEN r.source_entity_id = g.entity_id THEN r.target_entity_id
            ELSE r.source_entity_id
        END = ANY(g.path)
      )
)
SELECT DISTINCT
    e.id,
    e.name,
    e.entity_type,
    e.description,
    e.source_count,
    e.project,
    g.depth
FROM entity_graph g
JOIN entities e ON e.id = g.entity_id
ORDER BY g.depth, e.source_count DESC
""")

_RELATIONSHIPS_FOR_GRAPH_SQL = text("""
WITH RECURSIVE entity_graph(entity_id) AS (
    SELECT :entity_id::uuid

    UNION

    SELECT
        CASE
            WHEN r.source_entity_id = g.entity_id THEN r.target_entity_id
            ELSE r.source_entity_id
        END
    FROM relationships r
    JOIN entity_graph g ON (
        r.source_entity_id = g.entity_id OR r.target_entity_id = g.entity_id
    )
    WHERE (
        SELECT count(*) FROM entity_graph
    ) <= 50
)
SELECT r.id, r.source_entity_id, r.target_entity_id,
       r.relation_type, r.weight, r.evidence, r.created_at
FROM relationships r
WHERE r.source_entity_id IN (SELECT entity_id FROM entity_graph)
   OR r.target_entity_id IN (SELECT entity_id FROM entity_graph)
LIMIT 200
""")


async def search_entities(
    db_url: str,
    q: str,
    projects: list[str],
    limit: int = 20,
) -> list[dict]:
    """Full-text-like entity search by name/alias prefix."""
    q_lower = f"%{q.lower()}%"
    async with get_session(db_url) as session:
        result = await session.execute(
            _ENTITY_SEARCH_SQL,
            {"projects": projects, "q": q_lower, "q_raw": f"%{q}%", "limit": limit},
        )
        rows = result.fetchall()

    return [
        {
            "id": str(row[0]),
            "name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "aliases": row[4] or [],
            "source_count": row[5],
            "project": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
        }
        for row in rows
    ]


async def get_entity_graph(
    db_url: str,
    entity_id: str,
    max_depth: int = 2,
) -> dict:
    """Return entity + all nodes/edges within max_depth hops."""
    eid = uuid.UUID(entity_id)

    async with get_session(db_url) as session:
        # Nodes
        node_result = await session.execute(
            _ENTITY_GRAPH_SQL,
            {"entity_id": eid, "max_depth": max_depth},
        )
        node_rows = node_result.fetchall()

        # Edges — use simpler query for small graphs
        edge_result = await session.execute(
            text("""
                SELECT r.id, r.source_entity_id, r.target_entity_id,
                       r.relation_type, r.weight, r.evidence, r.created_at
                FROM relationships r
                WHERE r.source_entity_id = ANY(:ids) OR r.target_entity_id = ANY(:ids)
                LIMIT 200
            """),
            {"ids": [row[0] for row in node_rows]},
        )
        edge_rows = edge_result.fetchall()

    nodes = [
        {
            "id": str(row[0]),
            "name": row[1],
            "entity_type": row[2],
            "description": row[3],
            "source_count": row[4],
            "project": row[5],
            "depth": row[6],
        }
        for row in node_rows
    ]

    edges = [
        {
            "id": str(row[0]),
            "source": str(row[1]),
            "target": str(row[2]),
            "relation_type": row[3],
            "weight": row[4],
            "evidence": row[5] or [],
            "created_at": row[6].isoformat() if row[6] else None,
        }
        for row in edge_rows
    ]

    return {"nodes": nodes, "edges": edges}
