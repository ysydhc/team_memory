"""Entity search service (L2.5).

Two responsibilities:
1. Rule-based entity detection from a query string (no LLM, low latency).
2. PostgreSQL recursive CTE graph traversal to find related experience IDs.

These are used by SearchPipeline to enrich search results with entity-graph
context — surfaces experiences that are related via the entity graph even if
they don't match the query directly by vector/FTS.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import text

from team_memory.storage.database import get_session

logger = logging.getLogger("team_memory.entity_search")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Rule-based entity extractor (used on the query path, no LLM)
# ---------------------------------------------------------------------------

# Minimum length for an extracted entity name
_MIN_ENTITY_LEN = 2

# Patterns that look like tool/project names: CamelCase, ALLCAPS (≥2 chars),
# hyphenated-words, snake_case identifiers, or known keyword lists.
_CAMEL_PATTERN = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")  # CamelCase
_UPPER_PATTERN = re.compile(r"\b[A-Z]{2,}\b")  # ALLCAPS
_HYPHEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)+\b")  # kebab-case
_SNAKE_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b")  # snake_case
_DOT_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+\b")  # dotted.id
_BACKTICK_PATTERN = re.compile(r"`([^`]{2,40})`")

# Built-in lowercase technical entity lexicon for rule-based extraction
_BUILTIN_TECH_ENTITIES: frozenset[str] = frozenset(
    {
        "pgvector",
        "postgres",
        "postgresql",
        "redis",
        "alembic",
        "sqlalchemy",
        "asyncpg",
        "fastapi",
        "pydantic",
        "ollama",
        "litellm",
        "docker",
        "pytest",
        "ruff",
        "jieba",
        "hnsw",
        "rrf",
        "mcp",
        "celery",
        "nginx",
        "kubernetes",
        "k8s",
        "openai",
        "claude",
        "cursor",
        "hermes",
        "daemon",
        "webhook",
        "oauth",
        "jwt",
        "tm",
        "tm-cli",
        "status",
        "hook",
        "hooks",
        "wiki",
        "sqlite",
        "fts",
        "reranker",
        "embedding",
        "janitor",
        "oppo",
        "vivo",
        "xiaomi",
        "huawei",
        "adb",
        "frida",
        "mitmproxy",
        "webview",
        "mraid",
        "vast",
        "omsdk",
        "banner",
        "interstitial",
        "rewarded",
        "sdk",
        "adx",
        "dsp",
        "mediago",
        "applovin",
        "max",
        "bidding",
        "ecpm",
        "ctr",
        "cvr",
        "logcat",
        "uiautomator",
        "phone_pilot",
        "gradle",
        "kotlin",
        "swift",
        "objc",
        "unity",
        "tuanjie",
        "ragdoll",
        "firebase",
        "getapps",
        "cli",
    }
)

# Chinese compound technical term pattern (2-6 CJK prefix chars + domain suffix)
_CJK_TECH_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,6}(?:检索|图谱|连接池|迁移|缓存|重排|实体|索引|鉴权|死锁|超时|内存泄漏|中间件|网关|会话|向量|配置|管线|队列|召回|链路|节点|状态|命令行|广告|插屏|激励|横幅|抓包|真机|模拟器|数据库|钩子|监控|面板)"
)
_CJK_SPLIT_PATTERN = re.compile(
    r"(?:如何解决|如何排查|如何|怎么解决|怎么排查|怎么|为什么|排查|解决|修复|处理|优化|导致|出现|遇到|关于|使用|通过|"
    r"改成|可以|手动|自动|进行|目前|我们|现在|查看|看看|验证|一下|如果|完全|无法|没有|需要|帮我|梳理|整个|哪些|"
    r"平时|知道|做成|持续|观察|最近|选中|某个|打印|出来|"
    r"与|和|以及|及|或者|或|的|了|在|对|被|慢|延迟|问题|故障|错误|异常|失败)+"
)

_ENTITY_LEXICON_CACHE: dict[str, tuple[float, set[str]]] = {}
_ENTITY_LEXICON_TTL_SEC = 300.0


async def get_cached_entity_lexicon(db_url: str) -> set[str]:
    """Load lowercase entity names from PostgreSQL with a 300s in-memory TTL cache."""
    import time as _time

    if not db_url:
        return set()
    now = _time.monotonic()
    cached = _ENTITY_LEXICON_CACHE.get(db_url)
    if cached and (now - cached[0]) < _ENTITY_LEXICON_TTL_SEC:
        return cached[1]

    try:
        async with get_session(db_url) as session:
            result = await session.execute(
                text("SELECT DISTINCT lower(name) FROM entities WHERE length(name) >= 2")
            )
            names = {str(r[0]).strip().lower() for r in result.fetchall() if r[0]}
            _ENTITY_LEXICON_CACHE[db_url] = (now, names)
            return names
    except Exception as exc:
        logger.debug("get_cached_entity_lexicon failed (non-critical): %s", exc)
        return cached[1] if cached else set()


def extract_entities_from_query(
    query: str,
    lexicon: set[str] | list[str] | None = None,
) -> list[str]:
    """Extract candidate entity names from a search query using rules only.

    Returns a deduplicated list of candidate entity names (lowered canonical).
    Intentionally lenient — false positives are cheaper than false negatives here.
    """
    if not query:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        cleaned = token.strip()
        if len(cleaned) >= _MIN_ENTITY_LEN and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)

    # CamelCase tokens (LiteLLM, ClaudeCode, etc.)
    for m in _CAMEL_PATTERN.finditer(query):
        _add(m.group())

    # ALLCAPS tokens (MCP, API, SDK, etc.)
    for m in _UPPER_PATTERN.finditer(query):
        _add(m.group())

    # kebab-case tokens (litellm-proxy, claude-code, etc.)
    for m in _HYPHEN_PATTERN.finditer(query):
        _add(m.group())

    # snake_case and dotted technical identifiers
    for m in _SNAKE_PATTERN.finditer(query):
        _add(m.group())
    for m in _DOT_PATTERN.finditer(query):
        _add(m.group())

    # Multi-word double-quoted phrases if present: "LiteLLM proxy"
    for phrase in re.findall(r'"([^"]{2,50})"', query):
        _add(phrase)

    # Backtick-quoted terms: `pgvector`
    for phrase in _BACKTICK_PATTERN.findall(query):
        _add(phrase)

    # Built-in + custom lowercase technical lexicon (whole-word case-insensitive match)
    custom_lex: set[str] = set()
    if lexicon:
        custom_lex = {t.strip().lower() for t in lexicon if t and len(t.strip()) >= _MIN_ENTITY_LEN}
    active_lexicon: set[str] = set(_BUILTIN_TECH_ENTITIES) | custom_lex

    ascii_words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    for w in ascii_words:
        if w in active_lexicon:
            _add(w)

    # Multi-token, hyphenated, or CJK custom lexicon entries
    query_lower = query.lower()
    non_ascii_or_compound = [
        term for term in active_lexicon if (not term.isascii() or not term.isalnum()) and term in query_lower
    ]
    non_ascii_or_compound.sort(key=len, reverse=True)
    for term in non_ascii_or_compound[:12]:
        _add(term)

    # Chinese compound technical terms (split on query filler/conjunctions first)
    for seg in _CJK_SPLIT_PATTERN.split(query):
        seg = seg.strip()
        if not seg:
            continue
        for m in _CJK_TECH_PATTERN.finditer(seg):
            _add(m.group())

    # Expand extracted root tokens against caller-provided DB lexicon so compound
    # entities in PG (e.g. "status" -> "status_monitor", "tm" -> "tm-cli") match SQL equality.
    if custom_lex and candidates:
        base_candidates = list(candidates)
        for cand in base_candidates:
            cand_l = cand.lower()
            if len(cand_l) < 2:
                continue
            ascii_pat = (
                re.compile(rf"(?<![a-z0-9]){re.escape(cand_l)}(?![a-z0-9])")
                if cand_l.isascii()
                else None
            )
            matched_sub = 0
            for db_ent in sorted(custom_lex):
                if db_ent == cand_l or db_ent in seen:
                    continue
                if re.search(r"-[0-9a-f]{6,}$", db_ent):
                    continue
                if ascii_pat is not None:
                    is_match = bool(ascii_pat.search(db_ent))
                else:
                    is_match = cand_l in db_ent
                if is_match:
                    _add(db_ent)
                    matched_sub += 1
                    if matched_sub >= 3:
                        break

    return candidates


def compute_ppr_edge_weight(
    edge_weight: float,
    source_degree: int,
    target_degree: int,
    hop_depth: int,
    damping: float = 0.5,
    cross_project: bool = False,
    cross_project_discount: float = 0.85,
) -> float:
    """Compute specificity-weighted (degree-penalized) PPR score across a graph hop.

    Penalizes high-degree generic hub entities by 1 / sqrt(deg(u) * deg(v)),
    applies exponential hop damping (default 0.5^hop_depth), and applies a 0.85x
    discount when traversing across project boundaries.
    """
    denom = math.sqrt(max(1, int(source_degree)) * max(1, int(target_degree)))
    base = (max(0.1, float(edge_weight)) / denom) * (damping ** max(0, int(hop_depth)))
    if cross_project:
        base *= cross_project_discount
    return base


def run_lightweight_ppr(
    *,
    seed_entities: list[str],
    entities: list[dict],
    relationships: list[dict],
    experience_entities: list[dict],
    primary_projects: list[str] | None = None,
    max_hops: int = 3,
    damping: float = 0.55,
    cross_project_discount: float = 0.85,
) -> list[tuple[str, float]]:
    """Run 2~3 step lightweight Personalized PageRank (HippoRAG-style) over an entity graph.

    - Filters out expired bi-temporal edges (``valid_until is not None``).
    - Anchors on ``seed_entities`` (matching name or aliases, case-insensitive).
      Seeds in ``primary_projects`` receive full anchor weight; cross-project seeds receive
      ``cross_project_discount`` (0.85x).
    - Bridges same-name entities across projects during hop expansion with ``0.85x`` discount.
    - Maps converged entity PPR scores back to ``(experience_id, ppr_score)`` sorted descending.
    """
    if not seed_entities or not entities:
        return []

    seed_set = {s.strip().lower() for s in seed_entities if s and s.strip()}
    if not seed_set:
        return []

    primary_proj_set = set(primary_projects or ["*"])
    wildcard_proj = "*" in primary_proj_set or "all" in primary_proj_set

    # Index entities by id and normalized name
    ent_by_id: dict[str, dict] = {}
    ids_by_name: dict[str, list[str]] = {}
    for ent in entities:
        eid = str(ent.get("id", ""))
        if not eid:
            continue
        ent_by_id[eid] = ent
        ename = str(ent.get("name", "")).strip().lower()
        if ename:
            ids_by_name.setdefault(ename, []).append(eid)
        for alias in ent.get("aliases") or []:
            al = str(alias).strip().lower()
            if al:
                ids_by_name.setdefault(al, []).append(eid)

    # Filter active relationships (valid_until IS NULL)
    active_rels: list[dict] = [
        r for r in relationships if r.get("valid_until") is None
    ]

    # Compute node degrees (experience links + active incident edges)
    deg: dict[str, int] = {eid: max(1, int(ent.get("source_count") or 1)) for eid, ent in ent_by_id.items()}
    adj: dict[str, list[tuple[str, float, bool]]] = {eid: [] for eid in ent_by_id}

    for r in active_rels:
        u = str(r.get("source_entity_id") or r.get("source") or "")
        v = str(r.get("target_entity_id") or r.get("target") or "")
        if u not in ent_by_id or v not in ent_by_id:
            continue
        w = float(r.get("weight") or 1.0)
        deg[u] = deg.get(u, 1) + 1
        deg[v] = deg.get(v, 1) + 1
        cross = ent_by_id[u].get("project") != ent_by_id[v].get("project")
        adj[u].append((v, w, cross))
        adj[v].append((u, w, cross))

    # Add virtual cross-project bridges for identical entity names across repos
    for _ename, sibling_ids in ids_by_name.items():
        if len(sibling_ids) > 1:
            for i in range(len(sibling_ids)):
                for j in range(i + 1, len(sibling_ids)):
                    u, v = sibling_ids[i], sibling_ids[j]
                    if ent_by_id[u].get("project") != ent_by_id[v].get("project"):
                        adj[u].append((v, 1.0, True))
                        adj[v].append((u, 1.0, True))

    # Initialize seed scores (Hop 0)
    node_scores: dict[str, float] = {}
    for s_name in seed_set:
        for eid in ids_by_name.get(s_name, []):
            ent = ent_by_id[eid]
            proj = str(ent.get("project") or "default")
            in_primary = wildcard_proj or proj in primary_proj_set
            discount = 1.0 if in_primary else cross_project_discount
            init_score = (1.0 / math.sqrt(max(1, deg.get(eid, 1)))) * discount
            if init_score > node_scores.get(eid, 0.0):
                node_scores[eid] = init_score

    if not node_scores:
        return []

    # Multi-hop propagation (2~3 steps)
    frontier: dict[str, float] = dict(node_scores)
    for hop in range(1, max(1, int(max_hops)) + 1):
        next_frontier: dict[str, float] = {}
        for u, u_score in frontier.items():
            u_proj = str(ent_by_id[u].get("project") or "default")
            for v, edge_w, is_cross_edge in adj.get(u, []):
                v_proj = str(ent_by_id[v].get("project") or "default")
                is_external_proj = not wildcard_proj and (
                    v_proj not in primary_proj_set or is_cross_edge or u_proj != v_proj
                )
                hop_w = compute_ppr_edge_weight(
                    edge_weight=edge_w,
                    source_degree=deg.get(u, 1),
                    target_degree=deg.get(v, 1),
                    hop_depth=hop,
                    damping=damping,
                    cross_project=is_external_proj,
                    cross_project_discount=cross_project_discount,
                )
                propagated = u_score * hop_w
                if propagated > next_frontier.get(v, 0.0):
                    next_frontier[v] = propagated
                if propagated > node_scores.get(v, 0.0):
                    node_scores[v] = propagated
        if not next_frontier:
            break
        frontier = next_frontier

    # Map entity PPR scores to Experience IDs
    exp_scores: dict[str, float] = {}
    for link in experience_entities:
        eid = str(link.get("entity_id") or "")
        exp_id = str(link.get("experience_id") or "")
        if not eid or not exp_id or eid not in node_scores:
            continue
        ent_proj = str(ent_by_id[eid].get("project") or "default")
        proj_factor = 1.0 if (wildcard_proj or ent_proj in primary_proj_set) else cross_project_discount
        exp_scores[exp_id] = min(1.0, exp_scores.get(exp_id, 0.0) + node_scores[eid] * proj_factor)

    return sorted(exp_scores.items(), key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------

_GRAPH_CTE_SQL = text("""
WITH RECURSIVE entity_graph(entity_id, depth) AS (
    -- anchor: entities matching any of the given names (case-insensitive)
    SELECT e.id, 0
    FROM entities e
    WHERE lower(e.name) = ANY(:names)
      AND (e.project = ANY(:projects) OR '*' = ANY(:projects))

    UNION

    -- expansion: follow active outgoing + incoming edges up to :max_depth hops
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
      AND r.valid_until IS NULL
)
SELECT DISTINCT ee.experience_id
FROM entity_graph g
JOIN experience_entities ee ON ee.entity_id = g.entity_id
""")

_WEIGHTED_GRAPH_CTE_SQL = text("""
WITH RECURSIVE entity_deg AS (
    SELECT
        e.id,
        lower(e.name) AS norm_name,
        e.project,
        GREATEST(
            1,
            COALESCE(e.source_count, 1),
            (
                COALESCE((SELECT COUNT(*) FROM experience_entities ee WHERE ee.entity_id = e.id), 0)
                + COALESCE(
                    (
                        SELECT COUNT(*)
                        FROM relationships r
                        WHERE (r.source_entity_id = e.id OR r.target_entity_id = e.id)
                          AND r.valid_until IS NULL
                    ),
                    0
                )
            )
        )::float8 AS deg
    FROM entities e
),
entity_graph(anchor_id, entity_id, norm_name, entity_project, depth, score, path) AS (
    -- anchor: entities matching any of the given names or aliases (case-insensitive),
    -- penalized by 1 / sqrt(deg); primary project gets 1.0x, cross-project seed gets 0.85x
    SELECT
        e.id AS anchor_id,
        e.id AS entity_id,
        ed.norm_name AS norm_name,
        e.project AS entity_project,
        0 AS depth,
        (
            (1.0 / GREATEST(1.0, sqrt(ed.deg)))
            * CASE
                WHEN (e.project = ANY(:projects) OR '*' = ANY(:projects)) THEN 1.0
                ELSE 0.85
              END
        )::float8 AS score,
        ARRAY[e.id] AS path
    FROM entities e
    JOIN entity_deg ed ON ed.id = e.id
    WHERE (
        lower(e.name) = ANY(:names)
        OR EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(
                CASE
                    WHEN jsonb_typeof(e.aliases) = 'array' THEN e.aliases
                    ELSE '[]'::jsonb
                END
            ) AS a
            WHERE lower(a) = ANY(:names)
        )
    )

    UNION ALL

    -- expansion: follow active edges up to :max_depth hops (including cross-project same-name bridge with 0.85x)
    SELECT
        g.anchor_id,
        tgt_ed.id AS entity_id,
        tgt_ed.norm_name AS norm_name,
        tgt_ed.project AS entity_project,
        g.depth + 1,
        (
            g.score * (COALESCE(r.weight, 1.0) * 0.55)
            / GREATEST(1.0, sqrt(src_ed.deg * tgt_ed.deg))
            * CASE
                WHEN (tgt_ed.project = ANY(:projects) OR '*' = ANY(:projects))
                     AND src_ed.id = g.entity_id
                THEN 1.0
                ELSE 0.85
              END
        )::float8,
        g.path || tgt_ed.id
    FROM entity_graph g
    JOIN entity_deg src_ed ON (
        src_ed.id = g.entity_id OR src_ed.norm_name = g.norm_name
    )
    JOIN relationships r ON (
        r.source_entity_id = src_ed.id OR r.target_entity_id = src_ed.id
    )
    JOIN entity_deg tgt_ed ON tgt_ed.id = (
        CASE
            WHEN r.source_entity_id = src_ed.id THEN r.target_entity_id
            ELSE r.source_entity_id
        END
    )
    WHERE g.depth < :max_depth
      AND r.valid_until IS NULL
      AND NOT (tgt_ed.id = ANY(g.path))
),
per_anchor AS (
    SELECT
        ee.experience_id,
        g.anchor_id,
        MAX(
            g.score * CASE
                WHEN (ed.project = ANY(:projects) OR '*' = ANY(:projects))
                     AND ed.id = g.entity_id
                THEN 1.0
                ELSE 0.85
            END
        ) AS anchor_score
    FROM entity_graph g
    JOIN entity_deg ed ON (ed.id = g.entity_id OR ed.norm_name = g.norm_name)
    JOIN experience_entities ee ON ee.entity_id = ed.id
    GROUP BY ee.experience_id, g.anchor_id
)
SELECT experience_id, LEAST(1.0::float8, SUM(anchor_score))::float8 AS graph_score
FROM per_anchor
GROUP BY experience_id
ORDER BY graph_score DESC
LIMIT :limit
""")


async def find_experience_ids_by_entities(
    db_url: str,
    entity_names: list[str],
    projects: list[str],
    max_depth: int = 2,
    *,
    session: AsyncSession | None = None,
) -> list[str]:
    """Return experience IDs reachable from the given entity names within max_depth hops.

    Uses a bidirectional recursive CTE so both directions of relationships
    are explored.  Returns at most 50 IDs to keep overhead bounded.
    """
    if not entity_names or not projects:
        return []

    names_lower = [n.lower() for n in entity_names if n]
    params = {
        "names": names_lower,
        "projects": projects,
        "max_depth": max_depth,
    }

    if session is not None:
        result = await session.execute(_GRAPH_CTE_SQL, params)
        rows = result.fetchall()
    else:
        async with get_session(db_url) as s:
            result = await s.execute(_GRAPH_CTE_SQL, params)
            rows = result.fetchall()

    return [str(row[0]) for row in rows[:50]]


async def find_weighted_experiences_by_entities(
    db_url: str,
    entity_names: list[str],
    projects: list[str],
    max_depth: int = 2,
    limit: int = 50,
    *,
    session: AsyncSession | None = None,
) -> list[tuple[str, float]]:
    """Return (experience_id, graph_score) pairs via degree-penalized 2-hop PPR CTE.

    Falls back to find_experience_ids_by_entities with default score 0.5 if the
    PostgreSQL-specific weighted CTE cannot be executed (e.g. SQLite or test mock).
    """
    if not entity_names or not projects:
        return []

    names_lower = list(dict.fromkeys(n.lower() for n in entity_names if n))
    if not names_lower:
        return []

    params = {
        "names": names_lower,
        "projects": projects,
        "max_depth": max_depth,
        "limit": limit,
    }

    try:
        if session is not None:
            result = await session.execute(_WEIGHTED_GRAPH_CTE_SQL, params)
            rows = result.fetchall()
        else:
            async with get_session(db_url) as s:
                result = await s.execute(_WEIGHTED_GRAPH_CTE_SQL, params)
                rows = result.fetchall()
        weighted: list[tuple[str, float]] = []
        for row in rows[:limit]:
            exp_id = str(row[0])
            score = float(row[1]) if len(row) > 1 and row[1] is not None else 0.5
            weighted.append((exp_id, score))
        return weighted
    except Exception:
        if session is not None and hasattr(session, "rollback"):
            try:
                await session.rollback()
            except Exception:
                pass
        try:
            fallback_ids = await find_experience_ids_by_entities(
                db_url=db_url,
                entity_names=entity_names,
                projects=projects,
                max_depth=max_depth,
                session=session,
            )
            return [(eid, 0.5) for eid in fallback_ids[:limit]]
        except Exception:
            if session is not None and hasattr(session, "rollback"):
                try:
                    await session.rollback()
                except Exception:
                    pass
            return []


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
      AND r.valid_until IS NULL
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
    WHERE r.valid_until IS NULL
      AND (
        SELECT count(*) FROM entity_graph
    ) <= 50
)
SELECT r.id, r.source_entity_id, r.target_entity_id,
       r.relation_type, r.weight, r.evidence, r.created_at
FROM relationships r
WHERE r.valid_until IS NULL
  AND (
    r.source_entity_id IN (SELECT entity_id FROM entity_graph)
    OR r.target_entity_id IN (SELECT entity_id FROM entity_graph)
  )
LIMIT 200
""")


async def search_entities(
    db_url: str,
    q: str,
    projects: list[str],
    limit: int = 20,
    *,
    session: AsyncSession | None = None,
) -> list[dict]:
    """Full-text-like entity search by name/alias prefix."""
    q_lower = f"%{q.lower()}%"
    params = {"projects": projects, "q": q_lower, "q_raw": f"%{q}%", "limit": limit}
    if session is not None:
        result = await session.execute(_ENTITY_SEARCH_SQL, params)
        rows = result.fetchall()
    else:
        async with get_session(db_url) as s:
            result = await s.execute(_ENTITY_SEARCH_SQL, params)
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
    *,
    session: AsyncSession | None = None,
) -> dict:
    """Return entity + all nodes/edges within max_depth hops."""
    eid = uuid.UUID(entity_id)
    edge_sql = text("""
        SELECT r.id, r.source_entity_id, r.target_entity_id,
               r.relation_type, r.weight, r.evidence, r.created_at
        FROM relationships r
        WHERE r.valid_until IS NULL
          AND (r.source_entity_id = ANY(:ids) OR r.target_entity_id = ANY(:ids))
        LIMIT 200
    """)

    if session is not None:
        node_result = await session.execute(
            _ENTITY_GRAPH_SQL,
            {"entity_id": eid, "max_depth": max_depth},
        )
        node_rows = node_result.fetchall()
        edge_result = await session.execute(
            edge_sql,
            {"ids": [row[0] for row in node_rows]},
        )
        edge_rows = edge_result.fetchall()
    else:
        async with get_session(db_url) as s:
            # Nodes
            node_result = await s.execute(
                _ENTITY_GRAPH_SQL,
                {"entity_id": eid, "max_depth": max_depth},
            )
            node_rows = node_result.fetchall()

            # Edges — use simpler query for small graphs
            edge_result = await s.execute(
                edge_sql,
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
