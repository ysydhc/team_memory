"""Query builders for experience search -- pure SQL construction, no I/O."""

from __future__ import annotations

import re

from sqlalchemy import Select, cast, desc, func, literal, or_, select, text
from sqlalchemy.dialects.postgresql import BIT
from sqlalchemy.orm import aliased
from sqlalchemy.sql.elements import TextClause

from team_memory.storage.models import Experience

# ====================== Filter helpers ======================


# Placeholder titles auto-assigned to unrefined drafts (e.g. "Draft 6f25becf", "Session e2e-fina draft")
# or synthetic topic cluster headers ("主题-..."). Exclude them from active search results.
_PLACEHOLDER_TITLE_RE = r"^(Draft [0-9a-zA-Z_-]+|Session [0-9a-zA-Z_-]+ draft|主题-.*)$"


def active_filter(
    current_user: str | None = None,
    *,
    include_superseded: bool = False,
) -> list:
    """Build visibility WHERE clauses.

    Returns a list of SQLAlchemy filter expressions:
    - is_deleted == False (always)
    - published/promoted for all; OR published/promoted/draft(own) when current_user is set
    - valid_until IS NULL (bi-temporal active filter: excludes superseded experiences unless include_superseded=True)
    - exclude placeholder-titled ("Draft <hex>") and empty-titled experiences
    """
    base = [Experience.is_deleted == False]  # noqa: E712
    if current_user:
        base.append(
            or_(
                Experience.exp_status.in_(["published", "promoted"]),
                (Experience.created_by == current_user),
            )
        )
    else:
        base.append(Experience.exp_status.in_(["published", "promoted"]))
    # Bi-temporal filter: exclude superseded/invalidated experiences from active recall.
    if not include_superseded:
        base.append(Experience.valid_until.is_(None))
    # Drop unrefined placeholders from recall (not from storage).
    base.append(func.coalesce(func.trim(Experience.title), "") != "")
    base.append(~Experience.title.op("~")(_PLACEHOLDER_TITLE_RE))
    return base


def project_value(project: str | None) -> str:
    """Normalize project value for filtering/writes."""
    if project and project.strip():
        value = project.strip()
        alias_map = {
            "team-memory": "team_memory",
            "team_doc": "team_memory",
        }
        return alias_map.get(value, value)
    return "default"


def apply_scope_filter(
    query: Select,
    project: str,
    scope: str | None,
    current_user: str | None,
) -> Select:
    """Apply project + visibility filters to a statement."""
    visibility = scope
    if visibility and visibility != "all":
        vis_map = {"personal": "private", "team": "project", "global": "global"}
        vis_val = vis_map.get(visibility, visibility)
        query = query.where(Experience.visibility == vis_val)
        if vis_val == "private" and current_user:
            query = query.where(Experience.created_by == current_user)
        if vis_val != "global":
            query = query.where(Experience.project == project)
    else:
        query = query.where(or_(Experience.project == project, Experience.visibility == "global"))
    return query


# ====================== Search query builders ======================

MIN_BQ_STAGE1_CANDIDATES = 50


def compute_bq_stage1_limit(limit: int, oversample: int = 10, offset: int = 0) -> int:
    """Compute Stage-1 HNSW candidate pool size: max((limit + offset) * oversample, 50)."""
    effective_limit = max(int(limit), 1) + max(int(offset), 0)
    effective_oversample = max(int(oversample), 1)
    return max(effective_limit * effective_oversample, MIN_BQ_STAGE1_CANDIDATES)


def resolve_binary_quantize_settings(
    binary_quantize: bool | None = None,
    oversample: int | None = None,
) -> tuple[bool, int]:
    """Resolve binary-quantize HNSW toggle and oversample from SearchConfig when omitted."""
    if binary_quantize is not None and oversample is not None:
        return bool(binary_quantize), max(int(oversample), 1)
    try:
        from team_memory.config import get_settings

        search_cfg = get_settings().search
        cfg_enabled = bool(getattr(search_cfg, "binary_quantize_index", True))
        cfg_oversample = max(int(getattr(search_cfg, "binary_quantize_oversample", 10)), 1)
    except Exception:
        cfg_enabled = True
        cfg_oversample = 10
    return (
        cfg_enabled if binary_quantize is None else bool(binary_quantize),
        cfg_oversample if oversample is None else max(int(oversample), 1),
    )


def session_is_postgres(session: object) -> bool:
    """Return True when the given SQLAlchemy session is bound to PostgreSQL."""
    sync_sess = getattr(session, "sync_session", None)
    bind = getattr(sync_sess, "bind", None) if sync_sess is not None else None
    if bind is None:
        bind = getattr(session, "bind", None)
    dialect = getattr(bind, "dialect", None)
    return getattr(dialect, "name", None) == "postgresql"


async def enable_hnsw_iterative_scan(session: object) -> bool:
    """Enable pgvector 0.8+ HNSW iterative scan for the current transaction.

    ``SET LOCAL hnsw.iterative_scan = relaxed_order`` prevents under-recall when
    PostgreSQL RLS (tenant_id) or selective WHERE filters (project, tags, status)
    filter out rows during HNSW index traversal.
    """
    if not session_is_postgres(session):
        return False
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    return True


def binary_quantize_hamming_distance(embedding_col, query_embedding: list[float]):
    """Build ``binary_quantize(col)::bit(N) <~> binary_quantize(:q)::bit(N)`` expression.

    Matches the partial HNSW expression index created by migration 028.
    """
    col_dim = getattr(getattr(embedding_col, "type", None), "dim", None)
    dim = int(col_dim) if col_dim else len(query_embedding)
    bit_type = BIT(dim)
    col_bits = cast(func.binary_quantize(embedding_col), bit_type)
    query_vec = cast(literal(query_embedding, type_=embedding_col.type), embedding_col.type)
    query_bits = cast(func.binary_quantize(query_vec), bit_type)
    return col_bits.op("<~>")(query_bits)


def build_vector_search(
    query_embedding: list[float],
    *,
    project: str | None = None,
    current_user: str | None = None,
    tags: list[str] | None = None,
    min_similarity: float = 0.6,
    limit: int = 5,
    offset: int = 0,
    include_promoted: bool = True,
    include_superseded: bool = False,
    binary_quantize: bool = True,
    oversample: int = 10,
) -> Select:
    """Build vector similarity search statement. Returns SQLAlchemy Select.

    When ``binary_quantize`` is True, builds a two-stage query:
      - Stage 1 (CTE ``bq_candidates``): uses the pgvector binary-quantized HNSW
        expression index (``binary_quantize(embedding)::bit(N) <~> ...``) to fetch
        ``max((limit + offset) * oversample, 50)`` candidates.
      - Stage 2 (outer SELECT): reranks Stage-1 candidates by exact cosine
        similarity (``1 - (embedding <=> :q)``), applies ``min_similarity``,
        and returns top ``limit`` rows (with optional ``offset``).
    """
    if binary_quantize and query_embedding:
        stage1_limit = compute_bq_stage1_limit(limit=limit, oversample=oversample, offset=offset)
        bq_distance = binary_quantize_hamming_distance(Experience.embedding, query_embedding)

        stage1_q = select(Experience).where(Experience.embedding.is_not(None))
        if project not in ("*", "all"):
            stage1_q = stage1_q.where(Experience.project == project_value(project))
        for f in active_filter(current_user, include_superseded=include_superseded):
            stage1_q = stage1_q.where(f)
        if tags:
            stage1_q = stage1_q.where(Experience.tags.overlap(tags))

        candidates_cte = stage1_q.order_by(bq_distance).limit(stage1_limit).cte("bq_candidates")
        exp_alias = aliased(Experience, candidates_cte)
        similarity_expr = (1 - exp_alias.embedding.cosine_distance(query_embedding)).label(
            "similarity"
        )

        stmt = (
            select(exp_alias, similarity_expr)
            .where(similarity_expr >= min_similarity)
            .order_by(desc(similarity_expr))
            .limit(limit)
        )
        if offset > 0:
            stmt = stmt.offset(offset)
        return stmt

    similarity_expr = (1 - Experience.embedding.cosine_distance(query_embedding)).label("similarity")

    stmt = (
        select(Experience, similarity_expr)
        .where(Experience.embedding.is_not(None))
        .where(similarity_expr >= min_similarity)
        .order_by(desc(similarity_expr))
        .limit(limit)
    )
    if project not in ("*", "all"):
        stmt = stmt.where(Experience.project == project_value(project))

    if offset > 0:
        stmt = stmt.offset(offset)

    for f in active_filter(current_user, include_superseded=include_superseded):
        stmt = stmt.where(f)

    if tags:
        stmt = stmt.where(Experience.tags.overlap(tags))

    return stmt


_FTS_STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "和",
        "与",
        "或",
        "在",
        "是",
        "有",
        "被",
        "及",
        "中",
        "后",
        "时",
        "里",
        "上",
        "下",
        "怎么",
        "如何",
        "什么",
        "为什么",
        "解决",
        "问题",
        "排查",
        "修复",
        "方法",
        "逻辑",
        "分析",
        "确认",
        "导致",
        "出现",
        "提示",
        "报错",
        "处理",
        "策略",
        "方案",
        "要点",
        "这种",
        "那个",
        "这个",
    }
)


def _build_hybrid_tsquery(tokenized_query: str):
    """Build a hybrid AND/OR tsquery.

    Strategy: first 2 terms AND (precision), remaining terms OR (recall).
    - ≤2 terms: all AND (plainto_tsquery, exact as before)
    - >2 terms: first 2 AND, rest OR

    Examples:
        "Docker PostgreSQL"             → Docker & PostgreSQL
        "Docker PostgreSQL connection"  → Docker & PostgreSQL | connection
        "embedding model configuration" → embedding & model | configuration
    """
    # Strip tsquery meta-characters to prevent syntax errors
    sanitized = re.sub(r'[!&|():*\'"\\，。？！、；：]', " ", tokenized_query)
    raw_terms = [t for t in sanitized.split() if t]
    if not raw_terms:
        return func.plainto_tsquery("simple", tokenized_query)

    filtered = [
        t
        for t in raw_terms
        if t.lower() not in _FTS_STOPWORDS and not (len(t) == 1 and ord(t) > 127)
    ]
    terms = filtered if filtered else raw_terms

    # Deduplicate while preserving order
    unique_terms: list[str] = []
    seen: set[str] = set()
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            unique_terms.append(t)

    if len(unique_terms) <= 2:
        # Short query: keep strict AND
        return func.plainto_tsquery("simple", " ".join(unique_terms))

    # Prioritize technical ASCII identifiers in the leading positions when present
    ascii_tech = [t for t in unique_terms if all(ord(c) < 128 for c in t) and len(t) >= 2]
    non_ascii = [t for t in unique_terms if t not in ascii_tech]
    ordered = ascii_tech + non_ascii if ascii_tech else unique_terms

    # Build pairwise AND clauses across top terms so a single generic trailing word
    # cannot trigger a false-positive FTS match due to PostgreSQL '&' vs '|' precedence.
    top_terms = ordered[:8]
    clauses: list[str] = []

    # Specific technical identifiers (ASCII length >= 4, snake_case, or alphanumeric with digits)
    # can also match individually to preserve high recall on code symbols / error codes.
    specific_tech = [
        t
        for t in ascii_tech
        if len(t) >= 4 or "_" in t or any(c.isdigit() for c in t)
    ][:3]
    for t in specific_tech:
        clauses.append(f"'{t}'")

    for i in range(len(top_terms)):
        for j in range(i + 1, len(top_terms)):
            clauses.append(f"('{top_terms[i]}' & '{top_terms[j]}')")

    hybrid = " | ".join(clauses)
    return func.to_tsquery("simple", hybrid)


def build_fts_search(
    tokenized_query: str,
    *,
    project: str | None = None,
    current_user: str | None = None,
    tags: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
    include_promoted: bool = True,
    include_superseded: bool = False,
) -> Select:
    """Build full-text search statement. Returns SQLAlchemy Select.

    Expects pre-tokenized query text (jieba tokenization applied by caller).
    Uses hybrid AND/OR strategy: first 2 terms AND (precision), rest OR (recall).
    """
    ts_query = _build_hybrid_tsquery(tokenized_query)
    rank_expr = func.ts_rank_cd(Experience.fts, ts_query).label("rank")

    stmt = (
        select(Experience, rank_expr)
        .where(Experience.fts.op("@@")(ts_query))
        .order_by(desc(rank_expr))
        .limit(limit)
    )
    if project not in ("*", "all"):
        stmt = stmt.where(Experience.project == project_value(project))

    if offset > 0:
        stmt = stmt.offset(offset)

    for f in active_filter(current_user, include_superseded=include_superseded):
        stmt = stmt.where(f)

    if tags:
        stmt = stmt.where(Experience.tags.overlap(tags))

    return stmt


def build_duplicate_detection(
    *,
    project: str | None = None,
    threshold: float = 0.90,
    pair_cap: int = 400,
) -> TextClause:
    """Build duplicate detection query. Returns SQLAlchemy text() statement.

    The caller must bind params: proj, threshold, pair_cap.
    """
    return text(
        """
        SELECT a.id::text AS id_a, b.id::text AS id_b,
               (1 - (a.embedding <=> b.embedding))::double precision AS similarity
        FROM experiences a
        INNER JOIN experiences b ON a.id < b.id
        WHERE a.parent_id IS NULL AND b.parent_id IS NULL
          AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
          AND a.is_deleted = false AND b.is_deleted = false
          AND a.valid_until IS NULL AND b.valid_until IS NULL
          AND a.project = :proj AND b.project = :proj
          AND a.exp_status = 'published' AND b.exp_status = 'published'
          AND (1 - (a.embedding <=> b.embedding)) >= :threshold
        ORDER BY similarity DESC
        LIMIT :pair_cap
        """
    )
