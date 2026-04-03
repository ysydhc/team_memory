"""Query builders for experience search -- pure SQL construction, no I/O."""

from __future__ import annotations

from sqlalchemy import Select, desc, func, or_, select, text
from sqlalchemy.sql.elements import TextClause

from team_memory.storage.models import Experience

# ====================== Filter helpers ======================


def active_filter(current_user: str | None = None) -> list:
    """Build visibility WHERE clauses.

    Returns a list of SQLAlchemy filter expressions:
    - is_deleted == False (always)
    - published for all; OR published + owned-by-user when current_user is set
    """
    base = [Experience.is_deleted == False]  # noqa: E712
    if current_user:
        base.append(
            or_(
                Experience.exp_status == "published",
                (Experience.created_by == current_user),
            )
        )
    else:
        base.append(Experience.exp_status == "published")
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


def build_vector_search(
    query_embedding: list[float],
    *,
    project: str | None = None,
    current_user: str | None = None,
    tags: list[str] | None = None,
    min_similarity: float = 0.6,
    limit: int = 5,
) -> Select:
    """Build vector similarity search statement. Returns SQLAlchemy Select."""
    similarity_expr = (1 - Experience.embedding.cosine_distance(query_embedding)).label(
        "similarity"
    )

    stmt = (
        select(Experience, similarity_expr)
        .where(Experience.embedding.is_not(None))
        .where(Experience.project == project_value(project))
        .where(similarity_expr >= min_similarity)
        .order_by(desc(similarity_expr))
        .limit(limit)
    )

    for f in active_filter(current_user):
        stmt = stmt.where(f)

    if tags:
        stmt = stmt.where(Experience.tags.overlap(tags))

    return stmt


def build_fts_search(
    tokenized_query: str,
    *,
    project: str | None = None,
    current_user: str | None = None,
    tags: list[str] | None = None,
    limit: int = 5,
) -> Select:
    """Build full-text search statement. Returns SQLAlchemy Select.

    Expects pre-tokenized query text (jieba tokenization applied by caller).
    """
    ts_query = func.plainto_tsquery("simple", tokenized_query)
    rank_expr = func.ts_rank_cd(Experience.fts, ts_query).label("rank")

    stmt = (
        select(Experience, rank_expr)
        .where(Experience.fts.op("@@")(ts_query))
        .where(Experience.project == project_value(project))
        .order_by(desc(rank_expr))
        .limit(limit)
    )

    for f in active_filter(current_user):
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
          AND a.project = :proj AND b.project = :proj
          AND a.exp_status = 'published' AND b.exp_status = 'published'
          AND (1 - (a.embedding <=> b.embedding)) >= :threshold
        ORDER BY similarity DESC
        LIMIT :pair_cap
        """
    )
