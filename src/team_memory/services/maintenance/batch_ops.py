"""BatchOpsService: bulk read/write helpers for backfill CLI scripts.

Provides cursor-paginated queries for finding records missing embeddings,
loading entity maps, and updating embeddings/solutions in bulk.
All methods use raw SQL via sqlalchemy.text() to avoid ORM model imports
and keep the queries explicit.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text

from team_memory.storage.database import get_session


class BatchOpsService:
    """Service for bulk read/write operations used by backfill CLI scripts.

    Lives in services/ so it is allowed to reach the storage layer directly.
    All queries use raw sqlalchemy.text() for simplicity and portability.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    # ------------------------------------------------------------------
    # Experiences – missing embedding
    # ------------------------------------------------------------------

    async def list_experiences_missing_embedding(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]:
        """Return experiences with embedding IS NULL, ordered by created_at ASC.

        Supports cursor-based pagination: pass the last returned id as cursor_id
        to fetch the next page.
        """
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, title, description, solution, tags, project, created_at "
                "FROM experiences "
                "WHERE embedding IS NULL AND is_deleted = FALSE"
            )
            params: dict = {"limit": limit}
            if cursor_id is not None:
                sql += " AND id > :cursor::uuid"
                params["cursor"] = cursor_id
            sql += " ORDER BY created_at ASC LIMIT :limit"
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Archives – missing embedding
    # ------------------------------------------------------------------

    async def list_archives_missing_embedding(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]:
        """Return archives with embedding IS NULL."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, title, overview, solution_doc, project, created_at "
                "FROM archives "
                "WHERE embedding IS NULL"
            )
            params: dict = {"limit": limit}
            if cursor_id is not None:
                sql += " AND id > :cursor::uuid"
                params["cursor"] = cursor_id
            sql += " ORDER BY created_at ASC LIMIT :limit"
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Personal memories – missing embedding
    # ------------------------------------------------------------------

    async def list_personal_memories_missing_embedding(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]:
        """Return personal memories with embedding IS NULL."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, user_id, content, profile_kind, created_at "
                "FROM personal_memories "
                "WHERE embedding IS NULL"
            )
            params: dict = {"limit": limit}
            if cursor_id is not None:
                sql += " AND id > :cursor::uuid"
                params["cursor"] = cursor_id
            sql += " ORDER BY created_at ASC LIMIT :limit"
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Update embeddings
    # ------------------------------------------------------------------

    async def update_experience_embedding(
        self, experience_id: str, embedding: list[float]
    ) -> None:
        """UPDATE experiences SET embedding = :e WHERE id = :id."""
        async with get_session(self._db_url) as session:
            await session.execute(
                text(
                    "UPDATE experiences SET embedding = :e WHERE id = :id::uuid"
                ),
                {"e": embedding, "id": experience_id},
            )

    async def update_archive_embedding(
        self, archive_id: str, embedding: list[float]
    ) -> None:
        """UPDATE archives SET embedding = :e WHERE id = :id."""
        async with get_session(self._db_url) as session:
            await session.execute(
                text(
                    "UPDATE archives SET embedding = :e WHERE id = :id::uuid"
                ),
                {"e": embedding, "id": archive_id},
            )

    # ------------------------------------------------------------------
    # Published / promoted records (for graph backfill)
    # ------------------------------------------------------------------

    async def list_published_experiences(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]:
        """Return published or promoted experiences for graph backfill."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, title, description, solution, tags, project, "
                "exp_status, quality_tier, created_at "
                "FROM experiences "
                "WHERE exp_status IN ('published', 'promoted') AND is_deleted = FALSE"
            )
            params: dict = {"limit": limit}
            if cursor_id is not None:
                sql += " AND id > :cursor::uuid"
                params["cursor"] = cursor_id
            sql += " ORDER BY created_at ASC LIMIT :limit"
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    async def list_published_archives(
        self, limit: int = 50, cursor_id: str | None = None
    ) -> list[dict]:
        """Return published archives for graph backfill."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, title, overview, solution_doc, project, created_at "
                "FROM archives "
                "WHERE status IN ('published', 'active')"
            )
            params: dict = {"limit": limit}
            if cursor_id is not None:
                sql += " AND id > :cursor::uuid"
                params["cursor"] = cursor_id
            sql += " ORDER BY created_at ASC LIMIT :limit"
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Entity / experience map (wiki compilation)
    # ------------------------------------------------------------------

    async def load_experience_entity_map(self) -> dict[str, list[str]]:
        """Return {experience_id_str: [entity_name, ...]} for wiki compilation.

        Query joins entities with entity_experience_association to build
        a full map in a single round-trip.
        """
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT ee.experience_id, e.name "
                "FROM entities e "
                "JOIN entity_experience_association ee ON e.id = ee.entity_id"
            )
            result = await session.execute(text(sql))
            rows = result.all()
            mapping: dict[str, list[str]] = defaultdict(list)
            for experience_id, name in rows:
                mapping[str(experience_id)].append(name)
            return dict(mapping)

    # ------------------------------------------------------------------
    # Wiki generation
    # ------------------------------------------------------------------

    async def list_experiences_for_wiki(
        self, project: str | None = None
    ) -> list[dict]:
        """Return all published experiences for wiki generation."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, title, description, solution, project, "
                "exp_status, quality_tier, group_key, tags, created_at "
                "FROM experiences "
                "WHERE exp_status IN ('published', 'promoted') AND is_deleted = FALSE"
            )
            params: dict = {}
            if project is not None:
                sql += " AND project = :project"
                params["project"] = project
            sql += " ORDER BY created_at ASC"
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Entities – missing embedding
    # ------------------------------------------------------------------

    async def list_entities_missing_embedding(
        self, limit: int = 50
    ) -> list[dict]:
        """Return entities with embedding IS NULL."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, name, entity_type, description, project "
                "FROM entities "
                "WHERE embedding IS NULL "
                "ORDER BY name ASC LIMIT :limit"
            )
            result = await session.execute(text(sql), {"limit": limit})
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    async def update_entity_embedding(
        self, entity_id: str, embedding: list[float]
    ) -> None:
        """UPDATE entities SET embedding = :e WHERE id = :id."""
        async with get_session(self._db_url) as session:
            await session.execute(
                text(
                    "UPDATE entities SET embedding = :e WHERE id = :id::uuid"
                ),
                {"e": embedding, "id": entity_id},
            )

    # ------------------------------------------------------------------
    # Experiences – missing solution
    # ------------------------------------------------------------------

    async def list_experiences_missing_solution(
        self, limit: int = 50
    ) -> list[dict]:
        """Return experiences where solution IS NULL or empty."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, title, description, source, project, created_at "
                "FROM experiences "
                "WHERE (solution IS NULL OR solution = '') AND is_deleted = FALSE "
                "ORDER BY created_at ASC LIMIT :limit"
            )
            result = await session.execute(text(sql), {"limit": limit})
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    async def update_experience_solution(
        self, experience_id: str, solution: str
    ) -> None:
        """UPDATE experiences SET solution = :s WHERE id = :id."""
        async with get_session(self._db_url) as session:
            await session.execute(
                text(
                    "UPDATE experiences SET solution = :s WHERE id = :id::uuid"
                ),
                {"s": solution, "id": experience_id},
            )

    # ------------------------------------------------------------------
    # Personal memories – update embedding
    # ------------------------------------------------------------------

    async def update_personal_memory_embedding(
        self, memory_id: str, embedding: list[float]
    ) -> None:
        """UPDATE personal_memories SET embedding = :e WHERE id = :id."""
        async with get_session(self._db_url) as session:
            await session.execute(
                text(
                    "UPDATE personal_memories SET embedding = :e WHERE id = :id::uuid"
                ),
                {"e": embedding, "id": memory_id},
            )

    # ------------------------------------------------------------------
    # Entity deduplication
    # ------------------------------------------------------------------

    async def list_experiences_without_entity_links(
        self, limit: int = 0
    ) -> list[dict]:
        """Return published experiences that have no entity links."""
        async with get_session(self._db_url) as session:
            sql = (
                "SELECT id, title, description, solution, tags, project, updated_at "
                "FROM experiences "
                "WHERE exp_status = 'published' AND is_deleted = FALSE "
                "AND NOT EXISTS ("
                "    SELECT 1 FROM experience_entities ee "
                "    WHERE ee.experience_id = experiences.id"
                ") "
                "ORDER BY updated_at DESC"
            )
            params: dict = {}
            if limit > 0:
                sql += " LIMIT :limit"
                params["limit"] = limit
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            return [dict(r) for r in rows]

    async def find_duplicate_entities(self) -> list[dict]:
        """Return entity groups with same name (case-insensitive) appearing more than once."""
        async with get_session(self._db_url) as session:
            result = await session.execute(
                text("""
                SELECT lower(name) as lower_name,
                       array_agg(id ORDER BY source_count DESC) as ids,
                       array_agg(name ORDER BY source_count DESC) as names,
                       array_agg(project ORDER BY source_count DESC) as projects,
                       array_agg(source_count ORDER BY source_count DESC) as counts
                FROM entities
                GROUP BY lower(name)
                HAVING count(*) > 1
                ORDER BY count(*) DESC
                """)
            )
            return [
                {
                    "lower_name": row[0],
                    "ids": row[1],
                    "names": row[2],
                    "projects": row[3],
                    "counts": row[4],
                }
                for row in result.fetchall()
            ]

    async def merge_entity_group(
        self, primary_id: str, duplicate_ids: list[str]
    ) -> int:
        """Merge duplicate entities into primary. Returns affected row count."""
        affected = 0
        async with get_session(self._db_url) as session:
            for dup_id in duplicate_ids:
                r = await session.execute(
                    text("""
                    UPDATE experience_entities
                    SET entity_id = :primary_id
                    WHERE entity_id = :dup_id
                    AND NOT EXISTS (
                        SELECT 1 FROM experience_entities ee2
                        WHERE ee2.experience_id = experience_entities.experience_id
                        AND ee2.entity_id = :primary_id
                    )
                    """),
                    {"primary_id": primary_id, "dup_id": dup_id},
                )
                affected += r.rowcount
                await session.execute(
                    text("DELETE FROM experience_entities WHERE entity_id = :dup_id"),
                    {"dup_id": dup_id},
                )
                await session.execute(
                    text("""
                    DELETE FROM relationships r1
                    WHERE r1.source_entity_id = :primary_id
                    AND r1.relation_type IN (
                        SELECT r2.relation_type FROM relationships r2
                        WHERE r2.source_entity_id = :dup_id
                        AND r2.target_entity_id = r1.target_entity_id
                    )
                    """),
                    {"primary_id": primary_id, "dup_id": dup_id},
                )
                await session.execute(
                    text("""
                    DELETE FROM relationships r1
                    WHERE r1.target_entity_id = :primary_id
                    AND r1.relation_type IN (
                        SELECT r2.relation_type FROM relationships r2
                        WHERE r2.target_entity_id = :dup_id
                        AND r2.source_entity_id = r1.source_entity_id
                    )
                    """),
                    {"primary_id": primary_id, "dup_id": dup_id},
                )
                await session.execute(
                    text("UPDATE relationships SET source_entity_id = :primary_id WHERE source_entity_id = :dup_id"),
                    {"primary_id": primary_id, "dup_id": dup_id},
                )
                await session.execute(
                    text("UPDATE relationships SET target_entity_id = :primary_id WHERE target_entity_id = :dup_id"),
                    {"primary_id": primary_id, "dup_id": dup_id},
                )
                await session.execute(
                    text("DELETE FROM entities WHERE id = :dup_id"),
                    {"dup_id": dup_id},
                )
                await session.execute(
                    text("UPDATE entities SET source_count = source_count + 1 WHERE id = :primary_id"),
                    {"primary_id": primary_id},
                )
                affected += 1
            await session.commit()
        return affected

    async def count_entities(self) -> int:
        """Return total entity count."""
        async with get_session(self._db_url) as session:
            result = await session.execute(text("SELECT count(*) FROM entities"))
            return result.scalar() or 0
