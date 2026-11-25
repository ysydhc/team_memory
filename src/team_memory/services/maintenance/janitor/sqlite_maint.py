"""Search log dedup, response buffer pruning, and background tasks pruning for MemoryJanitor."""

from __future__ import annotations

import logging
from datetime import timedelta

from team_memory.services.maintenance.janitor.helpers import _utcnow

logger = logging.getLogger("team_memory.services.maintenance.janitor")


class JanitorStorageMaintMixin:
    """Mixin providing search log dedup and buffer/task table pruning."""

    async def dedup_search_logs(self, project: str | None = None) -> dict:
        """Delete duplicate search logs, keeping the earliest per hour / 去重搜索日志，保留每小时最早一条.

        Returns:
            Dict with dedup statistics.
        """
        from sqlalchemy import text

        async with self._session() as session:
            result = await session.execute(
                text("""
                    WITH ranked AS (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY query, project, date_trunc('hour', created_at)
                                   ORDER BY created_at ASC
                               ) as rn
                        FROM search_logs
                        WHERE (CAST(:project AS TEXT) IS NULL OR project = :project)
                    )
                    DELETE FROM search_logs
                    WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
                """),
                {"project": project},
            )
            deleted_count = result.rowcount

            logger.info(
                "Search log dedup completed: deleted %d duplicates in project '%s'",
                deleted_count,
                project or "all",
            )

            return {
                "deleted_count": deleted_count,
                "project": project,
            }

    async def prune_response_buffer(
        self,
        older_than_days: int | None = None,
        evaluated_older_than_days: int | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Prune old or already-evaluated rows from PostgreSQL response_buffer.

        Deletes rows matching either:
        - ``evaluated = TRUE`` and ``created_at < NOW() - evaluated_older_than_days``
        - ``created_at < NOW() - older_than_days`` (hard TTL fallback)
        """
        from sqlalchemy import text

        hard_days = int(
            older_than_days
            if older_than_days is not None
            else self._get_config("response_buffer_retention_days", 14)
        )
        eval_days = int(
            evaluated_older_than_days
            if evaluated_older_than_days is not None
            else self._get_config("response_buffer_evaluated_retention_days", 7)
        )
        now = _utcnow()
        hard_cutoff = now - timedelta(days=max(1, hard_days))
        eval_cutoff = now - timedelta(days=max(1, eval_days))

        async with self._session() as session:
            where_clause = "(evaluated = TRUE AND created_at < :eval_cutoff) OR (created_at < :hard_cutoff)"
            params = {"eval_cutoff": eval_cutoff, "hard_cutoff": hard_cutoff}
            if dry_run:
                res = await session.execute(
                    text(f"SELECT COUNT(*) FROM response_buffer WHERE {where_clause}"),
                    params,
                )
                count = int(res.scalar() or 0)
            else:
                res = await session.execute(
                    text(f"DELETE FROM response_buffer WHERE {where_clause}"),
                    params,
                )
                count = int(res.rowcount or 0)
                await session.commit()

            logger.info(
                "Response buffer pruning %s: %d rows (eval>%dd, hard>%dd)",
                "dry-run" if dry_run else "completed",
                count,
                eval_days,
                hard_days,
            )
            return {
                "deleted_count": 0 if dry_run else count,
                "would_delete_count": count,
                "dry_run": dry_run,
                "older_than_days": hard_days,
                "evaluated_older_than_days": eval_days,
            }

    async def prune_background_tasks(
        self,
        older_than_days: int | None = None,
        deprecated_types: list[str] | tuple[str, ...] | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Prune deprecated task types (e.g. graphiti_ingest) and old terminal background_tasks."""
        from sqlalchemy import text

        retention_days = int(
            older_than_days
            if older_than_days is not None
            else self._get_config("background_tasks_retention_days", 14)
        )
        dep_types = list(
            deprecated_types
            if deprecated_types is not None
            else self._get_config("deprecated_background_task_types", ["graphiti_ingest"])
        )
        cutoff = _utcnow() - timedelta(days=max(1, retention_days))

        async with self._session() as session:
            deprecated_count = 0
            expired_terminal_count = 0

            if dep_types:
                if dry_run:
                    r_dep = await session.execute(
                        text("SELECT COUNT(*) FROM background_tasks WHERE task_type = ANY(:types)"),
                        {"types": dep_types},
                    )
                    deprecated_count = int(r_dep.scalar() or 0)
                else:
                    r_dep = await session.execute(
                        text("DELETE FROM background_tasks WHERE task_type = ANY(:types)"),
                        {"types": dep_types},
                    )
                    deprecated_count = int(r_dep.rowcount or 0)

            term_where = (
                "status IN ('completed', 'failed') "
                "AND COALESCE(completed_at, created_at) < :cutoff"
            )
            term_params: dict[str, object] = {"cutoff": cutoff}
            if dep_types:
                term_where += " AND NOT (task_type = ANY(:types))"
                term_params["types"] = dep_types

            if dry_run:
                r_term = await session.execute(
                    text(f"SELECT COUNT(*) FROM background_tasks WHERE {term_where}"),
                    term_params,
                )
                expired_terminal_count = int(r_term.scalar() or 0)
            else:
                r_term = await session.execute(
                    text(f"DELETE FROM background_tasks WHERE {term_where}"),
                    term_params,
                )
                expired_terminal_count = int(r_term.rowcount or 0)
                await session.commit()

            total = deprecated_count + expired_terminal_count
            logger.info(
                "Background tasks pruning %s: %d total (deprecated=%d, expired_terminal=%d, >%dd)",
                "dry-run" if dry_run else "completed",
                total,
                deprecated_count,
                expired_terminal_count,
                retention_days,
            )
            return {
                "deleted_count": 0 if dry_run else total,
                "would_delete_count": total,
                "deprecated_count": deprecated_count,
                "expired_terminal_count": expired_terminal_count,
                "deprecated_types": dep_types,
                "older_than_days": retention_days,
                "dry_run": dry_run,
            }
