"""One-time migration script: convert flat experiences to parent+child hierarchy.

For each existing non-deleted experience E:
1. E stays in place as the parent (parent_id = NULL), preserving original ID
2. A new child C is created with parent_id = E.id, copying content fields
3. C gets its own embedding, E gets a new embedding based on title + summary

Usage:
    # Dry run (preview only, no changes):
    python scripts/migrate_flat_to_parent_child.py --dry-run

    # Execute migration:
    python scripts/migrate_flat_to_parent_child.py

    # With custom database URL:
    python scripts/migrate_flat_to_parent_child.py --db-url postgresql+asyncpg://user:pass@host/db
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select

from team_memory.config import load_settings
from team_memory.embedding.base import EmbeddingProvider
from team_memory.storage.database import close_db, get_session
from team_memory.storage.models import Experience

logger = logging.getLogger("migrate_flat_to_parent_child")


def create_embedding_provider(settings) -> EmbeddingProvider:
    """Create embedding provider from settings."""
    cfg = settings.embedding
    if cfg.provider == "ollama":
        from team_memory.embedding.ollama_provider import OllamaEmbedding
        return OllamaEmbedding(
            model=cfg.ollama.model,
            dim=cfg.ollama.dimension,
            base_url=cfg.ollama.base_url,
        )
    elif cfg.provider == "openai":
        import os

        from team_memory.embedding.openai_provider import OpenAIEmbedding
        api_key = cfg.openai.api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAIEmbedding(
            api_key=api_key,
            model=cfg.openai.model,
            dim=cfg.openai.dimension,
        )
    else:
        from team_memory.embedding.local_provider import LocalEmbedding
        return LocalEmbedding(
            model_name=cfg.local.model_name,
            device=cfg.local.device,
            dim=cfg.local.dimension,
        )


async def migrate(db_url: str, dry_run: bool = True):
    """Run the flat-to-parent-child migration."""
    settings = load_settings()
    if not db_url:
        db_url = settings.database.url

    embedding = create_embedding_provider(settings)

    logger.info("Starting migration (dry_run=%s)", dry_run)
    logger.info("Database: %s", db_url.split("@")[-1])  # hide credentials

    async with get_session(db_url) as session:
        # Find all flat experiences (parent_id is NULL, no children, not deleted)
        result = await session.execute(
            select(Experience)
            .where(Experience.parent_id.is_(None))
            .where(Experience.is_deleted == False)  # noqa: E712
            .order_by(Experience.created_at)
        )
        flat_experiences = list(result.scalars().all())

        # Filter: only those that DON'T already have children
        to_migrate = []
        for exp in flat_experiences:
            children_result = await session.execute(
                select(Experience.id)
                .where(Experience.parent_id == exp.id)
                .limit(1)
            )
            if children_result.scalar_one_or_none() is None:
                to_migrate.append(exp)

        logger.info(
            "Found %d flat experiences to migrate (out of %d total root experiences)",
            len(to_migrate),
            len(flat_experiences),
        )

        if not to_migrate:
            logger.info("Nothing to migrate. Exiting.")
            return

        migrated = 0
        failed = 0

        for exp in to_migrate:
            try:
                logger.info(
                    "[%d/%d] Migrating: %s (ID: %s)",
                    migrated + failed + 1,
                    len(to_migrate),
                    exp.title[:60],
                    exp.id,
                )

                if dry_run:
                    logger.info("  [DRY RUN] Would create child and regenerate parent embedding")
                    migrated += 1
                    continue

                # Step 1: Create child with content from parent
                child = Experience(
                    id=uuid.uuid4(),
                    parent_id=exp.id,
                    title=exp.title,
                    description=exp.description,
                    root_cause=exp.root_cause,
                    solution=exp.solution,
                    tags=exp.tags,
                    programming_language=exp.programming_language,
                    framework=exp.framework,
                    code_snippets=exp.code_snippets,
                    source=exp.source,
                    source_context=exp.source_context,
                    created_by=exp.created_by,
                    publish_status=exp.publish_status,
                    review_status=exp.review_status,
                    created_at=exp.created_at,
                )

                # Step 2: Generate child embedding from content
                child_text = f"{exp.title}\n{exp.description}\n{exp.solution}"
                if exp.root_cause:
                    child_text += f"\n{exp.root_cause}"
                try:
                    child_embedding = await embedding.encode_single(child_text)
                    child.embedding = child_embedding
                except Exception:
                    logger.warning("  Failed to generate child embedding, using parent's")
                    child.embedding = exp.embedding

                session.add(child)

                # Step 3: Regenerate parent embedding (based on title + brief summary)
                parent_text = f"{exp.title}\n{exp.description[:200]}"
                try:
                    parent_embedding = await embedding.encode_single(parent_text)
                    exp.embedding = parent_embedding
                except Exception:
                    logger.warning("  Failed to regenerate parent embedding, keeping original")

                await session.flush()
                migrated += 1
                logger.info("  OK: created child %s", child.id)

            except Exception as e:
                logger.error("  FAILED: %s", str(e))
                failed += 1

        if not dry_run:
            await session.commit()

        logger.info(
            "Migration complete: %d migrated, %d failed, %d total",
            migrated,
            failed,
            len(to_migrate),
        )

    await close_db()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate flat experiences to parent-child hierarchy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview migration without making changes",
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="Database URL (defaults to config.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(migrate(db_url=args.db_url, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
