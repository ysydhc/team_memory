#!/usr/bin/env python3
"""Re-embed all experiences with updated embedding text construction.

Rebuilds embeddings to include tags and (for parents) children titles.
Uses the existing EmbeddingQueue for async batch processing.

Usage:
    python -m scripts.re_embed [--batch-size 50] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select, update

from team_memory.config import get_settings
from team_memory.embedding import create_embedding_provider
from team_memory.storage.database import get_session
from team_memory.storage.models import Experience

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_embed_text(exp: Experience) -> str:
    """Build embedding text matching the updated save() logic."""
    parts = [exp.title or "", exp.description or "", exp.solution or ""]
    if exp.root_cause:
        parts.append(exp.root_cause)
    if exp.code_snippets:
        parts.append(exp.code_snippets)
    if exp.tags:
        parts.append(" ".join(exp.tags))

    if exp.children:
        child_titles = [c.title for c in exp.children if c.title]
        if child_titles:
            parts.append(" ".join(child_titles))

    return "\n".join(parts)


async def re_embed_batch(
    db_url: str,
    batch_size: int = 50,
    dry_run: bool = False,
) -> int:
    """Re-embed all experiences in batches."""
    settings = get_settings()
    provider = create_embedding_provider(settings.embedding)

    processed = 0
    offset = 0

    while True:
        async with get_session(db_url) as session:
            q = (
                select(Experience)
                .where(Experience.is_deleted == False)  # noqa: E712
                .order_by(Experience.created_at)
                .offset(offset)
                .limit(batch_size)
            )
            result = await session.execute(q)
            experiences = list(result.scalars().all())

            if not experiences:
                break

            for exp in experiences:
                embed_text = build_embed_text(exp)

                if dry_run:
                    logger.info(
                        "[DRY-RUN] Would re-embed %s: %s (text=%d chars)",
                        exp.id, exp.title, len(embed_text),
                    )
                    processed += 1
                    continue

                try:
                    embedding = await provider.encode_single(embed_text)
                    await session.execute(
                        update(Experience)
                        .where(Experience.id == exp.id)
                        .values(embedding=embedding, embedding_status="ready")
                    )
                    processed += 1
                    if processed % 10 == 0:
                        logger.info("Re-embedded %d experiences...", processed)
                except Exception:
                    logger.warning("Failed to re-embed %s: %s", exp.id, exp.title)

            if not dry_run:
                await session.commit()

            offset += batch_size

    logger.info("Done. Total re-embedded: %d", processed)
    return processed


def main():
    parser = argparse.ArgumentParser(description="Re-embed all experiences")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    db_url = settings.database.url

    asyncio.run(re_embed_batch(db_url, args.batch_size, args.dry_run))


if __name__ == "__main__":
    main()
