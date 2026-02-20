#!/usr/bin/env python3
"""Embedding migration tool (P3-5).

Re-generates embedding vectors for all experiences when switching
embedding models or dimensions.

Usage:
    # Incremental: only process experiences with missing/failed embeddings
    python scripts/migrate_embeddings.py

    # Full: re-process ALL experiences
    python scripts/migrate_embeddings.py --full

    # Dry run: show what would be processed
    python scripts/migrate_embeddings.py --dry-run

    # Custom batch size
    python scripts/migrate_embeddings.py --batch-size 50
"""

import argparse
import asyncio
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_embeddings")


async def main(args):
    from sqlalchemy import func, select

    from team_memory.config import load_settings
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience

    settings = load_settings()

    # Create embedding provider
    cfg = settings.embedding
    if cfg.provider == "ollama":
        from team_memory.embedding.ollama_provider import OllamaEmbedding
        embedding = OllamaEmbedding(
            model=cfg.ollama.model,
            dim=cfg.ollama.dimension,
            base_url=cfg.ollama.base_url,
        )
    elif cfg.provider == "openai":
        from team_memory.embedding.openai_provider import OpenAIEmbedding
        api_key = cfg.openai.api_key or os.environ.get("OPENAI_API_KEY", "")
        embedding = OpenAIEmbedding(
            api_key=api_key, model=cfg.openai.model, dim=cfg.openai.dimension,
        )
    else:
        from team_memory.embedding.local_provider import LocalEmbedding
        embedding = LocalEmbedding(
            model_name=cfg.local.model_name, device=cfg.local.device, dim=cfg.local.dimension,
        )

    db_url = settings.database.url

    # Count experiences to process
    async with get_session(db_url) as session:
        if args.full:
            q = select(func.count()).select_from(Experience).where(Experience.is_deleted == False)
        else:
            q = select(func.count()).select_from(Experience).where(
                Experience.is_deleted == False,
                Experience.embedding_status.in_(["pending", "failed", "ready"]),
            ).where(Experience.embedding == None)  # noqa: E711
        total = (await session.execute(q)).scalar() or 0

    logger.info("Found %d experiences to process (mode: %s)", total, "full" if args.full else "incremental")

    if args.dry_run:
        logger.info("Dry run — no changes made.")
        return

    if total == 0:
        logger.info("Nothing to process.")
        return

    processed = 0
    errors = 0
    start_time = time.time()

    offset = 0
    while offset < total:
        async with get_session(db_url) as session:
            if args.full:
                q = (
                    select(Experience)
                    .where(Experience.is_deleted == False)
                    .order_by(Experience.created_at)
                    .offset(offset)
                    .limit(args.batch_size)
                )
            else:
                q = (
                    select(Experience)
                    .where(
                        Experience.is_deleted == False,
                        Experience.embedding == None,  # noqa: E711
                    )
                    .order_by(Experience.created_at)
                    .limit(args.batch_size)
                )

            result = await session.execute(q)
            batch = result.scalars().all()

            if not batch:
                break

            for exp in batch:
                try:
                    text = f"{exp.title} {exp.description} {exp.solution}"
                    if exp.root_cause:
                        text += f" {exp.root_cause}"
                    vec = await embedding.embed(text)
                    exp.embedding = vec
                    exp.embedding_status = "ready"
                    processed += 1
                except Exception as e:
                    exp.embedding_status = "failed"
                    errors += 1
                    logger.warning("Failed to embed %s: %s", exp.id, e)

            await session.commit()

        offset += args.batch_size
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(
            "Progress: %d/%d (%.1f%%) | Errors: %d | Rate: %.1f/s",
            processed, total, (processed / total * 100), errors, rate,
        )

    elapsed = time.time() - start_time
    logger.info(
        "Done! Processed: %d | Errors: %d | Time: %.1fs",
        processed, errors, elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate embedding vectors")
    parser.add_argument("--full", action="store_true", help="Re-process ALL experiences")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size (default: 20)")
    args = parser.parse_args()
    asyncio.run(main(args))
