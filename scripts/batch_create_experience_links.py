#!/usr/bin/env python3
"""One-time batch create experience_links from existing experiences by vector similarity.

**Pre-MVP table:** the ``experience_links`` table was dropped in ``002_mvp_cleanup``.

Creates 'related' links between each experience and its top-k most similar others.
Idempotent: skips pairs that already have a link.

Usage:
    python scripts/batch_create_experience_links.py [--top-k 3] [--min-similarity 0.65] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import select

from team_memory.config import get_settings
from team_memory.storage.database import get_session
from team_memory.storage.models import Experience, ExperienceLink

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def _embedding_to_list(emb) -> list[float] | None:
    """Convert Experience.embedding (pgvector or list) to list[float]."""
    if emb is None:
        return None
    if isinstance(emb, list):
        return emb
    if hasattr(emb, "to_list"):
        return emb.to_list()
    try:
        return list(emb)
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


async def run(
    top_k: int = 3,
    min_similarity: float = 0.65,
    dry_run: bool = False,
) -> int:
    """Load experiences with embeddings, compute similar pairs, insert links."""
    settings = get_settings()
    db_url = settings.database.url
    created = 0

    async with get_session(db_url) as session:
        # Load all published, non-deleted experiences with ready embedding
        q = (
            select(Experience)
            .where(Experience.is_deleted == False)  # noqa: E712
            .where(Experience.embedding.is_not(None))
            .where(Experience.embedding_status == "ready")
            .where(Experience.exp_status == "published")
        )
        result = await session.execute(q)
        experiences = list(result.scalars().unique().all())

    if not experiences:
        logger.info("No experiences with ready embedding found.")
        return 0

    # Build id -> embedding list (and keep id for ordering)
    exp_list: list[tuple[str, list[float]]] = []
    for exp in experiences:
        vec = _embedding_to_list(exp.embedding)
        if vec:
            exp_list.append((str(exp.id), vec))

    logger.info("Loaded %d experiences with embeddings.", len(exp_list))

    # Collect (id_a, id_b) pairs with similarity >= min_similarity, top-k per exp
    pairs: set[tuple[str, str]] = set()
    for i, (id_a, vec_a) in enumerate(exp_list):
        sims: list[tuple[float, str]] = []
        for j, (id_b, vec_b) in enumerate(exp_list):
            if i == j:
                continue
            sim = _cosine_similarity(vec_a, vec_b)
            if sim >= min_similarity:
                sims.append((sim, id_b))
        sims.sort(key=lambda x: -x[0])
        for _, id_b in sims[:top_k]:
            # Normalize order so (a,b) and (b,a) collapse to one
            pair = (id_a, id_b) if id_a < id_b else (id_b, id_a)
            pairs.add(pair)

    logger.info("Candidate pairs (after top-k and min_similarity): %d", len(pairs))

    if not pairs:
        return 0

    # Load existing links to avoid duplicates
    async with get_session(db_url) as session:
        result = await session.execute(select(ExperienceLink))
        existing = result.scalars().all()
    existing_set: set[tuple[str, str]] = set()
    for link in existing:
        sid, tid = str(link.source_id), str(link.target_id)
        key = (sid, tid) if sid < tid else (tid, sid)
        existing_set.add(key)

    to_create = [(a, b) for (a, b) in pairs if (a, b) not in existing_set]
    logger.info("New links to create (after dedup): %d", len(to_create))

    if dry_run:
        for a, b in list(to_create)[:20]:
            logger.info("  would link %s <-> %s", a[:8], b[:8])
        if len(to_create) > 20:
            logger.info("  ... and %d more", len(to_create) - 20)
        return 0

    # Insert in a single session
    async with get_session(db_url) as session:
        for id_a, id_b in to_create:
            link = ExperienceLink(
                source_id=uuid.UUID(id_a),
                target_id=uuid.UUID(id_b),
                link_type="related",
                created_by="batch_create_experience_links",
            )
            session.add(link)
            created += 1
        # commit happens on exit of get_session context

    logger.info("Created %d experience links.", created)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time batch create experience_links by vector similarity."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Max related experiences per experience (default 3)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.65,
        help="Min cosine similarity to create a link (default 0.65)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only log what would be created, do not write DB",
    )
    args = parser.parse_args()
    n = asyncio.run(
        run(
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            dry_run=args.dry_run,
        )
    )
    if not args.dry_run and n > 0:
        logger.info("Done. Refresh experience detail pages to see 关联经验.")
    sys.exit(0)


if __name__ == "__main__":
    main()
