#!/usr/bin/env python3
"""Entity Backfill CLI — batch extract entities for experiences missing entity links.

Usage:
    # Full backfill (all experiences without entities)
    python scripts/daemon/entity_backfill.py

    # Dry run (show what would be processed)
    python scripts/daemon/entity_backfill.py --dry-run

    # Override model
    python scripts/daemon/entity_backfill.py --model DeepSeek-V3

    # Limit batch size
    python scripts/daemon/entity_backfill.py --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


async def _get_unextracted_experiences(
    db_url: str, limit: int = 0,
) -> list[dict]:
    """Fetch published experiences without entity links."""
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience
    from sqlalchemy import select, text

    experiences = []
    async with get_session(db_url) as session:
        # Find experience IDs that have no entity links
        subq = select(text("1")).select_from(
            text("experience_entities ee")
        ).where(text("ee.experience_id = experiences.id"))
        stmt = (
            select(Experience)
            .where(Experience.exp_status == "published")
            .where(~subq.exists())
            .order_by(Experience.updated_at.desc())
        )
        if limit > 0:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        experiences = [exp.to_dict() for exp in rows]

    from team_memory.storage.database import close_db
    await close_db()
    return experiences


async def _build_llm_config(
    model: str | None = None,
    base_url: str | None = None,
) -> "LLMConfig":
    """Build LLMConfig for entity extraction."""
    from team_memory.config import load_settings
    from team_memory.config.llm import LLMConfig

    settings = load_settings()
    ee_cfg = settings.entity_extraction

    resolved_model = model or ee_cfg.model
    resolved_base_url = base_url or ee_cfg.base_url
    api_key = os.environ.get(ee_cfg.api_key_env, "") or ""

    return LLMConfig(
        provider="generic",
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=api_key,
    )


async def run_backfill(
    model: str | None = None,
    base_url: str | None = None,
    limit: int = 0,
    dry_run: bool = False,
    concurrency: int = 3,
) -> None:
    """Run entity backfill for all unextracted experiences."""
    from team_memory.config import load_settings
    from team_memory.services.entity_extractor import EntityExtractor

    settings = load_settings()
    db_url = str(settings.database.url)

    # Fetch unextracted experiences
    experiences = await _get_unextracted_experiences(db_url, limit=limit)
    if not experiences:
        print("No experiences need entity extraction.")
        return

    print(f"Found {len(experiences)} experiences without entity links.")

    if dry_run:
        for i, exp in enumerate(experiences[:20], 1):
            title = exp.get("title", "?")[:60]
            eid = str(exp.get("id", ""))[:8]
            print(f"  {i:3d}. [{eid}] {title}")
        if len(experiences) > 20:
            print(f"  ... and {len(experiences) - 20} more")
        return

    # Build LLM config
    llm_config = await _build_llm_config(model, base_url)
    extractor = EntityExtractor(llm_config=llm_config, db_url=db_url)

    print(f"Using model: {llm_config.model}")
    print(f"Base URL: {llm_config.base_url}")
    print(f"Concurrency: {concurrency}")
    print()

    # Process with semaphore for concurrency control
    sem = asyncio.Semaphore(concurrency)
    succeeded = 0
    failed = 0
    failed_ids: list[str] = []
    start_time = time.time()

    async def _process_one(exp: dict) -> None:
        nonlocal succeeded, failed
        async with sem:
            exp_id = str(exp.get("id", ""))
            title = exp.get("title", "?")[:50]
            try:
                await extractor.extract_and_persist(
                    experience_id=exp_id,
                    title=exp.get("title", ""),
                    description=exp.get("description", ""),
                    solution=exp.get("solution") or "",
                    tags=exp.get("tags") or [],
                    project=exp.get("project", "default"),
                )
                succeeded += 1
                if succeeded % 10 == 0 or succeeded == len(experiences):
                    elapsed = time.time() - start_time
                    rate = succeeded / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {succeeded}/{len(experiences)} "
                        f"({rate:.1f}/s) — failed: {failed}"
                    )
            except Exception as e:
                failed += 1
                failed_ids.append(exp_id)
                print(f"  [FAIL] {exp_id[:8]} {title}: {e}")
            # Rate limit: small delay between requests
            await asyncio.sleep(0.5)

    # Run all tasks
    tasks = [_process_one(exp) for exp in experiences]
    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print()
    print(f"Backfill complete in {elapsed:.1f}s:")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    if failed_ids:
        print(f"  Failed IDs: {', '.join(fid[:8] for fid in failed_ids[:10])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill entity extraction for experiences",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override extraction model (default: from config)",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Override LLM base URL",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max experiences to process (0=all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without extracting",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Max concurrent LLM calls (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(run_backfill(
        model=args.model,
        base_url=args.base_url,
        limit=args.limit,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
