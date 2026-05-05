#!/usr/bin/env python3
"""Solution Backfill — extract solutions from experiences that lack them.

Similar to RefinementWorker but operates on existing published experiences.
For each experience without a solution, calls LLM to extract one from
the description/content, then updates the record.

Usage:
    python scripts/daemon/solution_backfill.py [--limit N] [--dry-run]

Requires TEAM_DATABASE_URL or .env file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from sqlalchemy import text
from team_memory.bootstrap import bootstrap
from team_memory.storage.database import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("solution_backfill")

_SOLUTION_PROMPT = """You are a technical knowledge curator. Given an experience's title and description, extract a concise solution.

Rules:
- If the description already contains a clear solution, extract and condense it
- If the description is a problem/bug report without a solution, write a brief summary of the problem as the solution (so it's not empty)
- If the description is too vague, write "No clear solution documented"
- Keep it under 500 characters
- Return ONLY the solution text, no JSON wrapper, no explanation

Example input:
  Title: Docker 拉取失败时用 SQLite + mock 完成单元测试
  Description: Docker image pull failed in CI. We switched to SQLite for local tests and used MockEmbeddingProvider.

Example output:
  Use SQLite + MockEmbeddingProvider for unit tests when Docker is unavailable.
"""


async def extract_solution(
    title: str,
    description: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int = 30,
) -> str | None:
    """Extract solution from experience using LLM."""
    user_msg = f"Title: {title}\nDescription: {description[:2000]}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SOLUTION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            solution = resp.json()["choices"][0]["message"]["content"].strip()
            return solution if solution else None
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        return None


async def main():
    parser = argparse.ArgumentParser(description="Backfill solutions for experiences")
    parser.add_argument("--limit", type=int, default=50, help="Max experiences to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update DB")
    args = parser.parse_args()

    ctx = bootstrap(enable_background=False)
    db_url = str(ctx.settings.database.url)

    # Get LLM config from entity_extraction settings
    ee_cfg = getattr(ctx.settings, "entity_extraction", None)
    if not ee_cfg:
        logger.error("No entity_extraction config found")
        return

    base_url = ee_cfg.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    model = ee_cfg.model
    api_key = os.environ.get(ee_cfg.api_key_env, "none")
    timeout = ee_cfg.timeout

    logger.info(f"Using LLM: {model} @ {base_url}")

    # Fetch experiences without solution
    async with get_session(db_url) as session:
        r = await session.execute(text("""
            SELECT id, title, description, source
            FROM experiences
            WHERE (solution IS NULL OR solution = '')
            AND exp_status = 'published'
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"limit": args.limit})
        rows = r.fetchall()

    logger.info(f"Found {len(rows)} experiences without solution")

    if not rows:
        logger.info("Nothing to do")
        return

    success = 0
    failed = 0
    skipped = 0

    for i, row in enumerate(rows):
        exp_id = str(row[0])
        title = row[1] or ""
        description = row[2] or ""
        source = row[3] or ""

        if not description.strip():
            logger.info(f"[{i+1}/{len(rows)}] SKIP (empty description): {title[:40]}")
            skipped += 1
            continue

        logger.info(f"[{i+1}/{len(rows)}] Processing: {title[:50]}")

        solution = await extract_solution(
            title=title,
            description=description,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )

        if not solution:
            logger.warning(f"  No solution extracted")
            failed += 1
            continue

        logger.info(f"  Solution: {solution[:80]}...")

        if args.dry_run:
            logger.info(f"  [DRY RUN] Would update experience {exp_id[:8]}")
        else:
            async with get_session(db_url) as session:
                await session.execute(text("""
                    UPDATE experiences
                    SET solution = :solution, updated_at = NOW()
                    WHERE id = :id
                """), {"solution": solution, "id": exp_id})
                await session.commit()
            logger.info(f"  Updated experience {exp_id[:8]}")

        success += 1
        # Rate limit
        await asyncio.sleep(0.5)

    logger.info(f"\nDone: {success} updated, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    asyncio.run(main())
