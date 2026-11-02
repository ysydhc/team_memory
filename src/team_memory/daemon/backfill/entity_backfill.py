#!/usr/bin/env python3
"""Entity Backfill CLI — batch extract entities for experiences missing entity links.

Usage:
    # Full backfill (all experiences without entities)
    python -m team_memory.daemon.backfill.entity_backfill

    # Dry run (show what would be processed)
    python -m team_memory.daemon.backfill.entity_backfill --dry-run

    # Override model
    python -m team_memory.daemon.backfill.entity_backfill --model DeepSeek-V4-Flash

    # Limit batch size
    python -m team_memory.daemon.backfill.entity_backfill --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


async def _build_llm_config(
    model: str | None = None,
    base_url: str | None = None,
) -> "LLMConfig":  # noqa: F821
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
    from team_memory.services.entity.entity_extractor import EntityExtractor
    from team_memory.services.maintenance.batch_ops import BatchOpsService

    settings = load_settings()
    db_url = str(settings.database.url)
    svc = BatchOpsService(db_url=db_url)

    # Fetch published experiences without entity links
    experiences = await svc.list_experiences_without_entity_links(limit=limit)
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
                    print(f"  Progress: {succeeded}/{len(experiences)} ({rate:.1f}/s) — failed: {failed}")
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
        "--model",
        type=str,
        default=None,
        help="Override extraction model (default: from config)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override LLM base URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max experiences to process (0=all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without extracting",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent LLM calls (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_backfill(
            model=args.model,
            base_url=args.base_url,
            limit=args.limit,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
