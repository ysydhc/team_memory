#!/usr/bin/env python3
"""Embedding backfill CLI — generate embeddings for experiences missing them.

Usage:
    # Dry run
    python -m team_memory.daemon.backfill.embedding_backfill --dry-run

    # Execute
    python -m team_memory.daemon.backfill.embedding_backfill

    # With limit
    python -m team_memory.daemon.backfill.embedding_backfill --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


async def run_backfill(
    dry_run: bool = False,
    limit: int = 0,
    chunk_size: int = 50,
    resume_from: str | None = None,
    checkpoint_path: str = "/tmp/tm-embedding-backfill-checkpoint.json",
) -> None:
    """Backfill missing embeddings with chunked, resumable execution."""
    import json as _json
    from pathlib import Path as _Path

    from sqlalchemy import text

    from team_memory.config import load_settings
    from team_memory.embedding.factory import create_embedding_provider
    from team_memory.storage.database import get_session

    # --- Checkpoint helpers ---
    def _load_checkpoint() -> dict | None:
        try:
            with open(checkpoint_path) as f:
                return _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            return None

    def _save_checkpoint(data: dict) -> None:
        _Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            _json.dump(data, f, indent=2)

    # Auto-resume from checkpoint if no explicit resume-from
    if resume_from is None:
        ckpt = _load_checkpoint()
        if ckpt:
            resume_from = ckpt.get("last_id")
            if resume_from:
                print(f"Resuming from checkpoint: {resume_from}")

    print("Loading settings...")
    settings = load_settings()
    db_url = str(settings.database.url)
    print("Creating embedding provider...")
    embedding_provider = create_embedding_provider(settings)
    print(f"Embedding provider created: {embedding_provider.__class__.__name__}")

    # Backfill experiences (chunked, resumable)
    print("\n--- Backfilling Experiences ---")
    _cursor_id = resume_from or ""
    _total_succeeded = 0
    _total_failed = 0
    _chunk_count = 0

    while True:
        if _cursor_id:
            _query = text(
                """SELECT id, title, description, solution, tags
                FROM experiences
                WHERE exp_status = 'published' AND embedding IS NULL
                  AND id > :cursor
                ORDER BY created_at ASC
                LIMIT :limit"""
            )
            _params = {"cursor": _cursor_id, "limit": chunk_size}
        else:
            _query = text(
                """SELECT id, title, description, solution, tags
                FROM experiences
                WHERE exp_status = 'published' AND embedding IS NULL
                ORDER BY created_at ASC
                LIMIT :limit"""
            )
            _params = {"limit": chunk_size}
        async with get_session(db_url) as session:
            result = await session.execute(_query, _params)
            rows = result.fetchall()

        if not rows:
            if _chunk_count == 0:
                print("All published experiences have embeddings.")
            else:
                print(f"Experiences backfill complete: {_total_succeeded} ok, {_total_failed} failed")
            break

        _chunk_count += 1
        chunk_total = limit if limit > 0 and limit < len(rows) else len(rows)

        if dry_run:
            for row in rows[:10]:
                print(f"  {str(row[0])[:8]}: {(row[1] or '')[:60]}")
            break

        print(f"  Chunk {_chunk_count}: processing {chunk_total} records (cursor={_cursor_id[:8] if _cursor_id else 'start'})")

        succeeded = 0
        failed = 0
        start = time.monotonic()
        _last_id = _cursor_id

        for i, row in enumerate(rows):
            exp_id = row[0]
            title = row[1] or ""
            description = row[2] or ""
            solution = row[3] or ""
            tags = row[4] or []

            embed_text = f"{title}\n{description}\n{solution or ''}"
            if tags:
                embed_text += f"\n{' '.join(str(t) for t in tags)}"
            embed_text = embed_text[:4000]

            try:
                embedding = await embedding_provider.encode_single(embed_text)
                if embedding is None:
                    failed += 1
                    continue

                async with get_session(db_url) as session:
                    await session.execute(
                        text("UPDATE experiences SET embedding = :embedding WHERE id = :id"),
                        {"embedding": str(embedding), "id": exp_id},
                    )
                    await session.commit()
                succeeded += 1
            except Exception as e:
                failed += 1
                print(f"  Failed {str(exp_id)[:8]}: {e}")

            _last_id = str(exp_id)

            if (i + 1) % 10 == 0:
                elapsed = time.monotonic() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"    {i + 1}/{chunk_total} ({rate:.1f}/s)")

        _total_succeeded += succeeded
        _total_failed += failed
        _cursor_id = _last_id

        # Save checkpoint after each chunk
        _save_checkpoint({
            "last_id": _cursor_id,
            "type": "experiences",
            "chunk": _chunk_count,
            "succeeded": _total_succeeded,
            "failed": _total_failed,
        })
        elapsed = time.monotonic() - start
        print(f"    Chunk done in {elapsed:.1f}s — ok={succeeded}, fail={failed}, total={_total_succeeded}/{_total_failed}")

        if limit > 0 and _total_succeeded + _total_failed >= limit:
            break

    # Backfill archives (chunked, resumable)
    print("\n--- Backfilling Archives ---")
    _arch_cursor = ""
    _arch_succeeded = 0
    _arch_failed = 0
    _arch_chunks = 0

    while True:
        if _arch_cursor:
            _aquery = text(
                """SELECT id, title, overview, solution_doc
                FROM archives
                WHERE embedding IS NULL AND id > :cursor
                ORDER BY created_at ASC
                LIMIT :limit"""
            )
            _params = {"cursor": _arch_cursor, "limit": chunk_size}
        else:
            _aquery = text(
                """SELECT id, title, overview, solution_doc
                FROM archives
                WHERE embedding IS NULL
                ORDER BY created_at ASC
                LIMIT :limit"""
            )
            _params = {"limit": chunk_size}
        async with get_session(db_url) as session:
            result = await session.execute(_aquery, _params)
            rows = result.fetchall()

        if not rows:
            if _arch_chunks == 0:
                print("All archives have embeddings.")
            else:
                print(f"Archives backfill complete ({_arch_succeeded} ok, {_arch_failed} failed)")
            break

        _arch_chunks += 1

        if dry_run:
            for row in rows[:5]:
                print(f"  {str(row[0])[:8]}: {(row[1] or '')[:60]}")
            break

        print(f"  Chunk {_arch_chunks}: {len(rows)} archives")
        start = time.monotonic()
        _last_id = _arch_cursor

        for i, row in enumerate(rows):
            arch_id = row[0]
            embed_text = f"{row[1] or ''}\n{row[2] or ''}\n{row[3] or ''}"[:4000]
            try:
                embedding = await embedding_provider.encode_single(embed_text)
                if embedding is None:
                    _arch_failed += 1
                    continue
                async with get_session(db_url) as session:
                    await session.execute(
                        text("UPDATE archives SET embedding = :embedding WHERE id = :id"),
                        {"embedding": str(embedding), "id": arch_id},
                    )
                    await session.commit()
                _arch_succeeded += 1
            except Exception as e:
                _arch_failed += 1
                print(f"  Failed {str(arch_id)[:8]}: {e}")
            _last_id = str(arch_id)

        _arch_cursor = _last_id
        _save_checkpoint({
            "last_id": _arch_cursor, "type": "archives",
            "chunk": _arch_chunks, "succeeded": _arch_succeeded, "failed": _arch_failed,
        })
        elapsed = time.monotonic() - start
        print(f"    Done {elapsed:.1f}s — ok={len(rows)-_arch_failed if _arch_failed==0 else '?'}")

        if limit > 0 and _arch_succeeded >= limit:
            break

    # Backfill personal memories
    print("\n--- Backfilling Personal Memories ---")
    async with get_session(db_url) as session:
        result = await session.execute(
            text(
                """
            SELECT id, content
            FROM personal_memories
            WHERE embedding IS NULL
            ORDER BY created_at ASC
        """
            )
        )
        pm_rows = result.fetchall()

    if not pm_rows:
        print("All personal memories have embeddings.")
    else:
        total = len(pm_rows)
        if limit > 0:
            pm_rows = pm_rows[:limit]

        print(
            f"Found {total} personal memories without embedding{f' (processing {len(pm_rows)})' if limit > 0 else ''}."
        )

        if not dry_run:
            succeeded = 0
            failed = 0
            start = time.monotonic()

            for i, row in enumerate(pm_rows):
                pm_id = row[0]
                content = row[1] or ""

                embed_text = content[:4000]

                try:
                    embedding = await embedding_provider.encode_single(embed_text)
                    if embedding is None:
                        failed += 1
                        continue

                    async with get_session(db_url) as session:
                        await session.execute(
                            text(
                                "UPDATE personal_memories SET embedding = :embedding WHERE id = :id"
                            ),
                            {"embedding": str(embedding), "id": pm_id},
                        )
                        await session.commit()

                    succeeded += 1
                except Exception as e:
                    failed += 1
                    print(f"  Failed {str(pm_id)[:8]}: {e}")

                if (i + 1) % 10 == 0 or i + 1 == len(pm_rows):
                    elapsed = time.monotonic() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {i + 1}/{len(pm_rows)} ({rate:.1f}/s) — failed: {failed}"
                    )

            elapsed = time.monotonic() - start
            print(
                f"Personal memories backfill complete in {elapsed:.1f}s: Succeeded {succeeded}, Failed {failed}"
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Embedding backfill — generate embeddings for missing experiences"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show count only")
    parser.add_argument(
        "--limit", type=int, default=0, help="Max records (0=all)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Process N records per chunk. Saves checkpoint after each chunk."
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from experience/archive ID (from checkpoint file or manual)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="/tmp/tm-embedding-backfill-checkpoint.json",
        help="Checkpoint file path (auto-resume if exists)"
    )
    args = parser.parse_args()

    print("Running backfill...")
    asyncio.run(run_backfill(
        dry_run=args.dry_run,
        limit=args.limit,
        chunk_size=args.chunk_size,
        resume_from=args.resume_from,
        checkpoint_path=args.checkpoint,
    ))
    print("Main finished.")


if __name__ == "__main__":
    main()
