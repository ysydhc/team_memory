#!/usr/bin/env python3
"""Build a stratified Golden Eval dataset from PostgreSQL search_logs.

Usage:
    python scripts/build_golden_set.py [--target-size 40] [--output evals/golden_set.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from team_memory.config import load_settings  # noqa: E402
from team_memory.project_resolver import load_dotenv_if_exists  # noqa: E402
from team_memory.services.analysis.golden_eval import (  # noqa: E402
    DEFAULT_GOLDEN_SET_PATH,
    build_golden_set_from_db,
)


def main() -> int:
    load_dotenv_if_exists(str(ROOT))
    parser = argparse.ArgumentParser(description="Build Golden Eval dataset from search_logs")
    parser.add_argument("--target-size", type=int, default=40, help="Target query count (default: 40)")
    parser.add_argument("--min-score", type=float, default=0.01, help="Minimum candidate score (default: 0.01)")
    parser.add_argument("--max-per-project", type=int, default=10, help="Max queries per project (default: 10)")
    parser.add_argument("--output", type=Path, default=DEFAULT_GOLDEN_SET_PATH, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Preview summary without writing file")
    args = parser.parse_args()

    settings = load_settings()
    db_url = os.environ.get("TEAM_MEMORY_DB_URL") or settings.database.url

    payload = asyncio.run(
        build_golden_set_from_db(
            db_url,
            target_size=args.target_size,
            min_score=args.min_score,
            max_per_project=args.max_per_project,
            output_path=None if args.dry_run else args.output,
        )
    )

    print(
        f"Golden Set built: {payload['total_queries']} queries across "
        f"{len(payload['project_distribution'])} project(s)"
    )
    print("Project distribution:", json.dumps(payload["project_distribution"], ensure_ascii=False))
    if args.dry_run:
        for q in payload["queries"][:5]:
            print(f"  [{q['id']}] ({q['project']}, freq={q['freq']}) {q['query'][:70]} -> {len(q['expected_ids'])} exp")
    else:
        print(f"Saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
