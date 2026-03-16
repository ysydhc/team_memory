#!/usr/bin/env python3
"""将「文件位置绑定」相关且为 personal 的经验改为 published，以便在 Web 端列表显示。

用法（在仓库根目录）:
  PYTHONPATH=src python tools/publish_file_location_experiences.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    repo_root = args.repo

    os.chdir(repo_root)
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    from team_memory.bootstrap import bootstrap, get_context
    from team_memory.storage.database import get_session
    from team_memory.storage.models import Experience
    from sqlalchemy import select

    bootstrap(enable_background=False)
    ctx = get_context()
    db_url = os.environ.get("TEAM_MEMORY_DB_URL") or ctx.settings.database.url

    async def run() -> None:
        async with get_session(db_url) as session:
            # 查 publish_status=personal 且 tag 含 file_location_binding 的经验
            stmt = (
                select(Experience)
                .where(Experience.publish_status == "personal")
                .where(Experience.is_deleted == False)
                .where(Experience.tags.overlap(["file_location_binding"]))
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            if not rows:
                print("没有找到 publish_status=personal 且 tag 含 file_location_binding 的经验。")
                return
            print(f"找到 {len(rows)} 条，将改为 published / visibility=project")
            for exp in rows:
                print(f"  - {exp.id} {exp.title[:50]}...")
                if args.dry_run:
                    continue
                exp.publish_status = "published"
                exp.visibility = "project"
            if not args.dry_run and rows:
                await session.commit()
                print("已提交。刷新 Web 端列表即可看到。")

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
