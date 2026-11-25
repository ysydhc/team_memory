"""Golden Eval, ops-digest, and staleness verification subcommands for tm-cli."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def cmd_eval(args: argparse.Namespace) -> None:
    """Run Golden Eval 3-way ablation benchmark."""
    import asyncio

    from team_memory.bootstrap import bootstrap
    from team_memory.services.analysis.golden_eval import (
        DEFAULT_GOLDEN_SET_PATH,
        DEFAULT_HISTORY_PATH,
        format_eval_report,
        run_ablation_eval,
    )

    ctx = bootstrap(enable_background=False)
    reranker = getattr(ctx.search_pipeline, "_reranker", None)
    golden_path = args.golden_set or DEFAULT_GOLDEN_SET_PATH
    history_path = None if args.no_save else (args.history or DEFAULT_HISTORY_PATH)

    async def _run() -> dict:
        try:
            return await run_ablation_eval(
                db_url=ctx.db_url,
                embedding_provider=ctx.embedding,
                settings=ctx.settings,
                reranker_provider=reranker,
                golden_set_path=golden_path,
                history_path=history_path,
                limit=args.limit,
            )
        finally:
            from team_memory.storage.database import close_db

            await close_db()

    record = asyncio.run(_run())
    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(format_eval_report(record))


def cmd_ops_digest(args: argparse.Namespace) -> None:
    """Run exception-driven 5-subsystem health inspection."""
    import asyncio

    from team_memory.bootstrap import _resolve_db_url
    from team_memory.config import load_settings
    from team_memory.services.analysis.observability import (
        format_ops_digest,
        get_observability_overview,
        send_feishu_webhook,
    )

    try:
        from dotenv import load_dotenv

        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass

    settings = load_settings()
    db_url = _resolve_db_url(settings)

    async def _run() -> dict:
        try:
            return await get_observability_overview(db_url, days=args.days)
        finally:
            from team_memory.storage.database import close_db

            await close_db()

    overview = asyncio.run(_run())
    if args.json:
        print(json.dumps(overview, ensure_ascii=False, indent=2))
        return

    text = format_ops_digest(overview, alert_only=args.alert_only, weekly=args.weekly)
    if text:
        print(text)
        webhook_url = args.webhook or os.environ.get("TEAM_MEMORY_FEISHU_WEBHOOK", "")
        if webhook_url:
            asyncio.run(send_feishu_webhook(webhook_url, text))


def cmd_verify_staleness(args: argparse.Namespace) -> None:
    """Verify code-grounded anchors against local repository and flag possibly_stale memories."""
    import asyncio

    from team_memory.bootstrap import _resolve_db_url
    from team_memory.config import load_settings
    from team_memory.project_resolver import infer_project_name, normalize_project_name
    from team_memory.services.maintenance.staleness_verifier import StalenessVerifier

    try:
        from dotenv import load_dotenv

        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass

    settings = load_settings()
    db_url = _resolve_db_url(settings)
    repo_root = str(Path(args.repo_path).expanduser().resolve())
    project = args.project or normalize_project_name(infer_project_name(repo_root)) or "default"

    async def _run() -> dict:
        try:
            verifier = StalenessVerifier(db_url)
            return await verifier.verify_project(
                project=project,
                repo_path=repo_root,
                dry_run=bool(args.dry_run),
            )
        finally:
            from team_memory.storage.database import close_db

            await close_db()

    report = asyncio.run(_run())
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(
        f"[verify-staleness] project={report.get('project')} repo={report.get('repo_path', repo_root)} "
        f"dry_run={report.get('dry_run', False)}"
    )
    print(
        f"  checked={report.get('checked', 0)} "
        f"marked_possibly_stale={report.get('marked_possibly_stale', 0)} "
        f"reverified_fresh={report.get('reverified_fresh', 0)} "
        f"skipped_no_anchors={report.get('skipped_no_anchors', 0)}"
    )
    for item in report.get("details") or []:
        print(
            f"  - {item['id'][:8]} {item['title']} "
            f"({item['old_status']} -> {item['new_status']}: {', '.join(item.get('reasons') or [])})"
        )
