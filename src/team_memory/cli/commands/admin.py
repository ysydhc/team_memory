"""Admin, queue, metrics, status monitor, and AutoDream subcommands for tm-cli."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from team_memory.cli.http_client import _get_queue_db_path


def cmd_queue_list(args: Any) -> None:
    """List pending retry operations / 列出待重试操作。"""
    from team_memory.daemon.buffers.pending_queue import PendingQueue

    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    ops = queue.list_ops(op_type=getattr(args, "op_type", None))
    if not ops:
        print("Queue is empty")
        return

    print(f"{'ID':<14} {'Type':<12} {'Session':<14} {'Retries':<8} {'Status':<10} {'Next Retry'}")
    print("-" * 80)
    for op in ops:
        next_retry = (op.get("next_retry_at") or "")[:19]
        print(
            f"{op['id']:<14} {op['op_type']:<12} {(op.get('session_id') or '')[:12]:<14} "
            f"{op['retries']:<8} {op['status']:<10} {next_retry}"
        )


def cmd_queue_stats(args: Any) -> None:
    """Print queue statistics / 打印队列统计。"""
    from team_memory.daemon.buffers.pending_queue import PendingQueue

    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    s = queue.stats()
    print(f"Queue: {s['pending']} pending, {s['retrying']} retrying, {s['failed']} failed")


def cmd_queue_retry(args: Any) -> None:
    """Force immediate retry of a queued operation / 强制立即重试。"""
    from datetime import datetime, timezone

    from team_memory.daemon.buffers.pending_queue import PendingQueue

    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    now = datetime.now(timezone.utc).isoformat()
    queue._db.execute(
        "UPDATE pending_ops SET next_retry_at = ?, status = 'pending' WHERE id = ?",
        (now, args.op_id),
    )
    queue._db.commit()
    print(f"Reset retry timer for {args.op_id} — will be picked up on next tick")


def cmd_queue_clear(args: Any) -> None:
    """Clear queue entries / 清空队列条目。"""
    from team_memory.daemon.buffers.pending_queue import PendingQueue

    db_path = _get_queue_db_path()
    try:
        queue = PendingQueue(db_path=db_path)
    except Exception as e:
        print(f"Error opening queue: {e}")
        return

    if getattr(args, "op_id", None):
        queue._db.execute("DELETE FROM pending_ops WHERE id = ?", (args.op_id,))
        queue._db.commit()
        print(f"Cleared {args.op_id}")
    elif getattr(args, "clear_all", None):
        count = queue._db.execute("SELECT COUNT(*) FROM pending_ops").fetchone()[0]
        queue._db.execute("DELETE FROM pending_ops")
        queue._db.commit()
        print(f"Cleared {count} entries")
    else:
        print("Use --id <id> or --all")


def cmd_metrics_show(args: Any) -> None:
    """Show current metrics / 显示当前指标."""
    from team_memory.daemon.observe.metrics_collector import MetricsCollector

    mc = MetricsCollector(db_path=_get_queue_db_path())
    rows = mc.query(name=getattr(args, "name", None))
    if not rows:
        print("No metrics recorded yet")
        return
    print(f"{'Name':<30} {'Labels':<30} {'Value':<10}")
    print("-" * 72)
    for r in rows:
        labels = r.get("labels", "{}")
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except Exception:
                pass
        if isinstance(labels, dict):
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        else:
            label_str = str(labels)
        print(f"{r['name']:<30} {label_str[:28]:<30} {r['value']:<10.2f}")


def cmd_metrics_summary(args: Any) -> None:
    """Print aggregated metrics summary / 打印聚合指标摘要."""
    from team_memory.daemon.observe.metrics_collector import MetricsCollector

    mc = MetricsCollector(db_path=_get_queue_db_path())
    s = mc.summary(hours=getattr(args, "hours", 24))
    if not s:
        print("No metrics data in time window")
        return
    for name, labels_dict in sorted(s.items()):
        print(f"{name}:")
        for label_key, val in sorted(labels_dict.items()):
            print(f"  {label_key}: {val:.2f}")


def _parse_since_to_days(since_val: str | None, default_days: int = 7) -> int:
    """Convert --since (e.g. '7d', '24h', '2026-10-25') into days lookback window."""
    if not since_val:
        return default_days
    s = str(since_val).strip().lower()
    if s.endswith("d") and s[:-1].isdigit():
        return max(1, int(s[:-1]))
    if s.endswith("h") and s[:-1].isdigit():
        return max(1, (int(s[:-1]) + 23) // 24)
    if s.isdigit():
        return max(1, int(s))
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta_days = (datetime.now(timezone.utc) - dt).days + 1
        return max(1, delta_days)
    except Exception:
        return default_days


def cmd_status(args: argparse.Namespace) -> None:
    """Show panoramic status across all TeamMemory nodes (one-shot, --trace, or --watch)."""
    import asyncio

    from team_memory.daemon.observe.status_monitor import run_status_monitor

    watch = bool(getattr(args, "watch", False))
    interval = float(getattr(args, "interval", 2.0))
    as_json = bool(getattr(args, "json", False))
    raw_days = int(getattr(args, "days", 7) or 7)
    days = _parse_since_to_days(getattr(args, "since", None), default_days=raw_days)
    server_url = getattr(args, "server_url", None)
    daemon_url = getattr(args, "daemon_url", None)
    db_path = getattr(args, "db_path", None)
    trace_target = getattr(args, "trace", None) or getattr(args, "target", None)
    no_interactive = bool(
        getattr(args, "no_interactive", False)
        or getattr(args, "batch", False)
        or os.environ.get("CI")
        or os.environ.get("TM_BATCH")
        or os.environ.get("NON_INTERACTIVE")
    )
    interactive = False if no_interactive else None
    limit = int(getattr(args, "limit", 12) or 12)

    exit_code = asyncio.run(
        run_status_monitor(
            watch=watch,
            interval=interval,
            as_json=as_json,
            days=days,
            limit=limit,
            server_url=server_url,
            daemon_url=daemon_url,
            db_path=db_path,
            trace_target=trace_target,
            interactive=interactive,
        )
    )
    if exit_code != 0 and not watch:
        sys.exit(exit_code)


def cmd_dream(args: argparse.Namespace) -> None:
    """Trigger AutoDream semantic memory consolidation (supports --dry-run / --force / --project / --model)."""
    import asyncio

    from team_memory.bootstrap import bootstrap
    from team_memory.daemon.integration.auto_dream import AutoDreamWorker
    from team_memory.daemon.integration.tm_sink import create_sink
    from team_memory.daemon.runtime.config import load_config

    dry_run = bool(getattr(args, "dry_run", False))
    force = bool(getattr(args, "force", False)) or dry_run
    project = getattr(args, "project", None) or None
    max_clusters = getattr(args, "max_clusters", None)
    model = getattr(args, "model", None) or None
    as_json = bool(getattr(args, "json", False))

    config = load_config()
    if config.tm.mode == "local":
        bootstrap(enable_background=False)

    sink = create_sink(
        {
            "mode": config.tm.mode,
            "base_url": config.tm.base_url,
            "user": config.tm.user,
        }
    )
    worker = AutoDreamWorker(config=config, sink=sink)
    result = asyncio.run(
        worker.run_once(
            dry_run=dry_run,
            force=force,
            project=project,
            max_clusters=max_clusters,
            model=model,
            trigger="manual",
        )
    )

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result.get("success", True):
            sys.exit(1)
        return

    mode_label = "DRY-RUN (no DB/watermark changes)" if dry_run else "LIVE"
    print(f"=== TeamMemory AutoDream [{mode_label}] ===")
    if result.get("skipped_run"):
        print(f"Status:   SKIPPED ({result.get('skipped_reason', 'unknown')})")
        if result.get("error"):
            print(f"Error:    {result['error']}")
            sys.exit(1)
        return

    tok = result.get("estimated_tokens") or {}
    print(
        f"Status:   {'OK' if result.get('success') else 'FAILED'}  "
        f"(duration={result.get('duration_ms', 0)}ms, model={result.get('model', '?')}, "
        f"watermark_updated={result.get('watermark_updated', False)})"
    )
    if project:
        print(f"Project:  {project}")
    print(
        f"Summary:  candidates={result.get('candidates', 0)} "
        f"(low_adoption={result.get('low_adoption_candidates', 0)}), "
        f"clusters={result.get('clusters', 0)}, "
        f"merged={result.get('merged', 0)}, "
        f"superseded={result.get('superseded', 0)}, "
        f"kept={result.get('kept', 0)}, "
        f"distilled={result.get('distilled', 0)}, "
        f"pruned={result.get('pruned', 0)}, "
        f"errors={result.get('errors', 0)}"
    )
    print(
        f"Tokens:   ~{tok.get('input', 0)} input + ~{tok.get('output', 0)} output "
        f"(total ~{tok.get('input', 0) + tok.get('output', 0)})"
    )
    if result.get("error"):
        print(f"Fatal:    {result['error']}")

    plan = result.get("plan") or []
    if plan:
        print("\n--- Consolidation Plan ---")
        for idx, item in enumerate(plan, start=1):
            action = item.get("action", "?")
            reason = item.get("reason", "")
            titles = item.get("titles") or {}
            print(f"\n[{idx}] {action}  ({len(item.get('cluster_ids', []))} items)")
            print(f"    Reason: {reason}")
            if action == "MERGE":
                pid = item.get("primary_id", "")
                sids = item.get("secondary_ids") or []
                print(f"    Primary:    {pid[:8]} — {titles.get(pid, '')}")
                for sid in sids:
                    print(f"    SoftDelete: {sid[:8]} — {titles.get(sid, '')}")
                prev = item.get("merged_preview") or {}
                if prev.get("title"):
                    print(f"    New Title:  {prev['title']}")
                if prev.get("solution_preview"):
                    snippet = prev["solution_preview"].replace("\n", " ")[:160]
                    print(f"    Preview:    {snippet}")
            elif action == "SUPERSEDE":
                wid = item.get("winner_id", "")
                sups = item.get("superseded_ids") or []
                print(f"    Winner:     {wid[:8]} — {titles.get(wid, '')}")
                for oid in sups:
                    print(f"    Supersede:  {oid[:8]} — {titles.get(oid, '')}")
            else:
                for cid, title in titles.items():
                    print(f"    Item:       {cid[:8]} — {title}")

    distill_plan = result.get("distill_plan") or []
    if distill_plan:
        print("\n--- Playbook Distillation Plan ---")
        for idx, d in enumerate(distill_plan, start=1):
            print(
                f"[{idx}] {d.get('title', '')} "
                f"(project={d.get('project', '?')}, sources={len(d.get('source_ids', []))}, "
                f"~{d.get('estimated_solution_tokens', 0)} tok)"
            )
            if d.get("reason"):
                print(f"    Reason: {d['reason']}")

    if not result.get("success", True):
        sys.exit(1)
