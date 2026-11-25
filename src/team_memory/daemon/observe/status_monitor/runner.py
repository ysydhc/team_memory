"""CLI / interactive runner for status_monitor (one-shot, trace, and live watch)."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from team_memory.daemon.observe.status_monitor.collectors import collect_status_snapshot
from team_memory.daemon.observe.status_monitor.panels.plain import render_status_plain
from team_memory.daemon.observe.status_monitor.panels.rich_dashboard import build_rich_renderable
from team_memory.daemon.observe.status_monitor.panels.trace_panel import render_recall_trace_plain
from team_memory.daemon.observe.status_monitor.trace_collector import inspect_recall_trace


def _make_console() -> Any:
    """Create a Rich Console with comfortable minimum width even in 80-col PTYs/pipes."""
    import shutil

    from rich.console import Console

    term_w = shutil.get_terminal_size((120, 40)).columns
    width = max(term_w, 118)
    return Console(width=width, _environ={})


def _print_trace_output(trace: dict[str, Any], as_json: bool = False) -> None:
    """Print recall trace as JSON or full-width ASCII flowchart + node breakdown."""
    if as_json:
        print(json.dumps(trace, indent=2, ensure_ascii=False))
        return
    print(render_recall_trace_plain(trace))


async def run_status_monitor(
    *,
    watch: bool = False,
    interval: float = 2.0,
    as_json: bool = False,
    days: int = 7,
    limit: int = 12,
    server_url: str | None = None,
    daemon_url: str | None = None,
    db_path: str | None = None,
    max_ticks: int | None = None,
    trace_target: str | None = None,
    interactive: bool | None = None,
) -> int:
    """Execute one-shot status, recall pipeline trace, or continuous live status monitor."""
    interval = max(0.5, float(interval))
    current_limit = max(1, int(limit))

    if trace_target:
        trace = await inspect_recall_trace(trace_target, limit=max(20, current_limit))
        _print_trace_output(trace, as_json=as_json)
        return 0 if "error" not in trace else 1

    if not watch:
        snapshot = await collect_status_snapshot(
            server_url=server_url,
            daemon_url=daemon_url,
            db_path=db_path,
            days=days,
            limit=current_limit,
        )
        if as_json:
            print(json.dumps(snapshot, indent=2, ensure_ascii=False))
            return 0 if snapshot.get("overall_status") != "unhealthy" else 1

        try:
            _make_console().print(build_rich_renderable(snapshot))
        except ImportError:
            print(render_status_plain(snapshot))

        rec_recalls = (snapshot.get("knowledge_base") or {}).get("recent_recalls") or []
        should_prompt = (
            interactive
            if interactive is not None
            else (sys.stdin.isatty() and sys.stdout.isatty() and bool(rec_recalls))
        )
        if should_prompt and rec_recalls:
            while True:
                max_idx = len(rec_recalls)
                try:
                    prompt_msg = (
                        f"\n👉 输入召回序号 [1-{max_idx}]、Log ID，或输入 m 加载更多"
                        " (直接回车 / q 退出): "
                    )
                    choice = input(prompt_msg).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not choice or choice.lower() in ("q", "quit", "exit"):
                    break
                low_c = choice.lower()
                if low_c in ("m", "more") or low_c.startswith(("m ", "more ", "+")):
                    parts = low_c.replace("+", " ").split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        added_or_target = int(parts[-1])
                        current_limit = (
                            current_limit + added_or_target
                            if choice.startswith("+")
                            else max(current_limit + 10, added_or_target)
                        )
                    else:
                        current_limit += 20
                    snapshot = await collect_status_snapshot(
                        server_url=server_url,
                        daemon_url=daemon_url,
                        db_path=db_path,
                        days=days,
                        limit=current_limit,
                    )
                    rec_recalls = (snapshot.get("knowledge_base") or {}).get("recent_recalls") or []
                    try:
                        _make_console().print(build_rich_renderable(snapshot))
                    except ImportError:
                        print(render_status_plain(snapshot))
                    continue

                trace = await inspect_recall_trace(choice, recent_recalls=rec_recalls, limit=current_limit)
                print()
                _print_trace_output(trace, as_json=False)

        return 0 if snapshot.get("overall_status") != "unhealthy" else 1

    # Watch mode (continuous live dashboard)
    tick = 0
    if as_json:
        try:
            while max_ticks is None or tick < max_ticks:
                tick += 1
                snapshot = await collect_status_snapshot(
                    server_url=server_url,
                    daemon_url=daemon_url,
                    db_path=db_path,
                    days=days,
                    limit=current_limit,
                )
                print(json.dumps(snapshot, ensure_ascii=False), flush=True)
                if max_ticks is not None and tick >= max_ticks:
                    break
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            pass
        return 0

    try:
        from rich.console import Console
        from rich.live import Live

        console = Console()
        first_snapshot = await collect_status_snapshot(
            server_url=server_url,
            daemon_url=daemon_url,
            db_path=db_path,
            days=days,
            limit=current_limit,
        )
        tick = 1
        with Live(
            build_rich_renderable(first_snapshot, {"tick": tick, "interval": interval}),
            console=console,
            refresh_per_second=2,
            screen=False,
        ) as live:
            while max_ticks is None or tick < max_ticks:
                await asyncio.sleep(interval)
                tick += 1
                snapshot = await collect_status_snapshot(
                    server_url=server_url,
                    daemon_url=daemon_url,
                    db_path=db_path,
                    days=days,
                    limit=current_limit,
                )
                live.update(build_rich_renderable(snapshot, {"tick": tick, "interval": interval}))
        return 0
    except ImportError:
        try:
            while max_ticks is None or tick < max_ticks:
                tick += 1
                snapshot = await collect_status_snapshot(
                    server_url=server_url,
                    daemon_url=daemon_url,
                    db_path=db_path,
                    days=days,
                    limit=current_limit,
                )
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(render_status_plain(snapshot, {"tick": tick, "interval": interval}) + "\n")
                sys.stdout.flush()
                if max_ticks is not None and tick >= max_ticks:
                    break
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            pass
        return 0
    except KeyboardInterrupt:
        return 0
