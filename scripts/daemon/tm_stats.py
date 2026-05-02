#!/usr/bin/env python3
"""TM Search Quality Report — CLI stats viewer.

Usage:
    python scripts/daemon/tm_stats.py [--days 30] [--granularity day|hour|week]

Outputs a Rich-formatted terminal report with:
- Search funnel (total → hit → used)
- Use rate trend (sparkline bars)
- Miss queries (no results)
- Low-use queries (results but never used)
- Project breakdown
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


def _bar(value: float, max_value: float, width: int = 20) -> str:
    """Generate a simple bar chart string."""
    if max_value <= 0:
        return "░" * width
    filled = int(width * value / max_value)
    return "█" * filled + "░" * (width - filled)


def _resolve_dsn() -> str:
    """Resolve DB DSN: TM Settings > DATABASE_URL env > alembic.ini > hardcoded fallback."""
    # 1. Try TM Settings (most reliable — same source as the main app)
    try:
        from team_memory.config import load_settings
        url = str(load_settings().database.url)
        # pydantic-settings returns asyncpg URL; asyncpg.connect needs plain postgresql://
        return url.replace("postgresql+asyncpg://", "postgresql://")
    except Exception:
        pass

    # 2. Fall back to DATABASE_URL env var
    env_url = os.environ.get("DATABASE_URL", "")
    if env_url:
        return env_url

    # 3. Parse alembic.ini
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(_PROJECT_ROOT, "alembic.ini"))
        url = cfg.get("alembic", "sqlalchemy.url", fallback="")
        if url:
            return url.replace("postgresql+asyncpg://", "postgresql://")
    except Exception:
        pass

    return "postgresql://developer:devpass@localhost:5433/team_memory"


async def run_stats(days: int, granularity: str) -> None:
    from daemon.search_log_writer import SearchLogWriter

    dsn = _resolve_dsn()
    writer = SearchLogWriter(dsn=dsn)

    try:
        # Fetch all data
        stats = await writer.get_stats(days=days)
        trends = await writer.get_trends(days=days, granularity=granularity)
        miss_queries = await writer.get_miss_queries(days=days, limit=10)
        low_use_queries = await writer.get_low_use_queries(days=days, limit=10)

        if "error" in stats:
            print(f"[ERROR] Failed to fetch stats: {stats['error']}")
            return

        # Render with Rich
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            console = Console()
        except ImportError:
            # Fallback: plain text output
            _render_plain(stats, trends, miss_queries, low_use_queries, days, granularity)
            return

        # --- Funnel ---
        total = stats["total_searches"]
        hit = stats["hit"]
        used = stats["used"]
        used_marker = stats.get("used_marker", 0)
        used_fuzzy = stats.get("used_fuzzy", 0)
        unjudged = stats["unjudged"]
        use_rate = stats["use_rate"]

        funnel_text = Text()
        funnel_text.append(f"  Total searches:  {total}\n", style="bold")
        funnel_text.append(f"  Hit (has results): {hit}", style="green")
        funnel_text.append(f"  ({hit/total*100:.0f}%)\n" if total else " (0%)\n")
        funnel_text.append(f"  Used by agent:  {used}", style="cyan")
        funnel_text.append(f"  ({use_rate*100:.1f}% use rate)\n")
        funnel_text.append(f"    ├─ marker: {used_marker}\n", style="dim")
        funnel_text.append(f"    └─ fuzzy:  {used_fuzzy}\n", style="dim")
        funnel_text.append(f"  Unjudged:  {unjudged}", style="yellow")
        funnel_text.append(f"  ({unjudged/hit*100:.0f}% of hits)\n" if hit else " (0%)\n")

        console.print(Panel(funnel_text, title=f"TM Search Funnel ({days}d)", border_style="blue"))

        # --- Trend ---
        if trends and "error" not in trends[0]:
            max_total = max((t["total"] for t in trends), default=1)
            table = Table(title=f"Use Rate Trend (by {granularity})", show_lines=False)
            table.add_column("Period", style="dim")
            table.add_column("Total", justify="right")
            table.add_column("Hit", justify="right", style="green")
            table.add_column("Used", justify="right", style="cyan")
            table.add_column("Use Rate", justify="right", style="bold")
            table.add_column("Volume", min_width=20)

            for t in trends[-14:]:  # Show last 14 periods
                period = t["period"] or ""
                # Shorten period display
                if granularity == "hour":
                    period = period[5:16]  # "04-25 14:00"
                elif granularity == "week":
                    period = period[0:10]
                else:
                    period = period[0:10]  # "2026-04-25"

                rate_str = f"{t['use_rate']*100:.0f}%"
                bar = _bar(t["total"], max_total)
                table.add_row(period, str(t["total"]), str(t["hit"]), str(t["used"]), rate_str, bar)

            console.print(table)

        # --- Miss Queries ---
        if miss_queries and "error" not in miss_queries[0]:
            miss_table = Table(title="Miss Queries (no results)", show_lines=False)
            miss_table.add_column("Query", style="red")
            miss_table.add_column("Count", justify="right")
            miss_table.add_column("Last Seen", style="dim")

            for q in miss_queries[:10]:
                last = q.get("last_seen", "")[0:10] if q.get("last_seen") else "-"
                miss_table.add_row(q["query"][:60], str(q["count"]), last)

            console.print(miss_table)

        # --- Low Use Queries ---
        if low_use_queries and "error" not in low_use_queries[0]:
            low_table = Table(title="Low Use Queries (hit but never used)", show_lines=False)
            low_table.add_column("Query", style="yellow")
            low_table.add_column("Searches", justify="right")
            low_table.add_column("Used", justify="right")

            for q in low_use_queries[:10]:
                low_table.add_row(q["query"][:60], str(q["count"]), str(q["used"]))

            console.print(low_table)

        # --- By Project ---
        by_project = stats.get("by_project", {})
        if by_project:
            proj_table = Table(title="By Project", show_lines=False)
            proj_table.add_column("Project")
            proj_table.add_column("Hit", justify="right")
            proj_table.add_column("Used", justify="right", style="cyan")

            for proj, data in sorted(by_project.items(), key=lambda x: x[1]["total"], reverse=True):
                proj_table.add_row(proj, str(data["total"]), str(data["used"]))

            console.print(proj_table)

    finally:
        await writer.close()


def _render_plain(stats, trends, miss_queries, low_use_queries, days, granularity) -> None:
    """Plain text fallback when Rich is not installed."""
    total = stats["total_searches"]
    hit = stats["hit"]
    used = stats["used"]
    used_marker = stats.get("used_marker", 0)
    used_fuzzy = stats.get("used_fuzzy", 0)
    unjudged = stats["unjudged"]
    use_rate = stats["use_rate"]

    print(f"\n=== TM Search Quality Report ({days}d) ===\n")
    print(f"  Total: {total}  Hit: {hit}  Used: {used} (marker:{used_marker} fuzzy:{used_fuzzy})  Rate: {use_rate:.1%}")
    print(f"  Unjudged: {unjudged}\n")

    if trends and "error" not in trends[0]:
        print(f"  Trend ({granularity}):")
        for t in trends[-7:]:
            period = (t["period"] or "")[0:10]
            bar = _bar(t["total"], max(x["total"] for x in trends), width=15)
            print(f"    {period}  {bar}  {t['total']:3d} hits  {t['use_rate']:.0%}")

    if miss_queries and "error" not in miss_queries[0]:
        print(f"\n  Miss Queries:")
        for q in miss_queries[:5]:
            print(f"    {q['query'][:50]:50s}  x{q['count']}")

    if low_use_queries and "error" not in low_use_queries[0]:
        print(f"\n  Low Use Queries:")
        for q in low_use_queries[:5]:
            print(f"    {q['query'][:50]:50s}  x{q['count']} searches, 0 used")


def main() -> None:
    parser = argparse.ArgumentParser(description="TM Search Quality Report")
    parser.add_argument("--days", type=int, default=7, help="Lookback period in days (default: 7)")
    parser.add_argument("--granularity", choices=["hour", "day", "week"], default="day",
                        help="Trend granularity (default: day)")
    args = parser.parse_args()

    asyncio.run(run_stats(args.days, args.granularity))


if __name__ == "__main__":
    main()
