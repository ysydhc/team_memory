#!/usr/bin/env python3
"""TM Search Quality Report — CLI stats viewer.

Usage:
    python -m team_memory.daemon.observe.tm_stats [--days 30] [--granularity day|hour|week]

Outputs a Rich-formatted terminal report with:
- Search funnel (total → hit → used) with WoW comparison
- Quality evaluation (faithfulness + arena)
- Weekly trend chart with sparkline bars
- Daily trend table with relevance
- Top used queries
- Miss queries (filtered)
- Low-use queries
- Project breakdown with use rate
"""

from __future__ import annotations

import argparse
import asyncio
import os

from team_memory.project_resolver import find_project_root, load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# Faithfulness lives in response_buffer (alias rb), which has no is_noise column
# but always links to a search_log (alias sl). Exclude noise via two filters:
#   1. the linked search_log's is_noise flag (rule/llm gate), and
#   2. legacy char_length fallback for rows logged before the flag existed.
# Used in queries that JOIN response_buffer rb LEFT JOIN search_logs sl.
BUFFER_QUALITY_FILTER = "sl.is_noise IS NOT TRUE AND NOT (char_length(rb.query) < 12 AND rb.query !~ '[A-Za-z]{4,}')"


def _bar(value: float, max_value: float, width: int = 20) -> str:
    """Generate a simple bar chart string."""
    if max_value <= 0:
        return "░" * width
    filled = int(width * value / max_value)
    return "█" * filled + "░" * (width - filled)


def _resolve_dsn() -> str:
    """Resolve DB DSN: TM Settings > DATABASE_URL > TEAM_MEMORY_DB_URL > alembic.ini."""
    try:
        from team_memory.config import load_settings

        url = str(load_settings().database.url).strip()
        if url:
            return url.replace("postgresql+asyncpg://", "postgresql://")
    except Exception:
        pass

    env_url = (os.environ.get("DATABASE_URL") or "").strip()
    if env_url:
        return env_url

    tm_legacy = (os.environ.get("TEAM_MEMORY_DB_URL") or "").strip()
    if tm_legacy:
        return tm_legacy.replace("postgresql+asyncpg://", "postgresql://")

    try:
        import configparser

        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(find_project_root(), "alembic.ini"))
        url = cfg.get("alembic", "sqlalchemy.url", fallback="").strip()
        if url:
            return url.replace("postgresql+asyncpg://", "postgresql://")
    except Exception:
        pass

    msg = (
        "Cannot resolve PostgreSQL DSN: set database.url via config YAML "
        "(or TEAM_MEMORY_DATABASE__URL), DATABASE_URL, TEAM_MEMORY_DB_URL, "
        "or sqlalchemy.url in alembic.ini."
    )
    raise RuntimeError(msg)


async def _faithfulness_breakdown(dsn: str, days: int, threshold: float = 0.7, above: bool = True) -> int:
    """Count faithfulness entries above/below threshold (excluding negative error sentinels)."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        op = ">=" if above else "<"
        count = await conn.fetchval(
            f"SELECT count(*) FROM response_buffer rb "
            f"LEFT JOIN search_logs sl ON rb.search_log_id = sl.id "
            f"WHERE rb.evaluated = TRUE AND rb.faithfulness_score >= 0 AND rb.faithfulness_score {op} $1 "
            f"AND {BUFFER_QUALITY_FILTER} "
            f"AND rb.created_at > NOW() - make_interval(days => $2)",
            threshold,
            days,
        )
        return count or 0
    finally:
        await conn.close()


async def _faithfulness_avg(dsn: str, days: int) -> float:
    """Average faithfulness score for evaluated entries (excluding negative error sentinels)."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        avg = await conn.fetchval(
            f"""
            SELECT avg(rb.faithfulness_score) FROM response_buffer rb
            LEFT JOIN search_logs sl ON rb.search_log_id = sl.id
            WHERE rb.evaluated = TRUE
              AND rb.faithfulness_score >= 0
              AND {BUFFER_QUALITY_FILTER}
              AND rb.created_at > NOW() - make_interval(days => $1)
            """,
            days,
        )
        return float(avg) if avg else 0.0
    finally:
        await conn.close()


async def _embedding_coverage(dsn: str) -> dict:
    """Vector-embedding coverage for searchable experiences.

    A published/promoted experience with a NULL embedding is invisible to
    vector/topic search (only FTS can reach it), so low coverage silently
    degrades recall. This surfaces the gap — e.g. after a vector-dimension
    migration that drops the embedding column and requires a re-backfill.
    """
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE embedding IS NULL) AS missing
            FROM experiences
            WHERE exp_status IN ('published', 'promoted') AND is_deleted = FALSE
            """
        )
        total = row["total"] or 0
        missing = row["missing"] or 0
        covered = total - missing
        ratio = covered / total if total > 0 else 1.0
        return {"total": total, "missing": missing, "covered": covered, "ratio": ratio}
    finally:
        await conn.close()


async def _bitemporal_health(dsn: str) -> dict:
    """Bi-temporal memory health breakdown for published/promoted experiences."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT count(*) FILTER (WHERE valid_until IS NULL) AS active_count,
                   count(*) FILTER (WHERE valid_until IS NOT NULL) AS superseded_count,
                   count(*) FILTER (WHERE last_verified_at IS NOT NULL) AS verified_count
            FROM experiences
            WHERE exp_status IN ('published', 'promoted') AND is_deleted = FALSE
            """
        )
        return {
            "active_count": (row["active_count"] or 0) if row else 0,
            "superseded_count": (row["superseded_count"] or 0) if row else 0,
            "verified_count": (row["verified_count"] or 0) if row else 0,
        }
    finally:
        await conn.close()


def format_telemetry_panel_data(telemetry: dict, bitemporal: dict | None = None) -> dict:
    """Normalize pipeline telemetry and bi-temporal health into display-ready numbers."""
    return {
        "avg_returned_tokens": round(float(telemetry.get("avg_returned_tokens") or 0.0), 1),
        "avg_total_ms": round(float(telemetry.get("avg_total_ms") or 0.0), 1),
        "graph_assisted_searches": int(telemetry.get("graph_assisted_searches") or 0),
        "active_experiences": int((bitemporal or {}).get("active_count") or 0),
        "superseded_experiences": int((bitemporal or {}).get("superseded_count") or 0),
        "verified_experiences": int((bitemporal or {}).get("verified_count") or 0),
    }


async def _get_low_faithfulness_entries(dsn: str, days: int, limit: int = 5) -> list[dict]:
    """Get entries with low faithfulness scores."""
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            f"SELECT rb.query, rb.faithfulness_score, rb.created_at::text "
            f"FROM response_buffer rb "
            f"LEFT JOIN search_logs sl ON rb.search_log_id = sl.id "
            f"WHERE rb.evaluated = TRUE AND rb.faithfulness_score < 0.7 AND rb.faithfulness_score >= 0 "
            f"AND {BUFFER_QUALITY_FILTER} "
            f"AND rb.created_at > NOW() - make_interval(days => $1) "
            f"ORDER BY rb.faithfulness_score ASC LIMIT $2",
            days,
            limit,
        )
        return [{"query": r["query"], "score": r["faithfulness_score"], "created_at": r["created_at"]} for r in rows]
    finally:
        await conn.close()


def _delta_style(delta: float) -> tuple[str, str]:
    """Return (formatted delta string, rich style) for WoW comparison."""
    if delta > 0:
        return f"↑{delta:.1f}%", "green"
    elif delta < 0:
        return f"↓{abs(delta):.1f}%", "red"
    return "→0%", "dim"


async def run_stats(days: int, granularity: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from team_memory.services.search.search_log_writer import SearchLogWriter

    dsn = _resolve_dsn()
    writer = SearchLogWriter(dsn=dsn)
    console = Console()

    try:
        stats = await writer.get_stats(days=days)
        trends = await writer.get_trends(days=days, granularity=granularity)
        miss_queries = await writer.get_miss_queries(days=days, limit=10)
        low_use_queries = await writer.get_low_use_queries(days=days, limit=10)

        if "error" in stats:
            console.print(f"[red][ERROR] Failed to fetch stats: {stats['error']}[/red]")
            return

        # --- 1. Search Funnel with WoW ---
        total = stats["total_searches"]
        hit = stats["hit"]
        used = stats["used"]
        judged = stats.get("judged", 0)
        unjudged = stats["unjudged"]
        legacy_unjudged = stats.get("legacy_unjudged", 0)
        use_rate = stats["use_rate"]

        funnel_text = Text()
        funnel_text.append(f"  Total searches:  {total}\n", style="bold")
        funnel_text.append(f"  Hit (has results): {hit}", style="green")
        funnel_text.append(f"  ({hit / total * 100:.0f}%)\n" if total else " (0%)\n")
        # Use rate is adoption among JUDGED recalls, so it isn't dragged down by
        # recalls still waiting for evaluation. Show judged + pending explicitly.
        funnel_text.append(f"  Judged:  {judged}", style="cyan")
        funnel_text.append(f"  ({judged / hit * 100:.0f}% of hits)\n" if hit else " (0%)\n")
        funnel_text.append(f"  Used by agent:  {used}", style="cyan")
        funnel_text.append(f"  ({use_rate * 100:.1f}% use rate, of judged)\n", style="bold")
        funnel_text.append(f"  Pending judge:  {unjudged}", style="yellow")
        funnel_text.append("  (has results, awaiting eval)\n", style="dim")
        if legacy_unjudged:
            funnel_text.append(f"  Legacy (pre-hook):  {legacy_unjudged}\n", style="dim yellow")

        # WoW comparison from stats (prev_use_rate delta)
        use_rate_delta = stats.get("use_rate_delta")
        prev_hit = stats.get("prev_hit", 0)
        if use_rate_delta is not None:
            funnel_text.append(f"\n  Use rate: {use_rate * 100:.1f}%  ", style="bold")
            if prev_hit > 0:
                delta_str, delta_style = _delta_style(use_rate_delta * 100)
                funnel_text.append(f"({delta_str} vs prev {days}d)", style=delta_style)
            else:
                funnel_text.append("(no prev period data)", style="dim")
            funnel_text.append("\n")

        console.print(Panel(funnel_text, title=f"Search Funnel ({days}d)", border_style="blue"))

        # --- 1b. Embedding coverage health check ---
        try:
            cov = await _embedding_coverage(dsn)
            if cov["total"] > 0:
                ratio_pct = cov["ratio"] * 100
                # < 95% coverage means a meaningful slice of the library is
                # invisible to vector search — flag it loudly.
                healthy = cov["ratio"] >= 0.95
                health_text = Text()
                health_text.append("  Vector coverage (published): ", style="bold")
                health_text.append(
                    f"{cov['covered']}/{cov['total']} ({ratio_pct:.0f}%)\n",
                    style="green" if healthy else "red",
                )
                if not healthy:
                    health_text.append(
                        f"  ⚠ {cov['missing']} experiences missing embeddings — "
                        "run: python -m team_memory.daemon.backfill.embedding_backfill\n",
                        style="red",
                    )
                console.print(Panel(health_text, title="Embedding Health", border_style="yellow"))
        except Exception:
            pass

        # --- 1c. Pipeline Telemetry & Bi-Temporal Health ---
        try:
            telemetry = await writer.get_telemetry_summary(days=days)
            bitemp = await _bitemporal_health(dsn)
            panel_data = format_telemetry_panel_data(telemetry, bitemp)
            telem_text = Text()
            telem_text.append(
                f"  Avg returned tokens: {panel_data['avg_returned_tokens']}  |  "
                f"Avg pipeline latency: {panel_data['avg_total_ms']} ms\n",
                style="cyan",
            )
            telem_text.append(
                f"  Graph-assisted recalls: {panel_data['graph_assisted_searches']}\n",
                style="green",
            )
            telem_text.append(
                f"  Bi-temporal experiences: {panel_data['active_experiences']} active, "
                f"{panel_data['superseded_experiences']} superseded, "
                f"{panel_data['verified_experiences']} verified\n",
                style="bold",
            )
            console.print(
                Panel(
                    telem_text,
                    title=f"Pipeline Telemetry & Bi-Temporal Health ({days}d)",
                    border_style="cyan",
                )
            )
        except Exception:
            pass

        # --- 2. Quality Evaluation (merged) ---
        quality_text = Text()

        # Legacy faithfulness section
        try:
            fb_high = await _faithfulness_breakdown(dsn, days, threshold=0.7, above=True)
            fb_low = await _faithfulness_breakdown(dsn, days, threshold=0.7, above=False)
            fb_evaluated = fb_high + fb_low
            avg_score = await _faithfulness_avg(dsn, days) if fb_evaluated > 0 else 0.0

            quality_text.append("  [Legacy] Faithfulness", style="bold")
            quality_text.append("  (noise-filtered)\n", style="dim")
            quality_text.append(f"    Evaluated: {fb_evaluated}  Avg: {avg_score:.2f}\n")
            quality_text.append(f"    High (≥0.7): {fb_high}", style="green")
            quality_text.append(f"  Low: {fb_low}\n", style="red" if fb_low > 0 else "dim")

            low_entries = await _get_low_faithfulness_entries(dsn, days, limit=3)
            if low_entries:
                quality_text.append("    Lowest:\n", style="dim")
                for e in low_entries:
                    quality_text.append(f"      • {e['score']:.2f} | {e['query'][:50]}\n", style="dim")
        except Exception:
            quality_text.append("  [Legacy] Faithfulness: unavailable\n", style="dim")

        quality_text.append("\n")

        # Arena section
        try:
            arena_stats = await writer.get_arena_stats(days=days)

            if "error" not in arena_stats:
                avg_rel = arena_stats.get("avg_relevance")
                avg_help = arena_stats.get("avg_helpfulness")
                dist = arena_stats.get("score_distribution", {})

                quality_text.append("  [Arena] Relevance & Helpfulness\n", style="bold")
                if avg_rel is not None or avg_help is not None:
                    quality_text.append(
                        f"    Avg relevance:   {avg_rel:.2f}\n"
                        if avg_rel is not None
                        else "    Avg relevance:   N/A\n",
                        style="cyan",
                    )
                    quality_text.append(
                        f"    Avg helpfulness: {avg_help:.2f}\n"
                        if avg_help is not None
                        else "    Avg helpfulness: N/A\n",
                        style="cyan",
                    )
                    quality_text.append(
                        f"    Distribution: H={dist.get('high', 0)} M={dist.get('medium', 0)} L={dist.get('low', 0)}\n"
                    )
                else:
                    quality_text.append("    No scored entries yet.\n", style="dim")

                quality_text.append("\n")

                # Arena champion
                try:
                    from team_memory.daemon.quality.arena.storage import ArenaStorage

                    arena_storage = ArenaStorage()
                    champion = arena_storage.get_champion()
                    if champion is not None:
                        epoch = arena_storage.get_current_epoch()
                        total_rounds = len(arena_storage.load_rounds())
                        converged = arena_storage.is_converged()
                        short_id = champion.id[:12] if len(champion.id) > 12 else champion.id
                        converged_str = "Yes" if converged else "No"

                        quality_text.append(f"  Champion: {short_id} ({champion.elo:.0f})\n", style="bold")
                        quality_text.append(f"  Epoch: {epoch} | Rounds: {total_rounds} | Conv: {converged_str}\n")
                    else:
                        quality_text.append("  Arena: not initialized\n", style="dim")
                except Exception:
                    quality_text.append("  Arena: not initialized\n", style="dim")
            else:
                quality_text.append("  [Arena] Error: unavailable\n", style="dim")
        except Exception:
            quality_text.append("  [Arena] Relevance & Helpfulness: unavailable\n", style="dim")

        console.print(Panel(quality_text, title=f"Quality Evaluation ({days}d)", border_style="magenta"))

        # --- 3. Weekly Trend Chart ---
        try:
            weekly_data = await writer.get_weekly_comparison(weeks=4)
            if weekly_data:
                weekly_table = Table(title="Weekly Trend (4w)", show_lines=False)
                weekly_table.add_column("Week", style="dim")
                weekly_table.add_column("HitRate", min_width=10)
                weekly_table.add_column("UseRate", min_width=10)
                weekly_table.add_column("HR%", justify="right", style="green")
                weekly_table.add_column("UR%", justify="right", style="cyan")
                weekly_table.add_column("Relevance", justify="right")

                max_hr = max((w.get("hit_rate", 0) for w in weekly_data), default=1)
                max_ur = max((w.get("use_rate", 0) for w in weekly_data), default=1)

                for w in weekly_data:
                    week_label = str(w.get("week_start", ""))[:5]
                    hr = w.get("hit_rate", 0)
                    ur = w.get("use_rate", 0)
                    hr_bar = _bar(hr, max_hr if max_hr > 0 else 1, width=8)
                    ur_bar = _bar(ur, max_ur if max_ur > 0 else 1, width=8)
                    hr_pct = f"{hr * 100:.0f}%"
                    ur_pct = f"{ur * 100:.0f}%"
                    rel = w.get("avg_relevance")
                    rel_str = f"{rel:.2f}" if rel is not None else "—"
                    weekly_table.add_row(week_label, hr_bar, ur_bar, hr_pct, ur_pct, rel_str)

                # Determine trend direction
                if len(weekly_data) >= 2:
                    first_ur = weekly_data[0].get("use_rate", 0)
                    last_ur = weekly_data[-1].get("use_rate", 0)
                    if last_ur > first_ur:
                        trend_label = "↑ improving"
                        trend_style = "green"
                    elif last_ur < first_ur:
                        trend_label = "↓ declining"
                        trend_style = "red"
                    else:
                        trend_label = "→ stable"
                        trend_style = "dim"
                    console.print(weekly_table)
                    console.print(f"  Use rate trend: [{trend_style}]{trend_label}[/{trend_style}]\n")
                else:
                    console.print(weekly_table)
        except Exception:
            pass

        # --- 4. Daily Trend Table (with avg_relevance) ---
        if trends and "error" not in trends[0]:
            max_total = max((t["total"] for t in trends), default=1)
            table = Table(title=f"Use Rate Trend (by {granularity})", show_lines=False)
            table.add_column("Period", style="dim")
            table.add_column("Total", justify="right")
            table.add_column("Hit", justify="right", style="green")
            table.add_column("Used", justify="right", style="cyan")
            table.add_column("Use Rate", justify="right", style="bold")
            table.add_column("Relevance", justify="right")
            table.add_column("Volume", min_width=20)

            for t in trends[-14:]:
                period = t["period"] or ""
                if granularity == "hour":
                    period = period[5:16]
                elif granularity == "week":
                    period = period[0:10]
                else:
                    period = period[0:10]

                judged_cnt = t.get("judged")
                if judged_cnt == 0 and t.get("hit", 0) > 0:
                    rate_str = "—"
                else:
                    rate_str = f"{t['use_rate'] * 100:.0f}%"
                rel = t.get("avg_relevance")
                rel_str = f"{rel:.2f}" if rel is not None else "—"
                bar = _bar(t["total"], max_total)
                table.add_row(period, str(t["total"]), str(t["hit"]), str(t["used"]), rate_str, rel_str, bar)

            console.print(table)

        # --- 5. Top Used Queries ---
        try:
            top_used = await writer.get_top_used_queries(days=days, limit=10)
            if top_used and "error" not in top_used[0]:
                top_table = Table(title="Top Used Queries", show_lines=False)
                top_table.add_column("Query")
                top_table.add_column("Used/Total", justify="right", style="cyan")
                top_table.add_column("Rate", justify="right", style="bold")
                top_table.add_column("Relevance", justify="right")

                for q in top_used:
                    query_str = q["query"][:60]
                    ratio = f"{q['used']}/{q['total']}"
                    rate = f"{q['use_rate'] * 100:.0f}%"
                    rel = q.get("avg_relevance")
                    rel_str = f"{rel:.2f}" if rel is not None else "—"
                    top_table.add_row(query_str, ratio, rate, rel_str)

                console.print(top_table)
        except Exception:
            pass

        # --- 6. Miss Queries (filtered) ---
        if miss_queries and "error" not in miss_queries[0]:
            filtered_misses = [
                q
                for q in miss_queries
                if not q["query"].startswith("@") and not q["query"].startswith("<") and len(q["query"]) >= 4
            ]
            if filtered_misses:
                miss_table = Table(title="Miss Queries (no results)", show_lines=False)
                miss_table.add_column("Query", style="red")
                miss_table.add_column("Count", justify="right")
                miss_table.add_column("Last Seen", style="dim")

                for q in filtered_misses[:10]:
                    last = q.get("last_seen", "")[0:10] if q.get("last_seen") else "-"
                    miss_table.add_row(q["query"][:60], str(q["count"]), last)

                console.print(miss_table)

        # --- 7. Low Use Queries (filtered) ---
        if low_use_queries and "error" not in low_use_queries[0]:
            filtered_low = [
                q
                for q in low_use_queries
                if not q["query"].startswith("@") and not q["query"].startswith("<") and len(q["query"]) >= 4
            ]
            low_table = Table(title="Low Use Queries (hit but never used)", show_lines=False)
            low_table.add_column("Query", style="yellow")
            low_table.add_column("Searches", justify="right")
            low_table.add_column("Used", justify="right")

            for q in filtered_low[:10]:
                low_table.add_row(q["query"][:60], str(q["count"]), str(q["used"]))

            console.print(low_table)

        # --- 8. By Project (with use rate) ---
        by_project = stats.get("by_project", {})
        if by_project:
            proj_table = Table(title="By Project", show_lines=False)
            proj_table.add_column("Project")
            proj_table.add_column("Hit", justify="right")
            proj_table.add_column("Judged", justify="right", style="dim")
            proj_table.add_column("Used", justify="right", style="cyan")
            proj_table.add_column("Use Rate", justify="right", style="bold")

            for proj, data in sorted(by_project.items(), key=lambda x: x[1]["total"], reverse=True):
                proj_total = data["total"]
                proj_used = data["used"]
                proj_judged = data.get("judged", proj_total)
                if proj_judged > 0:
                    proj_rate = f"{proj_used / proj_judged * 100:.0f}%"
                elif proj_total > 0:
                    proj_rate = "—"
                else:
                    proj_rate = "0%"
                proj_table.add_row(proj, str(proj_total), str(proj_judged), str(proj_used), proj_rate)

            console.print(proj_table)

    finally:
        await writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="TM Search Quality Report")
    parser.add_argument("--days", type=int, default=7, help="Lookback period in days (default: 7)")
    parser.add_argument(
        "--granularity", choices=["hour", "day", "week"], default="day", help="Trend granularity (default: day)"
    )
    args = parser.parse_args()

    asyncio.run(run_stats(args.days, args.granularity))


if __name__ == "__main__":
    main()
