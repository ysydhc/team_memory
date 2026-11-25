"""Rich TUI panoramic dashboard renderable builder."""

from __future__ import annotations

from typing import Any

from team_memory.daemon.observe.status_monitor.collectors import _clean_query_display


def _draw_bar(count: int, max_val: int, width: int = 16, fill: str = "█", empty: str = "░") -> str:
    """Render an intuitive Unicode horizontal bar for stage funnel comparison."""
    if max_val <= 0 or count <= 0:
        return empty * width
    filled = max(1, min(width, int(round((count / max_val) * width))))
    return (fill * filled) + (empty * (width - filled))


def build_rich_renderable(snapshot: dict[str, Any], watch_meta: dict[str, Any] | None = None) -> Any:
    """Build a Rich Group renderable for one-shot or Live TUI display."""
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    ts = snapshot.get("timestamp", "")
    overall = snapshot.get("overall_status", "unknown").upper()
    overall_style = {"HEALTHY": "bold green", "DEGRADED": "bold yellow", "UNHEALTHY": "bold red"}.get(
        overall, "bold white"
    )
    overall_icon = {"HEALTHY": "🟢", "DEGRADED": "🟡", "UNHEALTHY": "🔴"}.get(overall, "⚪")

    header_text = Text()
    header_text.append(f" {overall_icon} TeamMemory Status: {overall} ", style=overall_style)
    header_text.append(f"  |  Updated: {ts}", style="cyan")
    if watch_meta:
        tick = watch_meta.get("tick", 1)
        interval = watch_meta.get("interval", 2.0)
        header_text.append(f"  |  🔄 Live #{tick} ({interval:g}s) — Press Ctrl+C to exit", style="bold magenta")

    anomalies = snapshot.get("anomalies") or []
    if anomalies:
        header_text.append("\n ⚠ Anomalies: " + "  •  ".join(anomalies), style="bold red")

    # Panel 1: Core Server & Infra
    web = snapshot.get("core_server", {})
    pg = snapshot.get("knowledge_base", {})
    checks = web.get("checks") or {}
    db_chk = checks.get("database") or {}
    emb_chk = checks.get("embedding_health") or checks.get("embedding_provider") or {}
    llm_chk = checks.get("llm_provider") or {}
    cache_chk = checks.get("cache") or {}

    web_ok = bool(web.get("reachable")) and web.get("status") in ("healthy", "degraded", "up")
    db_ok = (db_chk.get("status") == "up") or bool(pg.get("reachable"))

    t_infra = Table.grid(padding=(0, 2))
    t_infra.add_column(style="bold")
    t_infra.add_column()
    t_infra.add_column(style="dim")

    t_infra.add_row(
        "Web / MCP (:9111)",
        "[green]● UP[/green]" if web_ok else "[red]● OFFLINE[/red]",
        f"v{web.get('version', '?')} | {web.get('latency_ms', '-')} ms",
    )
    t_infra.add_row(
        "PostgreSQL + pgvector",
        "[green]● UP[/green]" if db_ok else "[red]● DOWN[/red]",
        f"latency: {db_chk.get('latency_ms', 'direct OK' if db_ok else '-')} ms",
    )
    if web_ok:
        emb_up = emb_chk.get("status") == "up"
        llm_up = llm_chk.get("status") == "up"
        t_infra.add_row(
            "Embedding Service",
            "[green]● UP[/green]" if emb_up else "[yellow]● IDLE/WARN[/yellow]",
            f"{emb_chk.get('provider', '?')} ({emb_chk.get('latency_ms', '-')} ms)",
        )
        t_infra.add_row(
            "LLM Provider",
            "[green]● UP[/green]" if llm_up else "[red]● DOWN[/red]",
            f"{llm_chk.get('provider', '?')} {llm_chk.get('model', '')}".strip(),
        )
        t_infra.add_row(
            "Search Cache",
            "[green]● UP[/green]" if cache_chk.get("status") == "up" else "[yellow]● NONE[/yellow]",
            f"backend: {cache_chk.get('backend', 'none')}",
        )
    else:
        t_infra.add_row("Sub-checks", "[dim]Skipped (Web offline)[/dim]", "Run: make web")

    p_infra = Panel(t_infra, title="1. 核心服务与基础设施 (Core & Infra)", border_style="blue")

    # Panel 2: TM Daemon & Hooks
    dm = snapshot.get("daemon", {})
    dm_ok = bool(dm.get("reachable"))
    rw = dm.get("refinement_worker") or {}
    rw_ok = bool(rw.get("healthy", False)) if dm_ok else False
    hb = dm.get("heartbeat") or {}
    loc = snapshot.get("pipeline_local", {})
    cap = loc.get("hooks_capture") or {}
    latest_ev = cap.get("latest_event")

    t_daemon = Table.grid(padding=(0, 2))
    t_daemon.add_column(style="bold")
    t_daemon.add_column()
    t_daemon.add_column(style="dim")

    t_daemon.add_row(
        "TM Daemon (:3901)",
        "[green]● RUNNING[/green]" if dm_ok else "[red]● OFFLINE[/red]",
        f"mode: {dm.get('tm_mode', '-')} | {dm.get('latency_ms', '-')} ms",
    )
    t_daemon.add_row(
        "RefinementWorker",
        "[green]● HEALTHY[/green]" if rw_ok else "[red]● DOWN/STALLED[/red]",
        f"ticks: {rw.get('tick_count', 0)} | fails: {rw.get('fail_count', 0)}",
    )
    t_daemon.add_row(
        "Last Worker Tick",
        str(rw.get("last_tick_ok") or "never")[:19],
        f"wake recoveries: {hb.get('sleep_wake_count', 0)}",
    )
    t_daemon.add_row(
        "IDE Hook Payloads",
        f"[cyan]{cap.get('total_payloads', 0)} captured[/cyan]",
        (
            f"latest: {latest_ev.get('endpoint')} ({latest_ev.get('created_at', '')[:19]})"
            if latest_ev
            else "no hook events yet"
        ),
    )
    maint = dm.get("maintenance") or {}
    ad_running = bool(maint.get("auto_dream_running"))
    jn_running = bool(maint.get("janitor_running"))
    last_j = str(maint.get("last_janitor_at") or "")[:19]
    last_d = str(maint.get("last_dream_at") or "")[:19]
    d_stats = maint.get("last_dream_stats") or {}
    if not last_d or not d_stats:
        try:
            from team_memory.daemon.integration.auto_dream import _read_last_dream, read_last_dream_stats

            if not last_d:
                ld_dt = _read_last_dream()
                last_d = ld_dt.isoformat()[:19] if ld_dt else ""
            if not d_stats:
                d_stats = read_last_dream_stats() or {}
        except Exception:
            pass
    if maint:
        dream_failed = isinstance(d_stats, dict) and d_stats.get("success") is False
        maint_badge = (
            "[green]● ACTIVE[/green]"
            if (ad_running and jn_running and not dream_failed)
            else "[yellow]● DEGRADED[/yellow]"
        )
        maint_desc = f"dream: {'up' if ad_running else 'off'} | janitor: {'up' if jn_running else 'off'}"
        if last_j:
            maint_desc += f" | janitor: {last_j}"
        if last_d:
            maint_desc += f" | dream: {last_d}"
        t_daemon.add_row("AutoDream & Janitor", maint_badge, maint_desc)
        if isinstance(d_stats, dict) and d_stats:
            d_ok = d_stats.get("success", True)
            d_badge = "[green]● OK[/green]" if d_ok else "[red]● FAIL[/red]"
            d_desc = (
                f"cand={d_stats.get('candidates', 0)} clus={d_stats.get('clusters', 0)} "
                f"merge={d_stats.get('merged', 0)} sup={d_stats.get('superseded', 0)} "
                f"distill={d_stats.get('distilled', 0)} err={d_stats.get('errors', 0)}"
            )
            t_daemon.add_row("Last Dream Output", d_badge, d_desc)

    p_daemon = Panel(t_daemon, title="2. 守护进程与 Hook 感知 (Daemon & Hooks)", border_style="magenta")

    # Panel 3: Local Refinement Pipeline Funnel
    by_st = loc.get("sessions_by_status") or {}
    active_n = by_st.get("active", 0) + by_st.get("collecting", 0)
    needs_n = by_st.get("needs_refinement", 0)
    refining_n = by_st.get("refining", 0) + by_st.get("publishing", 0)
    done_n = by_st.get("published", 0) + by_st.get("refined", 0)
    skip_n = by_st.get("skipped", 0) + by_st.get("empty", 0)
    pq = loc.get("pending_queue") or {}

    funnel_summary = Text()
    if not loc.get("db_exists") or loc.get("error"):
        funnel_summary.append(
            f"  ⚠ SQLite Unavailable: {loc.get('error') or loc.get('db_path', '?')}\n",
            style="bold red",
        )
    funnel_summary.append(f"  🟢 Collecting: {active_n} ", style="green")
    funnel_summary.append(f"(Ready≥3 turns: {loc.get('ready_to_converge', 0)})  →  ", style="dim")
    funnel_summary.append(f"🟡 Queued: {needs_n}  →  ", style="yellow")
    funnel_summary.append(f"🔄 Refining: {refining_n}  →  ", style="cyan")
    funnel_summary.append(f"✅ Refined: {done_n} ", style="bold green")
    funnel_summary.append(f"(Skipped: {skip_n})\n", style="dim")
    wal_mb = loc.get("wal_size_mb", 0.0)
    cap_mb = loc.get("capture_db_size_mb", 0.0)
    funnel_summary.append(
        f"  💬 Messages: {loc.get('total_messages', 0)}   "
        f"🔧 Tool Obs: {loc.get('total_tool_observations', 0)}   "
        f"📊 Metrics: {loc.get('total_metrics', 0)}   "
        f"💾 SQLite: {loc.get('db_size_mb', 0.0):.2f} MB (wal {wal_mb:.2f} / cap {cap_mb:.2f})   "
        f"⏳ PendingQueue: pending={pq.get('pending', 0)} retrying={pq.get('retrying', 0)} ",
    )
    failed_n = pq.get("failed", 0)
    funnel_summary.append(
        f"failed={failed_n}",
        style="bold red" if failed_n > 0 else "dim",
    )

    rec_sess = loc.get("recent_sessions") or []
    t_sess = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    t_sess.add_column("Session ID")
    t_sess.add_column("Project")
    t_sess.add_column("Status")
    t_sess.add_column("Turns (U/A)")
    t_sess.add_column("Tools")
    t_sess.add_column("Updated At")
    for s in rec_sess[:4]:
        st_val = s["status"]
        st_color = {
            "active": "green",
            "collecting": "green",
            "needs_refinement": "yellow",
            "refining": "cyan",
            "publishing": "cyan",
            "published": "bold green",
            "refined": "bold green",
        }.get(st_val, "dim")
        t_sess.add_row(
            s["id"][:16],
            s["project"][:18],
            f"[{st_color}]{st_val}[/{st_color}]",
            f"{s['user_msgs']}/{s['asst_msgs']}",
            str(s["tool_obs"]),
            s["updated_at"][:19],
        )

    p_funnel = Panel(
        Group(funnel_summary, t_sess) if rec_sess else funnel_summary,
        title="3. 本地精炼管线漏斗 (Refinement Pipeline Funnel — SQLite)",
        border_style="yellow",
    )

    # Panel 4: Knowledge Base, Bi-Temporal Graph & Search Telemetry
    exp = pg.get("experiences") or {}
    kg = pg.get("knowledge_graph") or {}
    st = pg.get("search_telemetry") or {}
    emb_pct = exp.get("embedding_ratio", 1.0) * 100

    kb_text = Text()
    kb_text.append(
        f"  📚 Experiences:  Active={exp.get('active', 0)}  |  "
        f"Superseded (Bi-Temporal)={exp.get('superseded', 0)}  |  "
        f"Verified={exp.get('verified', 0)}  |  Draft/Review={exp.get('draft_or_review', 0)}\n",
        style="bold",
    )
    kb_text.append(
        f"  🕸  Knowledge Graph:  Entity Nodes={kg.get('entity_nodes', 0)}  |  "
        f"Relations (Edges)={kg.get('entity_relations', 0)}  |  "
        f"Projects={kg.get('projects', 0)}  |  Archives={pg.get('archives_total', 0)}\n",
        style="cyan",
    )
    kb_text.append(
        f"  🔍 Search ({st.get('days', 7)}d):  Total={st.get('total_searches', 0)}  |  "
        f"Hit={st.get('hit_rate', 0.0) * 100:.0f}%  |  "
        f"Adopted={st.get('use_rate', 0.0) * 100:.1f}%  |  "
        f"Faithfulness={st.get('faithfulness_avg', 0.0):.2f}  |  "
        f"Vector Cov={emb_pct:.0f}%\n",
        style="green",
    )
    kb_text.append(
        f"  ⚡ Pipeline Perf:  Avg Latency={st.get('avg_total_ms', 0.0):.1f} ms  |  "
        f"Avg Returned Tokens={st.get('avg_returned_tokens', 0.0):.0f}  |  "
        f"2-Hop Graph Assisted={st.get('graph_assisted_searches', 0)}",
        style="dim",
    )

    rec_exps = pg.get("recent_experiences") or []
    if rec_exps:
        kb_text.append("\n  🆕 Latest Published: ", style="bold")
        kb_text.append(
            "  •  ".join(f"[{e['project']}] {e['title'][:36]}" for e in rec_exps[:3]),
            style="dim",
        )

    p_kb = Panel(
        kb_text,
        title="4. 知识库、双时态图谱与检索遥测 (Knowledge Graph & Search Telemetry)",
        border_style="green",
    )

    # Panel 5: Recent Recalls List (numbered for interactive / CLI trace selection)
    rec_recalls = pg.get("recent_recalls") or []
    t_recalls = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    t_recalls.add_column("#", style="bold yellow", justify="right", no_wrap=True, min_width=4)
    t_recalls.add_column("Time", style="dim", no_wrap=True)
    t_recalls.add_column("Source", no_wrap=True)
    t_recalls.add_column("Project", style="cyan", no_wrap=True)
    t_recalls.add_column("4-Way(V/F/T/G)→Hit", justify="center", no_wrap=True)
    t_recalls.add_column("Perf / State", justify="right", no_wrap=True)
    t_recalls.add_column("Query & Top Hit Experience", ratio=1, min_width=26)

    for r in rec_recalls:
        idx = r.get("index", 0)
        sc = r.get("stage_counts") or {}
        sm = r.get("stage_metrics") or {}
        hit_n = r.get("hit_count", 0)
        if sc:
            v_n = sc.get("vec_count", 0)
            f_n = sc.get("fts_count", 0)
            t_n = sc.get("topic_count", 0)
            g_n = sc.get("graph_count", 0)
            way_str = (
                f"[blue]{v_n}[/blue]/[magenta]{f_n}[/magenta]/"
                f"[yellow]{t_n}[/yellow]/[green]{g_n}[/green]→[bold]{hit_n}[/bold]"
            )
        else:
            way_str = f"[dim]-/-/-/-[/dim]→[bold]{hit_n}[/bold]"

        lat_val = sm.get("total_ms")
        lat_str = f"{float(lat_val):.0f}ms" if lat_val is not None else "-"

        if r.get("is_noise"):
            used_str = "[red]🚫Noise[/red]"
        elif r.get("was_used"):
            used_str = "[bold green]✅Yes[/bold green]"
        else:
            used_str = "[dim]⏳-[/dim]"

        perf_state = f"{lat_str} {used_str}"
        q_disp = r.get("query_display") or _clean_query_display(r.get("query", ""), 42)
        top_title = r.get("top_hit_title") or ""
        q_cell = q_disp
        if top_title:
            q_cell += f"\n[dim]🎯 {top_title[:42]}[/dim]"

        t_recalls.add_row(
            f"[{idx}]",
            r.get("created_at", "")[11:19] or r.get("created_at", "")[:10],
            r.get("source", "mcp")[:10],
            r.get("project", "default")[:14],
            way_str,
            perf_state,
            q_cell,
        )

    hint_text = Text(
        "\n  💡 提示: 运行 `tm-cli status <序号>` (如 `tm-cli status 1`) 查看链路图；"
        "查看展示更多条数可用 `tm-cli status -n 30` 或在下方输入 `m`",
        style="bold yellow",
    )
    p_recalls = Panel(
        Group(t_recalls, hint_text) if rec_recalls else Text("  暂无召回记录 (No search logs found)", style="dim"),
        title=f"5. 最近召回记录列表 (Recent Recalls Top {len(rec_recalls)} — 选中序号查看完整召回链路)",
        border_style="bright_magenta",
    )

    return Group(
        Panel(header_text, border_style="cyan"),
        Columns([p_infra, p_daemon], expand=True),
        p_funnel,
        p_kb,
        p_recalls,
    )
