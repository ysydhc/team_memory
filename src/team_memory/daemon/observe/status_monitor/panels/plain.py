"""Plain-text panoramic status dashboard renderer."""

from __future__ import annotations

from typing import Any

from team_memory.daemon.observe.status_monitor.collectors import _clean_query_display


def _badge_plain(ok: bool, warn: bool = False, label_ok: str = "UP", label_down: str = "DOWN") -> str:
    if ok and not warn:
        return f"🟢 {label_ok}"
    if warn:
        return f"🟡 {label_ok}"
    return f"🔴 {label_down}"


def render_status_plain(snapshot: dict[str, Any], watch_meta: dict[str, Any] | None = None) -> str:
    """Format status snapshot as clean plain/ANSI text (works without Rich)."""
    lines: list[str] = []
    ts = snapshot.get("timestamp", "")
    overall = snapshot.get("overall_status", "unknown").upper()
    overall_icon = {"HEALTHY": "🟢", "DEGRADED": "🟡", "UNHEALTHY": "🔴"}.get(overall, "⚪")

    header = f"=== TeamMemory Panoramic Status  [{overall_icon} {overall}]  ({ts}) ==="
    if watch_meta:
        tick = watch_meta.get("tick", 1)
        interval = watch_meta.get("interval", 2.0)
        header += f"  [Watch #{tick} | every {interval:g}s | Ctrl+C to exit]"
    lines.append(header)

    anomalies = snapshot.get("anomalies") or []
    if anomalies:
        lines.append("  ⚠ Anomalies: " + " ; ".join(anomalies))
    lines.append("")

    # 1. Core Server & Infrastructure
    web = snapshot.get("core_server", {})
    pg = snapshot.get("knowledge_base", {})
    checks = web.get("checks") or {}
    db_chk = checks.get("database") or {}
    emb_chk = checks.get("embedding_health") or checks.get("embedding_provider") or {}
    llm_chk = checks.get("llm_provider") or {}
    cache_chk = checks.get("cache") or {}

    web_ok = bool(web.get("reachable")) and web.get("status") in ("healthy", "degraded", "up")
    db_ok = (db_chk.get("status") == "up") or bool(pg.get("reachable"))
    emb_ok = emb_chk.get("status") == "up"
    llm_ok = llm_chk.get("status") == "up"

    lines.append("┌─ 1. 核心服务与基础设施 (Core Server & Infra) ─────────────────────────────────")
    lines.append(
        f"│ Web / MCP Server ({web.get('url', ':9111')}): "
        f"{_badge_plain(web_ok, warn=(web.get('status') == 'degraded'))} "
        f"(v{web.get('version', '?')}, latency={web.get('latency_ms', '-')}ms)"
    )
    lines.append(
        f"│ PostgreSQL + pgvector: {_badge_plain(db_ok)} "
        f"(latency={db_chk.get('latency_ms', '-')}ms)"
    )
    if web_ok:
        lines.append(
            f"│ Embedding Provider:    {_badge_plain(emb_ok, label_down='IDLE/DOWN')} "
            f"(provider={emb_chk.get('provider', '?')}, latency={emb_chk.get('latency_ms', '-')}ms)"
        )
        lines.append(
            f"│ LLM Provider:          {_badge_plain(llm_ok)} "
            f"(provider={llm_chk.get('provider', '?')}, model={llm_chk.get('model', '-')})"
        )
        lines.append(
            f"│ Search Cache:          {_badge_plain(cache_chk.get('status') == 'up')} "
            f"(backend={cache_chk.get('backend', 'none')})"
        )
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 2. TM Daemon & Hooks
    dm = snapshot.get("daemon", {})
    dm_ok = bool(dm.get("reachable"))
    rw = dm.get("refinement_worker") or {}
    rw_ok = bool(rw.get("healthy", False)) if dm_ok else False
    hb = dm.get("heartbeat") or {}
    loc = snapshot.get("pipeline_local", {})
    cap = loc.get("hooks_capture") or {}
    latest_ev = cap.get("latest_event")

    lines.append("┌─ 2. 守护进程与客户端感知 (TM Daemon & IDE Hooks) ─────────────────────────────")
    lines.append(
        f"│ TM Daemon ({dm.get('url', ':3901')}):      "
        f"{_badge_plain(dm_ok)} (mode={dm.get('tm_mode', '-')}, latency={dm.get('latency_ms', '-')}ms)"
    )
    last_ok_str = str(rw.get("last_tick_ok") or "-")[:19]
    lines.append(
        f"│ RefinementWorker:      {_badge_plain(rw_ok, label_ok='HEALTHY', label_down='DOWN/UNHEALTHY')} "
        f"(ticks={rw.get('tick_count', 0)}, fails={rw.get('fail_count', 0)}, last_ok={last_ok_str})"
    )
    lines.append(
        f"│ Sleep/Wake Heartbeat:  checks={hb.get('check_count', 0)}, "
        f"wake_recoveries={hb.get('sleep_wake_count', 0)}"
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
        m_badge = "ACTIVE" if (ad_running and jn_running and not dream_failed) else "DEGRADED"
        lines.append(
            f"│ AutoDream & Janitor:   [{m_badge}] (dream={'up' if ad_running else 'off'}, "
            f"janitor={'up' if jn_running else 'off'}, last_janitor={last_j or 'none'}, "
            f"last_dream={last_d or 'never'})"
        )
        if isinstance(d_stats, dict) and d_stats:
            lines.append(
                f"│ Last Dream Output:     cand={d_stats.get('candidates', 0)}, "
                f"clusters={d_stats.get('clusters', 0)}, merged={d_stats.get('merged', 0)}, "
                f"superseded={d_stats.get('superseded', 0)}, distilled={d_stats.get('distilled', 0)}, "
                f"pruned={d_stats.get('pruned', 0)}, err={d_stats.get('errors', 0)} "
                f"({d_stats.get('trigger', '?')}, model={d_stats.get('model', '?')})"
            )
    if latest_ev:
        lines.append(
            f"│ IDE Hook Capture:      total={cap.get('total_payloads', 0)} events | "
            f"latest={latest_ev.get('endpoint')} ({latest_ev.get('source')}) @ {latest_ev.get('created_at', '')[:19]}"
        )
    else:
        lines.append(f"│ IDE Hook Capture:      total={cap.get('total_payloads', 0)} events")
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 3. Local Refinement Funnel
    by_st = loc.get("sessions_by_status") or {}
    active_n = by_st.get("active", 0) + by_st.get("collecting", 0)
    needs_n = by_st.get("needs_refinement", 0)
    refining_n = by_st.get("refining", 0) + by_st.get("publishing", 0)
    done_n = by_st.get("published", 0) + by_st.get("refined", 0)
    skip_n = by_st.get("skipped", 0) + by_st.get("empty", 0)
    pq = loc.get("pending_queue") or {}

    lines.append("┌─ 3. 本地精炼管线漏斗 (Refinement Pipeline Funnel — SQLite) ───────────────────")
    if not loc.get("db_exists") or loc.get("error"):
        warn_msg = loc.get("error") or f"DB not found ({loc.get('db_path', '?')})"
        lines.append(f"│ ⚠ SQLite Warning: {warn_msg}")
    lines.append(
        f"│ Session Funnel:  🟢 Collecting={active_n} (Ready≥3 turns: {loc.get('ready_to_converge', 0)}) "
        f"→ 🟡 Queued={needs_n} → 🔄 Refining={refining_n} → ✅ Refined={done_n} (Skipped={skip_n})"
    )
    wal_mb = loc.get("wal_size_mb", 0.0)
    cap_mb = loc.get("capture_db_size_mb", 0.0)
    lines.append(
        f"│ Buffer Stats:    Messages={loc.get('total_messages', 0)} | "
        f"Tool Observations={loc.get('total_tool_observations', 0)} | "
        f"Metrics={loc.get('total_metrics', 0)} | "
        f"SQLite={loc.get('db_size_mb', 0.0):.2f} MB (wal={wal_mb:.2f} MB, capture={cap_mb:.2f} MB)"
    )
    pq_warn = pq.get("failed", 0) > 0 or pq.get("retrying", 0) > 0
    pq_retry_only = pq.get("retrying", 0) > 0 and pq.get("failed", 0) == 0
    pq_badge = _badge_plain(not pq_warn, warn=pq_retry_only, label_ok="CLEAR", label_down="BACKLOG")
    lines.append(
        f"│ PendingQueue:    {pq_badge} "
        f"(pending={pq.get('pending', 0)}, retrying={pq.get('retrying', 0)}, failed={pq.get('failed', 0)})"
    )
    rec_sess = loc.get("recent_sessions") or []
    if rec_sess:
        lines.append("│ Recent Sessions:")
        for s in rec_sess[:3]:
            lines.append(
                f"│   • [{s['project']:<14}] {s['id'][:16]}  status={s['status']:<16} "
                f"msgs={s['user_msgs']}u/{s['asst_msgs']}a  tools={s['tool_obs']}  ({s['updated_at'][:19]})"
            )
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 4. Knowledge Base, Bi-Temporal Graph & Search Telemetry
    exp = pg.get("experiences") or {}
    kg = pg.get("knowledge_graph") or {}
    st = pg.get("search_telemetry") or {}
    emb_pct = exp.get("embedding_ratio", 1.0) * 100

    lines.append("┌─ 4. 知识库、双时态图谱与检索指标 (Knowledge Graph & Search Telemetry) ────────")
    lines.append(
        f"│ Experiences:     Active={exp.get('active', 0)} | "
        f"Superseded (History)={exp.get('superseded', 0)} | "
        f"Verified={exp.get('verified', 0)} | Draft/Review={exp.get('draft_or_review', 0)}"
    )
    lines.append(
        f"│ Vector Coverage: {exp.get('embedding_covered', 0)}/{exp.get('embedding_total', 0)} ({emb_pct:.0f}%) | "
        f"Archives={pg.get('archives_total', 0)} | BG Tasks={pg.get('background_tasks_pending', 0)}"
    )
    lines.append(
        f"│ Knowledge Graph: Entity Nodes={kg.get('entity_nodes', 0)} | "
        f"Relations (Edges)={kg.get('entity_relations', 0)} | Projects={kg.get('projects', 0)}"
    )
    lines.append(
        f"│ Search ({st.get('days', 7)}d):     Searches={st.get('total_searches', 0)} | "
        f"Hit={st.get('hit_rate', 0.0) * 100:.0f}% | Use={st.get('use_rate', 0.0) * 100:.1f}% | "
        f"Faithfulness={st.get('faithfulness_avg', 0.0):.2f}"
    )
    lines.append(
        f"│ Pipeline Perf:   Avg Latency={st.get('avg_total_ms', 0.0):.1f}ms | "
        f"Avg Tokens={st.get('avg_returned_tokens', 0.0):.0f} | "
        f"2-Hop Graph Assisted={st.get('graph_assisted_searches', 0)}"
    )
    rec_exps = pg.get("recent_experiences") or []
    if rec_exps:
        lines.append("│ Latest Published Experiences:")
        for e in rec_exps[:3]:
            lines.append(f"│   • [{e['project']:<14}] {e['id']}  {e['title']} ({e['created_at']})")
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    # 5. Recent Recalls List (numbered for interactive / CLI trace selection)
    rec_recalls = pg.get("recent_recalls") or []
    lines.append("┌─ 5. 最近召回记录列表 (Recent Recalls — 选择序号查看完整召回链路图) ──────────")
    if not rec_recalls:
        lines.append("│ (暂无召回记录 / No search logs found)")
    else:
        lines.append(
            "│ [#]   Time                Source      Project          "
            "4-Way(V/F/T/G) → Hit  Latency   Adopted  Query / Top Hit"
        )
        lines.append(
            "│ ────  ──────────────────  ──────────  ───────────────  "
            "────────────────────  ────────  ───────  ────────────────────────────"
        )
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
                way_str = f"{v_n}/{f_n}/{t_n}/{g_n} → {hit_n}"
            else:
                way_str = f"-/-/-/- → {hit_n}"

            lat_val = sm.get("total_ms")
            lat_str = f"{float(lat_val):.0f}ms" if lat_val is not None else "-"
            used_str = "✅Yes" if r.get("was_used") else ("🚫Noise" if r.get("is_noise") else "⏳-")
            q_disp = r.get("query_display") or _clean_query_display(r.get("query", ""), 42)
            top_title = r.get("top_hit_title") or ""
            top_suffix = f"  🎯 {top_title[:28]}" if top_title else ""
            lines.append(
                f"│ [{idx:<2}]  {r.get('created_at', '')[:19]:<18}  "
                f"{r.get('source', 'mcp')[:10]:<10}  "
                f"{r.get('project', 'default')[:15]:<15}  "
                f"{way_str:<20}  {lat_str:<8}  {used_str:<7}  {q_disp}{top_suffix}"
            )
        lines.append(
            "│ 💡 查看全链路节点图: `tm-cli status <序号>` | 查看更多条数: `tm-cli status -n 30` 或交互输入 `m`"
        )
    lines.append("└───────────────────────────────────────────────────────────────────────────────")

    return "\n".join(lines)
