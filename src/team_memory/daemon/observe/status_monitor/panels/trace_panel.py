"""Recall pipeline trace panel renderers (ASCII flowchart + Rich Panel)."""

from __future__ import annotations

from typing import Any

from team_memory.daemon.observe.status_monitor.collectors import _clean_query_display
from team_memory.daemon.observe.status_monitor.panels.rich_dashboard import _draw_bar


def render_recall_trace_plain(trace: dict[str, Any]) -> str:
    """Render the complete recall pipeline trace as human-intuitive ASCII flowchart + 6-node narrative."""
    if "error" in trace:
        return f"❌ {trace['error']}"

    log = trace["log"]
    qa = trace["query_analysis"]
    sc = trace["stage_counts"]
    sm = trace["stage_metrics"]
    exps = trace["recalled_experiences"]

    vec_n = int(sc.get("vec_count", 0))
    fts_n = int(sc.get("fts_count", 0))
    top_n = int(sc.get("topic_count", 0))
    grp_n = int(sc.get("graph_count", 0))
    grp_u = int(sc.get("graph_unique_added", 0))
    fused_n = int(sc.get("fused_count", 0))
    final_n = int(sc.get("final_count", len(exps)))

    emb_ms = sm.get("embedding_ms", "-")
    fuse_ms = sm.get("retrieve_fuse_ms", "-")
    boost_ms = sm.get("popularity_boost_ms", "-")
    rerank_ms = sm.get("rerank_ms", "-")
    filt_ms = sm.get("adaptive_filter_ms", "-")
    total_ms = sm.get("total_ms", "-")

    ret_tok = trace.get("returned_tokens", 0)
    cand_ents = qa.get("candidate_entities") or []
    match_ents = qa.get("matched_entities") or []
    g_paths = qa.get("graph_paths") or []
    added_syns = qa.get("added_synonyms") or []

    noise_label = "🚫 拦截(Noise)" if log.get("is_noise") else "🟢 放行(Pass)"
    used_label = "✅ 已采纳(Adopted)" if log.get("was_used") else "⏳ 未标记采纳"
    faith_val = log.get("faithfulness_score")
    faith_str = f"{faith_val:.2f}" if faith_val is not None else "-"

    lines: list[str] = []
    lines.append("═" * 92)
    lines.append(
        f" 🔎 TeamMemory 召回全链路深度追踪 (Recall Pipeline Trace)  "
        f"[#{log.get('index', '?')} | Log ID: {log.get('short_id')}]"
    )
    lines.append(
        f" 🕒 时间: {log.get('created_at')}  |  📍 入口: {log.get('source')}  |  "
        f"📁 项目: {log.get('project')}  |  ⏱ 总耗时: {total_ms} ms"
    )
    lines.append("═" * 92)
    lines.append("")

    # ── Part A: Visual End-to-End ASCII Pipeline Flowchart ──
    lines.append("【一、全链路可视化拓扑图 (End-to-End Recall Pipeline Flowchart)】")
    lines.append("")
    q_short = _clean_query_display(qa.get("raw_query", ""), 46)
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ① 入口与意图路由 (Entry & Intent Gate)                                  │")
    lines.append(f'  │   Query: "{q_short}"')
    lines.append(
        f"  │   Source: {log.get('source'):<10} | Project: {log.get('project'):<12} | "
        f"Intent: {log.get('intent_type'):<8} | {noise_label}"
    )
    lines.append("  └────────────────────────────────────┬────────────────────────────────────┘")
    lines.append("                                       ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append(
        f"  │ ② 查询扩展、实体识别与向量化 (Query Expansion & Embedding) [{str(emb_ms):>4}ms]   │"
    )
    lines.append(
        f"  │   同义词扩展: +{len(added_syns)} 词  |  规则提取实体: {len(cand_ents)} 个  |  "
        f"命中图谱锚点: {len(match_ents)} 个        │"
    )
    lines.append("  └──────┬──────────────────┬───────────────────┬───────────────────┬───────┘")
    lines.append("         │                  │                   │                   │")
    lines.append("         ▼                  ▼                   ▼                   ▼")
    lines.append("  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────┐")
    lines.append("  │ ③-A 向量检索 │   │ ③-B 全文检索 │   │ ③-C 主题聚类 │   │ ③-D 2-Hop 图谱 │")
    lines.append("  │ pgvector HNSW│   │  tsvector/词 │   │ Topic Cluster│   │ Degree-PPR扩散 │")
    lines.append(
        f"  │  命中: {vec_n:>3} 条 │   │  命中: {fts_n:>3} 条 │   "
        f"│  命中: {top_n:>3} 条 │   │ 命中:{grp_n:>2}(独+{grp_u:<2}) │"
    )
    lines.append("  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └────────┬───────┘")
    lines.append("         └──────────────────┴─────────┬────────┴────────────────────┘")
    lines.append("                                      ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ④ 多路 RRF 融合、双时态降权与断崖过滤 (Fusion, Bi-Temporal & Elbow)     │")
    lines.append(
        f"  │   RRF合并池: {fused_n:>2}条 ──► 热度加权({boost_ms}ms) ──► "
        f"Rerank({rerank_ms}ms) ──► 断崖截断: {final_n:>2}条 │"
    )
    lines.append("  └────────────────────────────────────┬────────────────────────────────────┘")
    lines.append("                                       ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ⑤ 渐进式披露组装与 Prompt 注入 (Progressive Disclosure)                 │")
    lines.append(
        f"  │   最终注入经验: {final_n:>2} 条  |  消耗 Token: ~{ret_tok}  |  "
        f"双时态降权(0.3x): {trace.get('superseded_hits', 0)} 条      │"
    )
    lines.append("  └────────────────────────────────────┬────────────────────────────────────┘")
    lines.append("                                       ▼")
    lines.append("  ┌─────────────────────────────────────────────────────────────────────────┐")
    lines.append("  │ ⑥ 事后采纳与忠实度评估闭环 (Post-Recall Adoption & Faithfulness)        │")
    lines.append(
        f"  │   状态: {used_label:<16} | 判定源: {log.get('judgment_source') or '-':<8} | "
        f"忠实度(Faithfulness): {faith_str:<5} │"
    )
    lines.append("  └─────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── Part B: Node-by-Node Detailed Breakdown & Human-Intuitive Explanation ──
    lines.append("【二、逐节点详情拆解与诊断说明 (Node-by-Node Detailed Breakdown)】")
    lines.append("")

    # Node 1
    lines.append("─── 节点 ①：入口与意图路由 (Entry & Intent Gate) ───────────────────────────────────")
    lines.append(f"  • 原始查询 (Query):   {qa.get('raw_query', '').strip()}")
    lines.append(
        f"  • 调用来源 (Source):  {log.get('source')}   |   项目作用域 (Project): {log.get('project')}   |   "
        f"意图分类 (Intent): {log.get('intent_type')}"
    )
    if log.get("is_noise"):
        noise_rsn = log.get("noise_reason") or "chit-chat/noise"
        lines.append(f"  • 噪音门禁 (Gate):    🚫 已拦截 (原因: {noise_rsn})")
    else:
        lines.append(
            "  • 噪音门禁 (Gate):    🟢 正常放行 — 非闲聊/确认类噪音，允许进入语义与图谱检索管线"
        )
    lines.append("")

    # Node 2
    lines.append("─── 节点 ②：查询扩展、实体提取与向量化 (Query Expansion & Entity Extraction) ────────")
    syn_desc = ("+" + ", ".join(added_syns)) if added_syns else "无额外同义词触发 (保持原查询语义)"
    lines.append(f"  • 同义词扩展:         {syn_desc}")
    cand_desc = ", ".join(cand_ents) if cand_ents else "(未提取到驼峰/缩写/技术词典实体)"
    lines.append(f"  • 规则提取候选实体:   {cand_desc}")
    if match_ents:
        ent_badges = [
            f"{m['name']} ({m['entity_type']}, 关联度={m['source_count']})" for m in match_ents
        ]
        lines.append(f"  • 知识图谱命中锚点:   🎯 {', '.join(ent_badges)}")
    else:
        lines.append("  • 知识图谱命中锚点:   ⚪ 未命中图谱已有实体节点 (本次依赖向量+全文双路检索)")
    lines.append(f"  • 向量编码耗时:       {emb_ms} ms")
    lines.append("")

    # Node 3
    max_c = max(vec_n, fts_n, top_n, grp_n, fused_n, 1)
    recon_note = (
        " (注: 历史日志未落盘分路计数，已根据当前库实时还原)"
        if trace.get("counts_reconstructed")
        else ""
    )
    lines.append(f"─── 节点 ③：四路并行召回漏斗对比 (4-Way Parallel Retrieval){recon_note} ────────")
    lines.append(
        f"  • 🔵 向量语义路 (Vector HNSW):   {_draw_bar(vec_n, max_c)}  {vec_n:>3} 条  "
        "(捕捉语义相似与跨表述经验)"
    )
    lines.append(
        f"  • 🟣 全文关键词 (FTS tsvector):  {_draw_bar(fts_n, max_c)}  {fts_n:>3} 条  "
        "(精确命中报错名/函数名/配置项)"
    )
    lines.append(
        f"  • 🟡 主题聚类路 (Topic Cluster): {_draw_bar(top_n, max_c)}  {top_n:>3} 条  "
        "(按高频主题簇补充上下文)"
    )
    lines.append(
        f"  • 🟢 2-Hop 图谱 (Entity PPR):    {_draw_bar(grp_n, max_c)}  {grp_n:>3} 条  "
        f"(图谱独有增补: +{grp_u} 条跨跳关联经验)"
    )
    if g_paths:
        lines.append("  • 🕸  激活的 1~2 跳实体关系路径 (Active Graph Paths):")
        for gp in g_paths[:5]:
            lines.append(f"       [{gp['src']}] ──({gp['rel']}, w={gp['weight']})──► [{gp['tgt']}]")
    lines.append("")

    # Node 4
    lines.append("─── 节点 ④：多路 RRF 融合、双时态降权与断崖截断 (RRF Fusion & Adaptive Filter) ────")
    lines.append(
        f"  • 漏斗收敛轨迹:       四路去重合并池 {fused_n} 条  ──►  "
        f"自适应断崖过滤 (Elbow Detection)  ──►  最终保留 {final_n} 条"
    )
    lines.append(
        f"  • 分阶段耗时拆解:     检索+RRF融合={fuse_ms}ms  |  热度/标题加权={boost_ms}ms  |  "
        f"重排(Rerank)={rerank_ms}ms  |  断崖截断={filt_ms}ms"
    )
    sup_n = trace.get("superseded_hits", 0)
    if sup_n > 0:
        lines.append(
            f"  • 双时态演进保护:     ⚠ 检测到 {sup_n} 条已演进旧经验 (valid_until IS NOT NULL)，"
            "已自动施加 0.3x 降权并打标"
        )
    else:
        lines.append(
            "  • 双时态演进保护:     ✅ 命中的经验均为当前最新有效版本 (Active Bi-Temporal State)"
        )
    lines.append("")

    # Node 5
    tok_note = " (估算)" if trace.get("tokens_estimated") else ""
    lines.append(
        f"─── 节点 ⑤：最终命中经验逐条拆解 "
        f"(Recalled Experiences — 共 {len(exps)} 条, ~{ret_tok}{tok_note} tokens) ───"
    )
    if not exps:
        lines.append("  ⚪ 本次召回未返回任何经验条目 (0 hits)。")
        lines.append(
            "     诊断建议: 检查查询词是否过于宽泛、项目名隔离是否正确，或知识库中尚无相关领域沉淀。"
        )
    else:
        for exp in exps:
            ch_str = " + ".join(exp["channels"])
            ver_badge = " [✓已验证]" if exp.get("verified") else ""
            sup_badge = (
                f" [⚠已演进→{exp.get('superseded_by')}]"
                if exp.get("is_superseded")
                else " [🟢现行有效]"
            )
            lines.append(f"  ┌─ Top #{exp['rank']}  {exp['title']}{ver_badge}{sup_badge}")
            lines.append(
                f"  │  ID: {exp['id']}  |  Project: {exp['project']}  |  "
                f"命中通道: {ch_str}  |  历史复用: {exp['use_count']}/{exp['recall_count']} 次"
            )
            if exp.get("tags") or exp.get("entities"):
                tags_s = ", ".join(exp.get("tags") or []) or "-"
                ents_s = ", ".join(exp.get("entities") or []) or "-"
                lines.append(f"  │  标签(Tags): {tags_s}   |   图谱实体(Entities): {ents_s}")
            if exp.get("problem_snippet"):
                lines.append(f"  │  问题摘要: {exp['problem_snippet']}")
            if exp.get("solution_snippet"):
                lines.append(f"  └─ 方案摘要: {exp['solution_snippet']}")
            else:
                lines.append("  └─ (无方案摘要)")
    lines.append("")

    # Node 6
    lines.append("─── 节点 ⑥：事后采纳与忠实度评估闭环 (Post-Recall Adoption & Faithfulness) ──────────")
    judg_src = log.get("judgment_source") or "尚无事后回复触发评估"
    lines.append(f"  • 采纳状态 (was_used): {used_label}   |   判定来源: {judg_src}")
    rel_val = log.get("relevance_score")
    help_val = log.get("helpfulness_score")
    lines.append(
        f"  • 三维质量评分:        忠实度(Faithfulness)={faith_str}  |  "
        f"相关性(Relevance)={rel_val if rel_val is not None else '-'}  |  "
        f"有用性(Helpfulness)={help_val if help_val is not None else '-'}"
    )
    snip = log.get("agent_response_snippet") or ""
    if snip:
        snip_clean = " ".join(snip.split())[:140]
        lines.append(f'  • Agent 回复抽样:      "{snip_clean}…"')
    else:
        lines.append(
            "  • 诊断说明:            当会话停止 (Stop Hook) 或调用 memory_submit_response 时，"
            "系统会自动比对 Agent 最终回复与召回经验，回填采纳率与忠实度。"
        )
    lines.append("═" * 92)

    return "\n".join(lines)


def build_recall_trace_rich(trace: dict[str, Any]) -> Any:
    """Build a Rich renderable for the selected recall's complete pipeline trace."""
    from rich.panel import Panel
    from rich.text import Text

    plain_text = render_recall_trace_plain(trace)
    if "error" in trace:
        return Panel(Text(plain_text, style="bold red"), title="Recall Trace Error", border_style="red")

    log = trace["log"]
    title = (
        f"🔎 召回全链路深度追踪 (Recall Pipeline Trace) — "
        f"[#{log.get('index', '?')} | Log ID: {log.get('short_id')} | Project: {log.get('project')}]"
    )
    return Panel(
        Text(plain_text),
        title=title,
        border_style="bright_cyan",
    )
