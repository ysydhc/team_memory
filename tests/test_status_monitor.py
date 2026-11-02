"""Unit tests for panoramic status monitor and CLI status/watch commands."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from team_memory.daemon.observe.status_monitor import (
    _inspect_local_sqlite,
    build_recall_trace_rich,
    build_rich_renderable,
    collect_status_snapshot,
    render_recall_trace_plain,
    render_status_plain,
    run_status_monitor,
)


def _create_sample_sqlite_dbs(tmp_path: Path) -> tuple[str, str]:
    drafts_db = tmp_path / "drafts.db"
    capture_db = tmp_path / "drafts_capture.db"

    conn = sqlite3.connect(str(drafts_db))
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT 'default',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE tool_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            command_summary TEXT DEFAULT '',
            target_files TEXT DEFAULT '[]',
            exit_code INTEGER,
            error_excerpt TEXT DEFAULT '',
            succeeded INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE pending_ops (
            id TEXT PRIMARY KEY,
            op_type TEXT NOT NULL,
            session_id TEXT,
            payload TEXT,
            retries INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            next_retry_at TEXT
        )
        """
    )

    conn.execute(
        "INSERT INTO sessions (id, project, status, updated_at) VALUES ('sess-active-1', 'team_doc', 'active', '2026-11-02 09:30:00')"
    )
    conn.execute(
        "INSERT INTO sessions (id, project, status, updated_at) VALUES ('sess-refined-2', 'team_doc', 'published', '2026-11-02 09:20:00')"
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES ('sess-active-1', 'user', ?)",
            (f"user msg {i}",),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES ('sess-active-1', 'assistant', ?)",
            (f"asst msg {i}",),
        )
    conn.execute(
        "INSERT INTO tool_observations (session_id, tool_name, command_summary, exit_code) VALUES ('sess-active-1', 'pytest', 'pytest -v', 0)"
    )
    conn.execute(
        "INSERT INTO pending_ops (id, op_type, session_id, status) VALUES ('op-1', 'save_exp', 'sess-refined-2', 'pending')"
    )
    conn.commit()
    conn.close()

    cap_conn = sqlite3.connect(str(capture_db))
    cap_conn.execute(
        """
        CREATE TABLE raw_payloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT,
            source TEXT,
            payload_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cap_conn.execute(
        "INSERT INTO raw_payloads (endpoint, source, payload_json, created_at) VALUES ('/hooks/before_prompt', 'antigravity', '{}', '2026-11-02 09:31:00')"
    )
    cap_conn.commit()
    cap_conn.close()

    return str(drafts_db), str(capture_db)


def test_inspect_local_sqlite(tmp_path: Path) -> None:
    drafts_db, capture_db = _create_sample_sqlite_dbs(tmp_path)
    info = _inspect_local_sqlite(drafts_db, capture_db)

    assert info["db_exists"] is True
    assert info["total_sessions"] == 2
    assert info["sessions_by_status"]["active"] == 1
    assert info["sessions_by_status"]["published"] == 1
    assert info["ready_to_converge"] == 1
    assert info["total_messages"] == 6
    assert info["total_tool_observations"] == 1
    assert info["pending_queue"]["pending"] == 1
    assert info["hooks_capture"]["total_payloads"] == 1
    assert info["hooks_capture"]["latest_event"]["endpoint"] == "/hooks/before_prompt"


@pytest.mark.asyncio
async def test_collect_and_render_status_snapshot(tmp_path: Path) -> None:
    drafts_db, _ = _create_sample_sqlite_dbs(tmp_path)

    fake_web = {
        "reachable": True,
        "url": "http://localhost:9111",
        "http_status": 200,
        "latency_ms": 12.5,
        "status": "healthy",
        "version": "0.2.0",
        "checks": {
            "database": {"status": "up", "latency_ms": 2.1},
            "embedding_health": {"status": "up", "provider": "openai", "latency_ms": 45.0},
            "llm_provider": {"status": "up", "provider": "generic", "model": "DeepSeek-V4"},
            "cache": {"status": "up", "backend": "memory"},
        },
        "heartbeat": {"check_count": 42, "sleep_wake_count": 1},
    }
    fake_daemon = {
        "reachable": True,
        "url": "http://127.0.0.1:3901",
        "http_status": 200,
        "latency_ms": 4.2,
        "status": "running",
        "tm_mode": "local",
        "refinement_worker": {
            "healthy": True,
            "tick_count": 18,
            "fail_count": 0,
            "last_tick_ok": "2026-11-02T09:30:00",
        },
        "heartbeat": {"check_count": 100, "sleep_wake_count": 0},
    }
    fake_pg = {
        "reachable": True,
        "experiences": {
            "active": 128,
            "superseded": 14,
            "verified": 85,
            "draft_or_review": 3,
            "embedding_covered": 128,
            "embedding_total": 128,
            "embedding_ratio": 1.0,
        },
        "knowledge_graph": {
            "entity_nodes": 340,
            "entity_relations": 512,
            "projects": 4,
        },
        "archives_total": 19,
        "background_tasks_pending": 0,
        "search_telemetry": {
            "days": 7,
            "total_searches": 64,
            "hits": 58,
            "hit_rate": 0.906,
            "used": 41,
            "use_rate": 0.707,
            "avg_total_ms": 84.3,
            "avg_returned_tokens": 620.0,
            "graph_assisted_searches": 17,
            "faithfulness_avg": 0.86,
        },
        "recent_experiences": [
            {
                "id": "87fe30ec",
                "title": "TeamMemory 架构升级：双时态知识图谱",
                "project": "team_doc",
                "status": "published",
                "created_at": "2026-11-02 09:20:00",
            }
        ],
        "recent_recalls": [
            {
                "index": 1,
                "id": "9a7d1122-3344-5566-7788-99aabbccddee",
                "short_id": "9a7d1122",
                "query": "如何排查 pgvector 检索与双时态图谱降权",
                "query_display": "如何排查 pgvector 检索与双时态图谱降权",
                "intent_type": "debug",
                "project": "team_doc",
                "source": "mcp+daemon",
                "result_ids": ["87fe30ec-1111-2222-3333-444455556666"],
                "hit_count": 1,
                "top_hit_title": "TeamMemory 架构升级：双时态知识图谱",
                "was_used": True,
                "judgment_source": "llm",
                "faithfulness_score": 0.92,
                "relevance_score": 0.95,
                "helpfulness_score": 0.90,
                "is_noise": False,
                "noise_reason": "",
                "stage_metrics": {
                    "embedding_ms": 18.4,
                    "retrieve_fuse_ms": 42.1,
                    "popularity_boost_ms": 0.5,
                    "rerank_ms": 0.0,
                    "adaptive_filter_ms": 1.2,
                    "total_ms": 62.2,
                },
                "stage_counts": {
                    "vec_count": 10,
                    "fts_count": 4,
                    "topic_count": 1,
                    "graph_count": 3,
                    "graph_unique_added": 2,
                    "fused_count": 12,
                    "final_count": 1,
                },
                "returned_tokens": 380,
                "created_at": "2026-11-02 10:15:00",
            }
        ],
    }

    with (
        patch(
            "team_memory.daemon.observe.status_monitor._probe_web_health",
            new=AsyncMock(return_value=fake_web),
        ),
        patch(
            "team_memory.daemon.observe.status_monitor._probe_daemon_status",
            new=AsyncMock(return_value=fake_daemon),
        ),
        patch(
            "team_memory.daemon.observe.status_monitor._inspect_postgres_knowledge",
            new=AsyncMock(return_value=fake_pg),
        ),
    ):
        snap = await collect_status_snapshot(db_path=drafts_db, days=7)
        assert snap["overall_status"] == "healthy"
        assert snap["anomalies"] == []

        plain = render_status_plain(snap, {"tick": 2, "interval": 2.0})
        assert "TeamMemory Panoramic Status" in plain
        assert "Watch #2" in plain
        assert "Entity Nodes=340" in plain
        assert "RefinementWorker" in plain
        assert "5. 最近召回记录列表" in plain
        assert "10/4/1/3 → 1" in plain

        # Ensure Rich renderable builds cleanly if rich is installed
        try:
            import rich  # noqa: F401

            renderable = build_rich_renderable(snap, {"tick": 2, "interval": 2.0})
            assert renderable is not None
        except ImportError:
            pass

        # Test watch loop with max_ticks=2 in JSON mode
        rc = await run_status_monitor(
            watch=True,
            interval=0.01,
            as_json=True,
            db_path=drafts_db,
            max_ticks=2,
        )
        assert rc == 0


def test_render_recall_trace_flowchart_and_nodes() -> None:
    sample_trace = {
        "log": {
            "index": 1,
            "id": "9a7d1122-3344-5566-7788-99aabbccddee",
            "short_id": "9a7d1122",
            "query": "如何排查 pgvector 检索与双时态图谱降权",
            "intent_type": "debug",
            "project": "team_doc",
            "source": "mcp+daemon",
            "result_ids": ["87fe30ec-1111-2222-3333-444455556666"],
            "hit_count": 1,
            "was_used": True,
            "agent_response_snippet": "根据双时态知识图谱经验，我们对 valid_until 非空的旧经验进行 0.3x 降权。",
            "judgment_source": "llm",
            "faithfulness_score": 0.92,
            "relevance_score": 0.95,
            "helpfulness_score": 0.90,
            "is_noise": False,
            "noise_reason": "",
            "created_at": "2026-11-02 10:15:00",
        },
        "query_analysis": {
            "raw_query": "如何排查 pgvector 检索与双时态图谱降权",
            "expanded_query": "如何排查 pgvector 检索与双时态图谱降权 entity graph",
            "added_synonyms": ["entity", "graph"],
            "candidate_entities": ["pgvector", "双时态图谱"],
            "matched_entities": [
                {"id": "e1", "name": "pgvector", "entity_type": "tool", "project": "team_doc", "source_count": 12}
            ],
            "graph_paths": [
                {"src": "SearchPipeline", "rel": "uses", "tgt": "pgvector", "weight": 1.0}
            ],
        },
        "stage_counts": {
            "vec_count": 10,
            "fts_count": 4,
            "topic_count": 1,
            "graph_count": 3,
            "graph_unique_added": 2,
            "fused_count": 12,
            "final_count": 1,
        },
        "counts_reconstructed": False,
        "stage_metrics": {
            "embedding_ms": 18.4,
            "retrieve_fuse_ms": 42.1,
            "popularity_boost_ms": 0.5,
            "rerank_ms": 0.0,
            "adaptive_filter_ms": 1.2,
            "total_ms": 62.2,
        },
        "superseded_hits": 0,
        "returned_tokens": 380,
        "tokens_estimated": False,
        "recalled_experiences": [
            {
                "rank": 1,
                "id": "87fe30ec-1111-2222-3333-444455556666",
                "short_id": "87fe30ec",
                "title": "TeamMemory 架构升级：双时态知识图谱",
                "project": "team_doc",
                "exp_status": "published",
                "experience_type": "feature",
                "channels": ["🔵Vector", "🟣FTS", "🟢Graph(2-Hop)"],
                "is_superseded": False,
                "superseded_by": "",
                "verified": True,
                "tags": ["bitemporal", "graph", "search"],
                "entities": ["pgvector", "SearchPipeline"],
                "problem_snippet": "如何处理过期经验与跨跳实体召回",
                "solution_snippet": "引入 valid_until 0.3x 降权与 2-hop Degree-PPR 图谱检索",
                "use_count": 5,
                "recall_count": 8,
            }
        ],
    }

    text_out = render_recall_trace_plain(sample_trace)
    assert "【一、全链路可视化拓扑图" in text_out
    assert "【二、逐节点详情拆解与诊断说明" in text_out
    assert "③-A 向量检索" in text_out
    assert "③-D 2-Hop 图谱" in text_out
    assert "[SearchPipeline] ──(uses, w=1.0)──► [pgvector]" in text_out
    assert "Top #1  TeamMemory 架构升级：双时态知识图谱" in text_out
    assert "忠实度(Faithfulness)=0.92" in text_out

    try:
        import rich  # noqa: F401

        panel = build_recall_trace_rich(sample_trace)
        assert panel is not None
    except ImportError:
        pass


def test_lookup_local_wiki_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from team_memory.daemon.observe.status_monitor import _lookup_local_wiki_entries

    wiki_db = tmp_path / "wiki.db"
    conn = sqlite3.connect(str(wiki_db))
    conn.execute(
        """
        CREATE TABLE wiki_entries (
            id TEXT PRIMARY KEY,
            title TEXT,
            category TEXT,
            content TEXT,
            tags TEXT,
            project TEXT,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO wiki_entries (id, title, category, content, tags, project, updated_at)
        VALUES ('36fd44e6-e0d7-4f3b-ae15-1d0681da2371', 'PhonePilot Android UI Dump SOP', 'sop',
                'How to inspect Android view hierarchy via phone_pilot', '["android", "sop"]', 'phone_pilot', '2026-11-02')
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "team_memory.daemon.observe.status_monitor._WIKI_DB_CANDIDATES",
        (str(wiki_db),),
    )

    found = _lookup_local_wiki_entries(["36fd44e6-e0d7-4f3b-ae15-1d0681da2371"])
    assert "36fd44e6-e0d7-4f3b-ae15-1d0681da2371" in found
    entry = found["36fd44e6-e0d7-4f3b-ae15-1d0681da2371"]
    assert entry["title"] == "PhonePilot Android UI Dump SOP"
    assert entry["project"] == "phone_pilot"


def test_cjk_entity_extraction_and_synonym_expansion() -> None:
    from team_memory.services.entity.entity_search import extract_entities_from_query
    from team_memory.services.search.query_expansion import expand_query_synonyms

    custom_lexicon = {"知识图谱", "召回链路", "tm-cli", "tm-daemon", "status_monitor"}
    entities = extract_entities_from_query(
        "我可以把tm的状态监控改成实时召回链路展示吗",
        lexicon=custom_lexicon,
    )
    assert "tm" in entities
    assert "tm-cli" in entities
    assert "召回链路" in entities
    assert "改成" not in entities
    assert "可以" not in entities

    syn_map = {"召回": "recall", "状态": "status", "tm": "tm-cli"}
    expanded = expand_query_synonyms("查看tm的状态与召回结果", tag_synonyms=syn_map)
    assert "recall" in expanded
    assert "status" in expanded
    assert "tm-cli" in expanded

