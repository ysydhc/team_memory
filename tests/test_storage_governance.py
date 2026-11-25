"""Unit tests for SQLite status monitoring, TTL maintenance, PayloadCapture truncation, and PG buffer/task pruning."""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.client_config import ClientConfig
from team_memory.config.janitor import JanitorConfig
from team_memory.daemon.buffers.payload_capture import PayloadCapture, truncate_payload_values
from team_memory.daemon.buffers.sqlite_maintenance import (
    SqliteMaintenancePolicy,
    run_sqlite_maintenance,
)
from team_memory.daemon.observe.status_monitor import (
    _inspect_local_sqlite,
    _resolve_endpoints_and_paths,
    collect_status_snapshot,
    render_status_plain,
)
from team_memory.services.maintenance.janitor import MemoryJanitor


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _create_full_local_dbs(tmp_path: Path) -> tuple[Path, Path]:
    """Create realistic drafts.db (WAL mode) and drafts_capture.db with mixed-age data."""
    drafts_db = tmp_path / "drafts.db"
    capture_db = tmp_path / "drafts_capture.db"

    conn = sqlite3.connect(str(drafts_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL DEFAULT 'default',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
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
        );
        CREATE TABLE search_correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            search_log_id TEXT NOT NULL,
            query TEXT,
            result_ids TEXT,
            created_at TEXT NOT NULL,
            consumed INTEGER DEFAULT 0
        );
        CREATE TABLE drafts (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            project TEXT NOT NULL,
            conversation_id TEXT,
            content TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            labels TEXT DEFAULT '{}',
            recorded_at TEXT NOT NULL
        );
        CREATE TABLE pending_ops (
            id TEXT PRIMARY KEY,
            op_type TEXT NOT NULL,
            session_id TEXT,
            payload_json TEXT NOT NULL,
            retries INTEGER DEFAULT 0,
            next_retry_at TEXT NOT NULL,
            last_error TEXT,
            checkpoint_json TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        """
    )

    # Sessions: 1 old published (45d), 1 recent published (5d), 1 old active (45d)
    conn.execute(
        "INSERT INTO sessions VALUES ('s-old-pub', 'team_doc', 'published', ?, ?)",
        (_iso_days_ago(50), _iso_days_ago(45)),
    )
    conn.execute(
        "INSERT INTO sessions VALUES ('s-new-pub', 'team_doc', 'published', ?, ?)",
        (_iso_days_ago(5), _iso_days_ago(5)),
    )
    conn.execute(
        "INSERT INTO sessions VALUES ('s-old-active', 'team_doc', 'active', ?, ?)",
        (_iso_days_ago(45), _iso_days_ago(45)),
    )

    # Messages for each session
    for sid, days in [("s-old-pub", 45), ("s-new-pub", 5), ("s-old-active", 45)]:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
            (sid, "x" * 2000, _iso_days_ago(days)),
        )

    # Tool observations: old on published (should delete), old on active (MUST keep!), recent on published (keep)
    conn.execute(
        "INSERT INTO tool_observations (session_id, tool_name, created_at) VALUES ('s-new-pub', 'bash', ?)",
        (_iso_days_ago(20),),
    )
    conn.execute(
        "INSERT INTO tool_observations (session_id, tool_name, created_at) VALUES ('s-old-active', 'bash', ?)",
        (_iso_days_ago(20),),
    )
    conn.execute(
        "INSERT INTO tool_observations (session_id, tool_name, created_at) VALUES ('s-new-pub', 'bash', ?)",
        (_iso_days_ago(2),),
    )

    # Drafts: old published (delete), recent published (keep), old needs_refinement (MUST keep!)
    conn.execute(
        "INSERT INTO drafts VALUES ('d-old-pub', 't', 'p', 'c', 'body', 'published', ?, ?)",
        (_iso_days_ago(40), _iso_days_ago(40)),
    )
    conn.execute(
        "INSERT INTO drafts VALUES ('d-new-pub', 't', 'p', 'c', 'body', 'published', ?, ?)",
        (_iso_days_ago(5), _iso_days_ago(5)),
    )
    conn.execute(
        "INSERT INTO drafts VALUES ('d-old-need', 't', 'p', 'c', 'body', 'needs_refinement', ?, ?)",
        (_iso_days_ago(40), _iso_days_ago(40)),
    )

    # Metrics: 3 old (>7d), 2 recent (<7d)
    for i in range(3):
        conn.execute(
            "INSERT INTO metrics (name, value, recorded_at) VALUES ('m', ?, ?)",
            (float(i), _iso_days_ago(10 + i)),
        )
    for i in range(2):
        conn.execute(
            "INSERT INTO metrics (name, value, recorded_at) VALUES ('m', ?, ?)",
            (float(i), _iso_days_ago(1 + i)),
        )

    # Pending ops:
    # 1) failed for already-published session (recent 1d) -> MUST be reconciled & deleted!
    # 2) failed > 7d for active session -> deleted by TTL
    # 3) failed < 7d for active session -> kept
    # 4) pending for active session -> MUST be kept!
    conn.execute(
        "INSERT INTO pending_ops VALUES ('op-pub-fail', 'refine', 's-new-pub', '{}', 2, ?, 'err', NULL, 'failed', ?)",
        (_iso_days_ago(1), _iso_days_ago(1)),
    )
    conn.execute(
        "INSERT INTO pending_ops VALUES ('op-old-fail', 'refine', 's-old-active', '{}', 2, ?, 'err', NULL, 'failed', ?)",
        (_iso_days_ago(10), _iso_days_ago(10)),
    )
    conn.execute(
        "INSERT INTO pending_ops VALUES ('op-new-fail', 'refine', 's-old-active', '{}', 2, ?, 'err', NULL, 'failed', ?)",
        (_iso_days_ago(2), _iso_days_ago(2)),
    )
    conn.execute(
        "INSERT INTO pending_ops VALUES ('op-pending', 'refine', 's-old-active', '{}', 0, ?, NULL, NULL, 'pending', ?)",
        (_iso_days_ago(10), _iso_days_ago(10)),
    )

    conn.commit()
    conn.close()

    # Capture DB with 1 old row (>7d) and 1 recent oversized row (>16KB)
    cap = sqlite3.connect(str(capture_db))
    cap.execute(
        """
        CREATE TABLE raw_payloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT NOT NULL,
            source TEXT DEFAULT 'unknown',
            payload_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cap.execute(
        "INSERT INTO raw_payloads (endpoint, source, payload_json, created_at) VALUES (?, ?, ?, ?)",
        ("/hooks/after_response", "cursor", json.dumps({"msg": "old"}), _iso_days_ago(10)),
    )
    cap.execute(
        "INSERT INTO raw_payloads (endpoint, source, payload_json, created_at) VALUES (?, ?, ?, ?)",
        (
            "/hooks/after_response",
            "cursor",
            json.dumps({"agent_response": "A" * 30000, "session_id": "s1"}),
            _iso_days_ago(1),
        ),
    )
    cap.commit()
    cap.close()

    return drafts_db, capture_db


class TestStatusMonitorSqliteFixes:
    def test_tilde_path_expanded_in_client_config_and_status_monitor(self, tmp_path: Path, monkeypatch) -> None:
        import team_memory.client_config as cc_mod

        cfg_dir = tmp_path / "tm_cfg"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "config.toml"
        cfg_file.write_text('[conversation]\ndb_path = "~/.cache/tm-pipeline/drafts.db"\n', encoding="utf-8")
        monkeypatch.setattr(cc_mod, "_CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(cc_mod, "_CONFIG_FILE", cfg_file)
        monkeypatch.delenv("TM_CONVERSATION_DB", raising=False)

        cfg = ClientConfig.load()
        assert not cfg.conversation.db_path.startswith("~")
        assert cfg.conversation.db_path.endswith("/.cache/tm-pipeline/drafts.db")

        cfg_src, sources = ClientConfig.load_with_sources()
        assert not cfg_src.conversation.db_path.startswith("~")
        assert sources["conversation.db_path"] == "config.toml"

        _, _, resolved_db, resolved_cap = _resolve_endpoints_and_paths(
            db_path="~/.cache/tm-pipeline/drafts.db"
        )
        assert not resolved_db.startswith("~")
        assert not resolved_cap.startswith("~")

    def test_inspect_local_sqlite_includes_wal_and_capture_sizes(self, tmp_path: Path) -> None:
        drafts_db, capture_db = _create_full_local_dbs(tmp_path)
        # Create a non-trivial WAL file to verify WAL size inclusion
        wal_path = Path(f"{drafts_db}-wal")
        wal_path.write_bytes(b"\x00" * (2 * 1024 * 1024))

        info = _inspect_local_sqlite(str(drafts_db), str(capture_db))
        assert info["db_exists"] is True
        assert info["wal_size_mb"] >= 2.0
        assert info["db_size_mb"] >= info["db_main_size_mb"] + 2.0
        assert info["total_metrics"] == 5
        assert info["total_drafts"] == 3
        assert info["total_messages"] == 3
        assert info["total_tool_observations"] == 3
        assert info["pending_queue"]["failed"] == 3
        assert info["pending_queue"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_missing_sqlite_db_reports_anomaly_not_silent_zero(self, tmp_path: Path) -> None:
        missing_db = str(tmp_path / "nonexistent_drafts.db")
        with (
            patch(
                "team_memory.daemon.observe.status_monitor._probe_web_health",
                new_callable=AsyncMock,
                return_value={"reachable": True, "status": "healthy", "checks": {}},
            ),
            patch(
                "team_memory.daemon.observe.status_monitor._probe_daemon_status",
                new_callable=AsyncMock,
                return_value={"reachable": True, "status": "running", "refinement_worker": {"healthy": True}},
            ),
            patch(
                "team_memory.daemon.observe.status_monitor._inspect_postgres_knowledge",
                new_callable=AsyncMock,
                return_value={"reachable": True, "experiences": {}, "knowledge_graph": {}, "search_telemetry": {}},
            ),
        ):
            snap = await collect_status_snapshot(db_path=missing_db)
            assert any("Local SQLite unavailable" in a for a in snap["anomalies"])
            plain = render_status_plain(snap)
            assert "SQLite Warning" in plain


class TestPayloadCaptureTruncation:
    @pytest.mark.asyncio
    async def test_capture_truncates_large_strings_preserving_schema(self, tmp_path: Path) -> None:
        cap_db = tmp_path / "cap.db"
        cap = PayloadCapture(db_path=str(cap_db), max_rows_per_endpoint=5, max_str_len=200)
        await cap.start()
        try:
            await cap.capture(
                "/hooks/after_response",
                {"agent_response": "Z" * 5000, "count": 42, "nested": {"text": "Y" * 1000}},
                source="cursor",
            )
            summary = await cap.get_field_summary("/hooks/after_response")
            assert summary["count"] == 1
            assert "str" in summary["fields"]["agent_response"]["types"]
            assert "int" in summary["fields"]["count"]["types"]
            assert "dict" in summary["fields"]["nested"]["types"]

            payloads = await cap.get_payloads("/hooks/after_response")
            assert payloads[0]["payload"]["agent_response"].endswith("...[truncated]")
            assert len(payloads[0]["payload"]["agent_response"]) < 250
        finally:
            await cap.stop()

    def test_truncate_payload_values_helper(self) -> None:
        out = truncate_payload_values({"a": "x" * 300, "b": list(range(100))}, max_str_len=50, max_list_items=10)
        assert out["a"].endswith("...[truncated]")
        assert len(out["b"]) == 10


class TestSqliteMaintenance:
    def test_dry_run_and_apply_ttl_cleanup(self, tmp_path: Path) -> None:
        drafts_db, capture_db = _create_full_local_dbs(tmp_path)
        state_file = tmp_path / "maint_state.json"
        policy = SqliteMaintenancePolicy(
            metrics_retention_days=7,
            tool_observations_retention_days=14,
            finished_sessions_retention_days=30,
            failed_ops_retention_days=7,
            capture_retention_days=7,
            capture_max_rows_per_endpoint=100,
            capture_max_str_len=500,
        )

        # Dry run must not mutate rows
        dry = run_sqlite_maintenance(
            drafts_db,
            capture_db_path=capture_db,
            policy=policy,
            dry_run=True,
            state_file=state_file,
        )
        assert dry["drafts_db"]["deleted"]["metrics"] == 3
        assert dry["drafts_db"]["deleted"]["tool_observations"] == 1
        assert dry["drafts_db"]["deleted"]["sessions"] == 1
        assert dry["drafts_db"]["deleted"]["messages"] == 1
        assert dry["drafts_db"]["deleted"]["drafts"] == 1
        assert dry["drafts_db"]["deleted"]["pending_ops"] == 2
        assert dry["capture_db"]["deleted_rows"] == 1
        assert dry["capture_db"]["truncated_rows"] == 1

        # Actual run
        res = run_sqlite_maintenance(
            drafts_db,
            capture_db_path=capture_db,
            policy=policy,
            force_vacuum=True,
            dry_run=False,
            state_file=state_file,
        )
        assert res["drafts_db"]["vacuumed"] is True
        assert res["drafts_db"]["wal_checkpoint"] is not None
        assert res["capture_db"]["vacuumed"] is True
        assert state_file.exists()

        # Verify remaining data integrity
        conn = sqlite3.connect(str(drafts_db))
        assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2  # s-new-pub + s-old-active
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM tool_observations").fetchone()[0] == 2  # active old + pub recent
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 2  # d-new-pub + d-old-need
        ops = {r[0] for r in conn.execute("SELECT id FROM pending_ops").fetchall()}
        assert ops == {"op-new-fail", "op-pending"}
        conn.close()

        cap = sqlite3.connect(str(capture_db))
        rows = cap.execute("SELECT payload_json FROM raw_payloads").fetchall()
        assert len(rows) == 1
        parsed = json.loads(rows[0][0])
        assert parsed["agent_response"].endswith("...[truncated]")
        assert parsed["session_id"] == "s1"
        cap.close()


class TestJanitorPgBufferAndTaskPruning:
    @pytest.mark.asyncio
    async def test_prune_response_buffer_dry_run_and_delete(self) -> None:
        mock_session = AsyncMock()
        dry_res = MagicMock()
        dry_res.scalar.return_value = 42
        del_res = MagicMock()
        del_res.rowcount = 42
        mock_session.execute.side_effect = [dry_res, del_res]

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        janitor = MemoryJanitor("postgresql+asyncpg://test", config=JanitorConfig())
        janitor._session = _fake_session  # type: ignore[method-assign]

        r_dry = await janitor.prune_response_buffer(dry_run=True)
        assert r_dry["dry_run"] is True
        assert r_dry["would_delete_count"] == 42
        assert r_dry["deleted_count"] == 0

        r_del = await janitor.prune_response_buffer(dry_run=False)
        assert r_del["dry_run"] is False
        assert r_del["deleted_count"] == 42
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prune_background_tasks_dry_run_and_delete(self) -> None:
        mock_session = AsyncMock()
        r1 = MagicMock()
        r1.scalar.return_value = 915
        r2 = MagicMock()
        r2.scalar.return_value = 5
        r3 = MagicMock()
        r3.rowcount = 915
        r4 = MagicMock()
        r4.rowcount = 5
        mock_session.execute.side_effect = [r1, r2, r3, r4]

        @asynccontextmanager
        async def _fake_session():
            yield mock_session

        janitor = MemoryJanitor("postgresql+asyncpg://test", config=JanitorConfig())
        janitor._session = _fake_session  # type: ignore[method-assign]

        d = await janitor.prune_background_tasks(dry_run=True)
        assert d["would_delete_count"] == 920
        assert d["deprecated_count"] == 915
        assert d["expired_terminal_count"] == 5
        assert d["deleted_count"] == 0

        a = await janitor.prune_background_tasks(dry_run=False)
        assert a["deleted_count"] == 920
        mock_session.commit.assert_awaited_once()
