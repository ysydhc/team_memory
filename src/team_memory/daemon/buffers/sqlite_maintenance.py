"""Automated SQLite TTL pruning, WAL checkpointing, and space reclamation.

Maintains local pipeline SQLite databases (drafts.db and drafts_capture.db):
- Prunes expired metrics (default 7 days)
- Prunes expired tool_observations for non-active sessions (default 14 days)
- Prunes finished (published/refined/skipped/empty) sessions, their messages,
  search_correlations, and finished drafts (default 30 days)
- Reconciles failed pending_ops for already-published sessions & prunes expired failed ops (default 7 days)
- Runs PRAGMA wal_checkpoint(TRUNCATE) after cleanup
- Periodically or when freelist ratio >= threshold runs VACUUM (with safe lock fallback)
- Prunes and truncates oversized payloads in drafts_capture.db
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from team_memory.daemon.buffers.payload_capture import truncate_payload_values

logger = logging.getLogger("team_memory.daemon.buffers.sqlite_maintenance")

DEFAULT_STATE_FILE = Path.home() / ".cache" / "tm-pipeline" / "last_sqlite_maintenance.json"
_TERMINAL_SESSION_STATUSES = ("published", "refined", "skipped", "empty")
_TERMINAL_DRAFT_STATUSES = ("published", "refined", "skipped", "archived")


@dataclass
class SqliteMaintenancePolicy:
    """Retention and compaction policy for local SQLite pipeline databases."""

    enabled: bool = True
    interval_hours: int = 6
    metrics_retention_days: int = 7
    tool_observations_retention_days: int = 14
    finished_sessions_retention_days: int = 30
    failed_ops_retention_days: int = 7
    capture_retention_days: int = 7
    capture_max_rows_per_endpoint: int = 100
    capture_max_str_len: int = 2000
    vacuum_interval_days: int = 7
    vacuum_freelist_ratio: float = 0.25


def _file_sizes_bytes(db_path: Path) -> dict[str, int]:
    """Return byte sizes of main DB, -wal, and -shm files."""
    main_b = db_path.stat().st_size if db_path.exists() else 0
    wal_p = Path(f"{db_path}-wal")
    shm_p = Path(f"{db_path}-shm")
    wal_b = wal_p.stat().st_size if wal_p.exists() else 0
    shm_b = shm_p.stat().st_size if shm_p.exists() else 0
    return {
        "main_bytes": main_b,
        "wal_bytes": wal_b,
        "shm_bytes": shm_b,
        "total_bytes": main_b + wal_b + shm_b,
    }


def _read_state(state_file: Path | None) -> dict[str, Any]:
    if state_file is None or not state_file.exists():
        return {}
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(state_file: Path | None, data: dict[str, Any]) -> None:
    if state_file is None:
        return
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to write sqlite maintenance state: %s", exc)


def _should_vacuum(
    conn: sqlite3.Connection,
    *,
    last_vacuum_iso: str | None,
    policy: SqliteMaintenancePolicy,
    force_vacuum: bool,
    now: datetime,
) -> tuple[bool, float]:
    """Decide whether VACUUM should run based on freelist ratio or elapsed interval."""
    page_count = int((conn.execute("PRAGMA page_count").fetchone() or [0])[0] or 0)
    freelist_count = int((conn.execute("PRAGMA freelist_count").fetchone() or [0])[0] or 0)
    freelist_ratio = (freelist_count / page_count) if page_count > 0 else 0.0

    if force_vacuum:
        return True, freelist_ratio
    if freelist_ratio >= policy.vacuum_freelist_ratio and freelist_count > 0:
        return True, freelist_ratio
    if last_vacuum_iso:
        try:
            last_dt = datetime.fromisoformat(last_vacuum_iso)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if (now - last_dt).total_seconds() >= policy.vacuum_interval_days * 86400:
                return True, freelist_ratio
        except Exception:
            pass
    return False, freelist_ratio


def _maintain_drafts_db(
    db_path: Path,
    policy: SqliteMaintenancePolicy,
    *,
    now: datetime,
    last_vacuum_iso: str | None,
    force_vacuum: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Prune expired rows in drafts.db, checkpoint WAL, and optionally VACUUM."""
    result: dict[str, Any] = {
        "db_path": str(db_path),
        "exists": db_path.exists(),
        "dry_run": dry_run,
        "size_before": _file_sizes_bytes(db_path),
        "deleted": {
            "metrics": 0,
            "tool_observations": 0,
            "messages": 0,
            "search_correlations": 0,
            "sessions": 0,
            "drafts": 0,
            "pending_ops": 0,
        },
        "wal_checkpoint": None,
        "vacuumed": False,
        "freelist_ratio_before_vacuum": 0.0,
    }
    if not db_path.exists():
        result["size_after"] = result["size_before"]
        return result

    metrics_cutoff = (now - timedelta(days=max(1, policy.metrics_retention_days))).isoformat()
    tool_cutoff = (now - timedelta(days=max(1, policy.tool_observations_retention_days))).isoformat()
    finished_cutoff = (now - timedelta(days=max(1, policy.finished_sessions_retention_days))).isoformat()
    failed_ops_cutoff = (now - timedelta(days=max(1, policy.failed_ops_retention_days))).isoformat()

    conn = sqlite3.connect(str(db_path), timeout=15.0)
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        # 1. Prune expired metrics
        if "metrics" in tables:
            if dry_run:
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM metrics WHERE recorded_at < ?",
                    (metrics_cutoff,),
                ).fetchone()[0]
                result["deleted"]["metrics"] = int(cnt or 0)
            else:
                cur = conn.execute("DELETE FROM metrics WHERE recorded_at < ?", (metrics_cutoff,))
                result["deleted"]["metrics"] = int(cur.rowcount or 0)

        # 2. Identify expired terminal sessions
        expired_session_ids: list[str] = []
        if "sessions" in tables:
            placeholders = ",".join("?" for _ in _TERMINAL_SESSION_STATUSES)
            rows = conn.execute(
                f"SELECT id FROM sessions WHERE status IN ({placeholders}) "
                "AND COALESCE(updated_at, created_at) < ?",
                (*_TERMINAL_SESSION_STATUSES, finished_cutoff),
            ).fetchall()
            expired_session_ids = [str(r[0]) for r in rows]

        # 3. Prune expired tool_observations (never touch active/in-flight sessions)
        if "tool_observations" in tables:
            if "sessions" in tables:
                term_placeholders = ",".join("?" for _ in _TERMINAL_SESSION_STATUSES)
                cond = (
                    f"(created_at < ? AND (session_id NOT IN ("
                    f"SELECT id FROM sessions WHERE status NOT IN ({term_placeholders})"
                    f")))"
                )
                params: tuple[Any, ...] = (tool_cutoff, *_TERMINAL_SESSION_STATUSES)
            else:
                cond = "created_at < ?"
                params = (tool_cutoff,)

            if dry_run:
                cnt = conn.execute(f"SELECT COUNT(*) FROM tool_observations WHERE {cond}", params).fetchone()[0]
                result["deleted"]["tool_observations"] = int(cnt or 0)
            else:
                cur = conn.execute(f"DELETE FROM tool_observations WHERE {cond}", params)
                result["deleted"]["tool_observations"] = int(cur.rowcount or 0)

        # 4. Prune messages, search_correlations, and terminal sessions older than finished_sessions_retention_days
        if expired_session_ids:
            batch_size = 500
            for i in range(0, len(expired_session_ids), batch_size):
                chunk = expired_session_ids[i : i + batch_size]
                ph = ",".join("?" for _ in chunk)
                if "messages" in tables:
                    if dry_run:
                        cnt = conn.execute(
                            f"SELECT COUNT(*) FROM messages WHERE session_id IN ({ph})",
                            chunk,
                        ).fetchone()[0]
                        result["deleted"]["messages"] += int(cnt or 0)
                    else:
                        cur = conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", chunk)
                        result["deleted"]["messages"] += int(cur.rowcount or 0)
                if "tool_observations" in tables:
                    if dry_run:
                        cnt = conn.execute(
                            f"SELECT COUNT(*) FROM tool_observations WHERE session_id IN ({ph}) AND created_at >= ?",
                            (*chunk, tool_cutoff),
                        ).fetchone()[0]
                        result["deleted"]["tool_observations"] += int(cnt or 0)
                    else:
                        cur = conn.execute(f"DELETE FROM tool_observations WHERE session_id IN ({ph})", chunk)
                        result["deleted"]["tool_observations"] += int(cur.rowcount or 0)
                if "search_correlations" in tables:
                    if dry_run:
                        cnt = conn.execute(
                            f"SELECT COUNT(*) FROM search_correlations WHERE session_id IN ({ph})", chunk
                        ).fetchone()[0]
                        result["deleted"]["search_correlations"] += int(cnt or 0)
                    else:
                        cur = conn.execute(f"DELETE FROM search_correlations WHERE session_id IN ({ph})", chunk)
                        result["deleted"]["search_correlations"] += int(cur.rowcount or 0)
                if dry_run:
                    result["deleted"]["sessions"] += len(chunk)
                else:
                    cur = conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", chunk)
                    result["deleted"]["sessions"] += int(cur.rowcount or 0)

        # 5. Prune finished drafts older than finished_sessions_retention_days
        if "drafts" in tables:
            d_ph = ",".join("?" for _ in _TERMINAL_DRAFT_STATUSES)
            d_params = (*_TERMINAL_DRAFT_STATUSES, finished_cutoff)
            if dry_run:
                cnt = conn.execute(
                    f"SELECT COUNT(*) FROM drafts WHERE status IN ({d_ph}) "
                    "AND COALESCE(updated_at, created_at) < ?",
                    d_params,
                ).fetchone()[0]
                result["deleted"]["drafts"] = int(cnt or 0)
            else:
                cur = conn.execute(
                    f"DELETE FROM drafts WHERE status IN ({d_ph}) "
                    "AND COALESCE(updated_at, created_at) < ?",
                    d_params,
                )
                result["deleted"]["drafts"] = int(cur.rowcount or 0)

        # 6. Reconcile/prune failed pending_ops (already-published session OR older than failed_ops_retention_days)
        if "pending_ops" in tables:
            if "sessions" in tables:
                t_ph = ",".join("?" for _ in _TERMINAL_SESSION_STATUSES)
                po_where = (
                    f"status = 'failed' AND ("
                    f"created_at < ? OR session_id IN (SELECT id FROM sessions WHERE status IN ({t_ph}))"
                    f")"
                )
                po_params: tuple[Any, ...] = (failed_ops_cutoff, *_TERMINAL_SESSION_STATUSES)
            else:
                po_where = "status = 'failed' AND created_at < ?"
                po_params = (failed_ops_cutoff,)

            if dry_run:
                cnt = conn.execute(f"SELECT COUNT(*) FROM pending_ops WHERE {po_where}", po_params).fetchone()[0]
                result["deleted"]["pending_ops"] = int(cnt or 0)
            else:
                cur = conn.execute(f"DELETE FROM pending_ops WHERE {po_where}", po_params)
                result["deleted"]["pending_ops"] = int(cur.rowcount or 0)

        if not dry_run:
            conn.commit()
            # 7. WAL checkpoint (TRUNCATE)
            try:
                cp = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if cp:
                    result["wal_checkpoint"] = {
                        "busy": int(cp[0]),
                        "log_frames": int(cp[1]),
                        "checkpointed_frames": int(cp[2]),
                    }
            except sqlite3.Error as exc:
                result["wal_checkpoint_error"] = str(exc)

            # 8. Conditional VACUUM
            do_vac, ratio = _should_vacuum(
                conn,
                last_vacuum_iso=last_vacuum_iso,
                policy=policy,
                force_vacuum=force_vacuum,
                now=now,
            )
            result["freelist_ratio_before_vacuum"] = round(ratio, 4)
            if do_vac:
                try:
                    conn.execute("VACUUM")
                    result["vacuumed"] = True
                    # Checkpoint again after VACUUM to keep WAL minimal
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error as exc:
                    result["vacuum_error"] = str(exc)
                    logger.warning("SQLite VACUUM skipped on %s: %s", db_path, exc)
    finally:
        conn.close()

    result["size_after"] = _file_sizes_bytes(db_path)
    return result


def _maintain_capture_db(
    capture_db_path: Path,
    policy: SqliteMaintenancePolicy,
    *,
    now: datetime,
    last_vacuum_iso: str | None,
    force_vacuum: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Prune expired/excess raw_payloads, compact oversized payloads, and VACUUM drafts_capture.db."""
    result: dict[str, Any] = {
        "db_path": str(capture_db_path),
        "exists": capture_db_path.exists(),
        "dry_run": dry_run,
        "size_before": _file_sizes_bytes(capture_db_path),
        "deleted_rows": 0,
        "truncated_rows": 0,
        "vacuumed": False,
    }
    if not capture_db_path.exists():
        result["size_after"] = result["size_before"]
        return result

    cutoff = (now - timedelta(days=max(1, policy.capture_retention_days))).isoformat()
    conn = sqlite3.connect(str(capture_db_path), timeout=15.0)
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "raw_payloads" not in tables:
            result["size_after"] = _file_sizes_bytes(capture_db_path)
            return result

        # 1. Delete rows older than capture_retention_days
        if dry_run:
            c1 = conn.execute("SELECT COUNT(*) FROM raw_payloads WHERE created_at < ?", (cutoff,)).fetchone()[0]
            result["deleted_rows"] += int(c1 or 0)
        else:
            cur = conn.execute("DELETE FROM raw_payloads WHERE created_at < ?", (cutoff,))
            result["deleted_rows"] += int(cur.rowcount or 0)

        # 2. Enforce max_rows_per_endpoint
        endpoints = [
            r[0]
            for r in conn.execute("SELECT DISTINCT endpoint FROM raw_payloads").fetchall()
        ]
        for ep in endpoints:
            if dry_run:
                c2 = conn.execute(
                    "SELECT COUNT(*) FROM raw_payloads WHERE endpoint = ? AND id NOT IN ("
                    "SELECT id FROM raw_payloads WHERE endpoint = ? ORDER BY id DESC LIMIT ?"
                    ")",
                    (ep, ep, policy.capture_max_rows_per_endpoint),
                ).fetchone()[0]
                result["deleted_rows"] += int(c2 or 0)
            else:
                cur = conn.execute(
                    "DELETE FROM raw_payloads WHERE endpoint = ? AND id NOT IN ("
                    "SELECT id FROM raw_payloads WHERE endpoint = ? ORDER BY id DESC LIMIT ?"
                    ")",
                    (ep, ep, policy.capture_max_rows_per_endpoint),
                )
                result["deleted_rows"] += int(cur.rowcount or 0)

        # 3. Compact existing oversized payload_json rows (> 16 KB) while preserving JSON structure
        oversized_threshold = max(8192, policy.capture_max_str_len * 4)
        oversized_rows = conn.execute(
            "SELECT id, payload_json FROM raw_payloads WHERE length(payload_json) > ?",
            (oversized_threshold,),
        ).fetchall()
        if dry_run:
            result["truncated_rows"] = len(oversized_rows)
        else:
            for row_id, raw_json in oversized_rows:
                try:
                    parsed = json.loads(raw_json)
                    compacted = json.dumps(
                        truncate_payload_values(parsed, max_str_len=policy.capture_max_str_len),
                        ensure_ascii=False,
                    )
                    if len(compacted) < len(raw_json):
                        conn.execute(
                            "UPDATE raw_payloads SET payload_json = ? WHERE id = ?",
                            (compacted, row_id),
                        )
                        result["truncated_rows"] += 1
                except Exception:
                    continue

        if not dry_run:
            conn.commit()
            do_vac, ratio = _should_vacuum(
                conn,
                last_vacuum_iso=last_vacuum_iso,
                policy=policy,
                force_vacuum=force_vacuum,
                now=now,
            )
            result["freelist_ratio_before_vacuum"] = round(ratio, 4)
            if do_vac:
                try:
                    conn.execute("VACUUM")
                    result["vacuumed"] = True
                except sqlite3.Error as exc:
                    result["vacuum_error"] = str(exc)
                    logger.warning("SQLite VACUUM skipped on %s: %s", capture_db_path, exc)
    finally:
        conn.close()

    result["size_after"] = _file_sizes_bytes(capture_db_path)
    return result


def run_sqlite_maintenance(
    db_path: str | Path,
    *,
    capture_db_path: str | Path | None = None,
    policy: SqliteMaintenancePolicy | None = None,
    force_vacuum: bool = False,
    dry_run: bool = False,
    state_file: Path | str | None = DEFAULT_STATE_FILE,
) -> dict[str, Any]:
    """Run full SQLite TTL cleanup, WAL checkpoint, and conditional VACUUM."""
    pol = policy or SqliteMaintenancePolicy()
    p_db = Path(os.path.expanduser(str(db_path)))
    p_cap = (
        Path(os.path.expanduser(str(capture_db_path)))
        if capture_db_path is not None
        else p_db.with_name(p_db.stem + "_capture.db")
    )
    p_state = Path(os.path.expanduser(str(state_file))) if state_file is not None else None

    now = datetime.now(timezone.utc)
    prev_state = _read_state(p_state)
    last_vac_drafts = prev_state.get("last_vacuum_drafts_at")
    last_vac_cap = prev_state.get("last_vacuum_capture_at")

    drafts_res = _maintain_drafts_db(
        p_db,
        pol,
        now=now,
        last_vacuum_iso=last_vac_drafts,
        force_vacuum=force_vacuum,
        dry_run=dry_run,
    )
    capture_res = _maintain_capture_db(
        p_cap,
        pol,
        now=now,
        last_vacuum_iso=last_vac_cap,
        force_vacuum=force_vacuum,
        dry_run=dry_run,
    )

    summary = {
        "ran_at": now.isoformat(),
        "dry_run": dry_run,
        "drafts_db": drafts_res,
        "capture_db": capture_res,
        "last_vacuum_drafts_at": now.isoformat() if drafts_res.get("vacuumed") else last_vac_drafts,
        "last_vacuum_capture_at": now.isoformat() if capture_res.get("vacuumed") else last_vac_cap,
    }
    if not dry_run:
        _write_state(p_state, summary)
        logger.info(
            "SQLite maintenance completed: drafts deleted=%s (%.2f->%.2f MB), "
            "capture deleted=%d truncated=%d (%.2f->%.2f MB)",
            drafts_res["deleted"],
            drafts_res["size_before"]["total_bytes"] / (1024 * 1024),
            drafts_res["size_after"]["total_bytes"] / (1024 * 1024),
            capture_res["deleted_rows"],
            capture_res["truncated_rows"],
            capture_res["size_before"]["total_bytes"] / (1024 * 1024),
            capture_res["size_after"]["total_bytes"] / (1024 * 1024),
        )
    return summary


def read_last_sqlite_maintenance(state_file: Path | str | None = DEFAULT_STATE_FILE) -> dict[str, Any]:
    """Read the last persisted SQLite maintenance summary."""
    p_state = Path(os.path.expanduser(str(state_file))) if state_file is not None else None
    return _read_state(p_state)


class SqliteMaintenanceWorker:
    """Periodic background worker that runs SQLite maintenance inside TM Daemon."""

    def __init__(
        self,
        db_path: str,
        *,
        capture_db_path: str | None = None,
        policy: SqliteMaintenancePolicy | None = None,
        state_file: Path | str | None = DEFAULT_STATE_FILE,
        initial_delay_seconds: float = 60.0,
    ) -> None:
        self._db_path = db_path
        self._capture_db_path = capture_db_path
        self._policy = policy or SqliteMaintenancePolicy()
        self._state_file = Path(os.path.expanduser(str(state_file))) if state_file else None
        self._initial_delay_seconds = initial_delay_seconds
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_result: dict[str, Any] | None = None

    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def start(self) -> asyncio.Task | None:
        if not self._policy.enabled or self._db_path == ":memory:":
            return None
        if self.is_running():
            return self._task
        self._running = True
        self._task = asyncio.create_task(self._loop())
        return self._task

    def stop(self) -> None:
        self._running = False

    async def _loop(self) -> None:
        interval_seconds = max(300, int(self._policy.interval_hours) * 3600)
        try:
            await asyncio.sleep(self._initial_delay_seconds)
            while self._running:
                try:
                    self._last_result = await asyncio.to_thread(
                        run_sqlite_maintenance,
                        self._db_path,
                        capture_db_path=self._capture_db_path,
                        policy=self._policy,
                        state_file=self._state_file,
                    )
                except Exception as exc:
                    logger.warning("SqliteMaintenanceWorker run failed: %s", exc)
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            pass
