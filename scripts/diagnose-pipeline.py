#!/usr/bin/env python3
"""TM Pipeline Diagnostic — 详细诊断数据链路状态。

Usage: python scripts/diagnose-pipeline.py [--session SESSION_ID] [--recent N]

Shows:
- Per-project message statistics
- Session details with message counts
- Recent messages with content preview
- PayloadCapture summary
- Convergence status
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path.home() / ".cache" / "tm-pipeline" / "drafts.db"
CAPTURE_DB = Path.home() / ".cache" / "tm-pipeline" / "drafts_capture.db"


def get_db():
    if not DB_PATH.exists():
        print(f"❌ DB not found: {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def show_project_stats(db):
    print("=== 按 Project 统计 ===\n")
    rows = db.execute("""
        SELECT COALESCE(s.project, 'unknown') as proj,
               COUNT(DISTINCT s.id) as sessions,
               SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) as user_n,
               SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) as asst_n
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id
        GROUP BY s.project
        ORDER BY sessions DESC
    """).fetchall()

    print(f"  {'Project':<25s} {'Sessions':>8s} {'User':>6s} {'Asst':>6s} {'Status'}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 10}")
    for proj, sessions, user_n, asst_n in rows:
        status = "✅ OK" if asst_n > 0 else "⚠️  无 assistant 消息"
        print(f"  {proj:<25s} {sessions:>8d} {user_n:>6d} {asst_n:>6d}   {status}")
    print()


def show_sessions(db, limit=10):
    print(f"=== 最近 {limit} 个 Session ===\n")
    rows = db.execute(f"""
        SELECT s.id, s.project, s.status, s.created_at, s.updated_at,
               (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'user') as user_n,
               (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'assistant') as asst_n
        FROM sessions s
        ORDER BY s.updated_at DESC
        LIMIT {limit}
    """).fetchall()

    for sid, proj, status, created, updated, user_n, asst_n in rows:
        status_icon = {"active": "🟢", "needs_refinement": "🟡", "published": "✅"}.get(status, "❓")
        conv_status = "已收敛" if status == "needs_refinement" else ("已发布" if status == "published" else "进行中")
        print(f"  {status_icon} {sid[:20]}")
        print(f"     Project: {proj}  Status: {status} ({conv_status})")
        print(f"     Messages: {user_n} user, {asst_n} assistant")
        print(f"     Created: {created}")
        print(f"     Updated: {updated}")
        if asst_n == 0 and user_n > 0:
            print("     ⚠️  有 user 消息但无 assistant — hook 可能没触发或旧数据")
        print()


def show_messages(db, session_id=None, limit=10):
    if session_id:
        where = f"WHERE m.session_id = '{session_id}'"
        label = f"Session {session_id[:16]}"
    else:
        where = ""
        label = "全部"

    print(f"=== 最近 {limit} 条消息 ({label}) ===\n")
    rows = db.execute(f"""
        SELECT m.session_id, m.role, substr(m.content, 1, 100), m.created_at, s.project
        FROM messages m
        LEFT JOIN sessions s ON m.session_id = s.id
        {where}
        ORDER BY m.id DESC
        LIMIT {limit}
    """).fetchall()

    for sid, role, content, ts, proj in rows:
        role_icon = "👤" if role == "user" else "🤖"
        print(f"  {role_icon} [{role:9s}] {proj or '?':<20s} {content}")
        print(f"     session={sid[:20]}  time={ts}")
        print()


def show_convergence_status(db):
    print("=== 收敛状态 ===\n")

    active = db.execute("SELECT COUNT(*) FROM sessions WHERE status='active'").fetchone()[0]
    needs = db.execute("SELECT COUNT(*) FROM sessions WHERE status='needs_refinement'").fetchone()[0]
    published = db.execute("SELECT COUNT(*) FROM sessions WHERE status='published'").fetchone()[0]

    # Active sessions with enough messages for convergence
    ready = db.execute("""
        SELECT COUNT(*) FROM sessions s
        WHERE s.status = 'active'
        AND (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'assistant') >= 3
    """).fetchone()[0]

    # Active sessions with NO assistant messages
    no_asst = db.execute("""
        SELECT COUNT(*) FROM sessions s
        WHERE s.status = 'active'
        AND (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'assistant') = 0
    """).fetchone()[0]

    print(f"  Active: {active}")
    print(f"  Needs refinement: {needs}")
    print(f"  Published: {published}")
    print(f"  Ready for convergence (3+ assistant): {ready}")
    print(f"  Stuck (0 assistant messages): {no_asst}")
    if no_asst > 0:
        print(f"  ⚠️  {no_asst} 个 session 没有 assistant 消息")
        print("     可能原因: 旧 daemon 运行时的数据，或 hook 没有触发")
    print()


def show_payload_capture():
    print("=== PayloadCapture ===\n")
    if not CAPTURE_DB.exists():
        print("  ⚠️  Capture DB 不存在（还未捕获任何 payload）")
        return

    db = sqlite3.connect(str(CAPTURE_DB))
    rows = db.execute("""
        SELECT endpoint, COUNT(*) as cnt
        FROM raw_payloads
        GROUP BY endpoint
        ORDER BY cnt DESC
    """).fetchall()

    if not rows:
        print("  ⚠️  未捕获任何 payload")
        return

    for endpoint, cnt in rows:
        print(f"  {endpoint:<35s} {cnt:>5d} payloads")

    # Show latest payload per endpoint
    print("\n  最新 payload 示例:")
    endpoints = [r[0] for r in rows]
    for ep in endpoints[:3]:
        row = db.execute(
            "SELECT source, payload_json FROM raw_payloads WHERE endpoint = ? ORDER BY id DESC LIMIT 1", (ep,)
        ).fetchone()
        if row:
            source, payload_json = row
            payload = json.loads(payload_json)
            fields = list(payload.keys())
            print(f"    {ep} (source={source})")
            print(f"      fields: {', '.join(fields)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="TM Pipeline Diagnostic")
    parser.add_argument("--session", help="Show details for specific session ID")
    parser.add_argument("--recent", type=int, default=10, help="Number of recent items to show")
    args = parser.parse_args()

    db = get_db()

    print("=" * 60)
    print("  TM Pipeline Diagnostic")
    print("=" * 60)
    print()

    show_project_stats(db)
    show_sessions(db, limit=args.recent)
    show_convergence_status(db)
    show_messages(db, session_id=args.session, limit=args.recent)
    show_payload_capture()


if __name__ == "__main__":
    main()
