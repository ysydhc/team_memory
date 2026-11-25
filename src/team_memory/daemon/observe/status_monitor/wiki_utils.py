"""Local Wiki SQLite lookup and query display formatting helpers for status_monitor."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any

_WIKI_DB_CANDIDATES: tuple[str, ...] = (
    os.path.join(os.path.expanduser("~"), ".cache", "tm-pipeline", "wiki.db"),
    os.path.join(os.path.expanduser("~"), ".team_memory", "wiki.db"),
)


def _pkg_attr(name: str, default: Any) -> Any:
    """Resolve an attribute from the parent status_monitor package for mock.patch compatibility."""
    pkg = sys.modules.get("team_memory.daemon.observe.status_monitor")
    return getattr(pkg, name, default)


def _parse_json_field(val: Any) -> dict[str, Any]:
    """Parse JSONB or JSON string field into a dictionary."""
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_list_field(val: Any) -> list[str]:
    """Safely parse a JSONB / array / text field from asyncpg into a list of experience ID strings."""
    raw_items: list[Any] = []
    if isinstance(val, (list, tuple)):
        raw_items = list(val)
    elif isinstance(val, str) and val.strip():
        s = val.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    raw_items = parsed
            except Exception:
                raw_items = [s]
        else:
            raw_items = [s]

    out: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            eid = str(item.get("id") or item.get("experience_id") or "").strip()
            if eid:
                out.append(eid)
        elif item:
            s_item = str(item).strip()
            if s_item:
                out.append(s_item)
    return out


def _lookup_local_wiki_entries(
    ids: list[str],
    candidate_dbs: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Look up Wiki entry metadata from local SQLite wiki.db files by ID."""
    if not ids:
        return {}
    dbs = candidate_dbs if candidate_dbs is not None else _pkg_attr("_WIKI_DB_CANDIDATES", _WIKI_DB_CANDIDATES)
    found: dict[str, dict[str, Any]] = {}
    placeholders = ",".join("?" for _ in ids)
    for db_p in dbs:
        if not os.path.exists(db_p):
            continue
        try:
            conn = sqlite3.connect(f"file:{db_p}?mode=ro", uri=True, timeout=1.0)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    f"""
                    SELECT *
                    FROM wiki_entries
                    WHERE id IN ({placeholders})
                    """,
                    tuple(ids),
                )
                for row in cur.fetchall():
                    keys = set(row.keys())
                    wid = str(row["id"])
                    if wid in found:
                        continue
                    raw_tags = row["tags"] if "tags" in keys else ""
                    tags_list: list[str] = []
                    if raw_tags:
                        try:
                            tags_list = [str(t) for t in json.loads(raw_tags)]
                        except Exception:
                            tags_list = [str(raw_tags)]
                    found[wid] = {
                        "id": wid,
                        "title": str(row["title"] or "") if "title" in keys else "",
                        "content": str(row["content"] or "") if "content" in keys else "",
                        "project": str(row["project"] or "default") if "project" in keys else "default",
                        "tags": tags_list[:6],
                        "source_url": str(row["source_url"] or "") if "source_url" in keys else "",
                    }
            finally:
                conn.close()
        except Exception:
            continue
    return found


def _clean_query_display(query: str, max_len: int = 58) -> str:
    """Clean up single-line display of a recall query."""
    raw = query or ""
    if "<CONTEXT_SUMMARY>" in raw or "# User Requests" in raw:
        sec = raw.split("# User Requests")[-1].split("# Previous Session Summary")[0]
        req_lines = [
            ln.strip()
            for ln in sec.splitlines()
            if ln.strip() and ln.strip()[0].isdigit() and ". " in ln
        ]
        if req_lines:
            last_req = req_lines[-1].split(". ", 1)[-1].strip()
            raw = f"[Ctx] {last_req}"
    q = " ".join(raw.replace("\n", " ").split())
    if len(q) > max_len:
        return q[: max_len - 1] + "…"
    return q
