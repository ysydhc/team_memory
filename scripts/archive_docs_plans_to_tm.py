#!/usr/bin/env python3
"""Archive docs/plans/*.md snapshots to Team Memory (POST + attachment upload).

Historically this repo listed concrete paths under ``docs/plans/``; those files were
removed after archival. To run again, restore Markdown (or add new paths) and append
entries to ``SPECS`` with keys: rel_path, title, content_type, value_summary, scope,
scope_ref, tags, overview, solution_doc.

Usage (repo root):
  set -a && source .env && set +a
  uv run python scripts/archive_docs_plans_to_tm.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

# Append dicts here when local plan Markdown exists (see module docstring).
SPECS: list[dict[str, str | list[str]]] = []


def _base() -> str:
    return os.environ.get("TM_BASE_URL", "http://localhost:9111").rstrip("/")


def _headers_json() -> dict[str, str]:
    key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if not key:
        print("TEAM_MEMORY_API_KEY required", file=sys.stderr)
        sys.exit(1)
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _project() -> str:
    return os.environ.get("TEAM_MEMORY_PROJECT", "team_memory")


def main() -> None:
    if not SPECS:
        print(
            "archive_docs_plans_to_tm: SPECS is empty; nothing to archive.",
            file=sys.stderr,
        )
        print(json.dumps({"docs_plans_archived": []}, ensure_ascii=False, indent=2))
        return

    headers = _headers_json()
    project = _project()
    results: list[dict[str, str]] = []

    with httpx.Client(timeout=120.0) as client:
        r0 = client.get(
            f"{_base()}/api/v1/archives",
            params={"q": "【docs-plans】", "limit": 30, "project": project},
            headers={"Authorization": headers["Authorization"]},
        )
        if r0.is_success:
            existing = {it.get("title") for it in r0.json().get("items", [])}
            print("Existing 【docs-plans】 titles:", sorted(existing) or "(none)")

        for spec in SPECS:
            path = ROOT / str(spec["rel_path"])
            if not path.is_file():
                print("SKIP missing file:", path, file=sys.stderr)
                continue

            body = {
                "title": spec["title"],
                "solution_doc": spec["solution_doc"],
                "content_type": spec["content_type"],
                "value_summary": spec["value_summary"],
                "overview": spec["overview"],
                "tags": spec["tags"],
                "project": project,
                "scope": spec["scope"],
                "scope_ref": spec["scope_ref"],
            }
            r = client.post(f"{_base()}/api/v1/archives", json=body, headers=headers)
            r.raise_for_status()
            item = r.json().get("item") or {}
            aid = str(item.get("archive_id", ""))
            print("Archive OK:", spec["title"], "->", aid, item.get("action"))

            up = f"{_base()}/api/v1/archives/{aid}/attachments/upload"
            with path.open("rb") as f:
                ru = client.post(
                    up,
                    headers={"Authorization": headers["Authorization"]},
                    params={"project": project},
                    files={"file": (path.name, f, "text/markdown")},
                    data={
                        "kind": "plan_doc",
                        "source_path": str(spec["rel_path"]),
                    },
                )
            ru.raise_for_status()
            print("  attached:", path.name)

            results.append(
                {"title": str(spec["title"]), "archive_id": aid, "file": str(spec["rel_path"])}
            )

    print(json.dumps({"docs_plans_archived": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
