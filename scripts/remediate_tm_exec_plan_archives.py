#!/usr/bin/env python3
"""Remediate low-quality TM archives: fetch attachment texts, then operator edits YAML and POSTs.

Default manifest: data/TM-ARCHIVE-MANIFEST.md (see docs/README.md).

Usage (after filling remediate_tm_archive_bodies.yaml):
  set -a && source .env && set +a
  python scripts/remediate_tm_exec_plan_archives.py --apply remediate_tm_archive_bodies.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx
import yaml

# Local cache (gitignored): export-sources / remediation JSON / optional copies
_DEFAULT_TMP = Path(".tmp/data")
_DEFAULT_TM_EXPORT_DIR = _DEFAULT_TMP / "tm-archive-export"
_DEFAULT_TM_REMEDIATION_DIR = _DEFAULT_TMP / "tm-remediation"
# Committed slug → archive_id table for export-sources
_DEFAULT_MANIFEST = Path("data/TM-ARCHIVE-MANIFEST.md")


def _pick_plan_text(archive: dict) -> tuple[str, str]:
    """Return (label, markdown) from attachments — prefer path/snippet mentioning plan.md."""
    atts = archive.get("attachments") or []
    candidates: list[tuple[int, str, str]] = []
    for a in atts:
        sn = (a.get("snippet") or "") + " " + (a.get("path") or "")
        aid = a.get("id")
        if not aid:
            continue
        score = 0
        if "1-plan/plan.md" in sn or sn.endswith("plan.md"):
            score += 10
        elif "plan.md" in sn:
            score += 5
        elif "/plan.md" in sn:
            score += 4
        elif "plan" in sn.lower():
            score += 1
        candidates.append((score, str(aid), sn.strip()))
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1], candidates[0][2] if candidates else ("", "")


def fetch_archive(client: httpx.Client, base: str, headers: dict, aid: str, project: str) -> dict:
    r = client.get(
        f"{base}/api/v1/archives/{aid}",
        headers=headers,
        params={"project": project},
    )
    r.raise_for_status()
    return r.json()


def download_attachment(
    client: httpx.Client,
    base: str,
    headers: dict,
    archive_id: str,
    attachment_id: str,
    project: str,
) -> bytes:
    r = client.get(
        f"{base}/api/v1/archives/{archive_id}/attachments/{attachment_id}/file",
        headers=headers,
        params={"project": project},
    )
    r.raise_for_status()
    return r.content


def cmd_export(manifest: Path, out_dir: Path) -> None:
    """Export plan (or best attachment) text per archive_id from manifest table."""
    base = os.environ.get("TM_BASE_URL", "http://localhost:9111").rstrip("/")
    key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if not key:
        sys.exit("TEAM_MEMORY_API_KEY required")
    project = os.environ.get("TEAM_MEMORY_PROJECT", "team_memory")
    headers = {"Authorization": f"Bearer {key}"}

    raw = manifest.read_text(encoding="utf-8")
    ids: dict[str, str] = {}
    for line in raw.splitlines():
        m = re.search(
            r"\|\s*(\S+)\s*\|\s*【exec-plan·completed】(\S+)\s*\|\s*([a-f0-9-]{36})\s*\|", line
        )
        if m:
            ids[m.group(2)] = m.group(3)

    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120.0) as client:
        for slug, aid in sorted(ids.items()):
            arch = fetch_archive(client, base, headers, aid, project)
            att_id, _ = _pick_plan_text(arch)
            if not att_id:
                (out_dir / f"{slug}.missing.txt").write_text("no attachments\n", encoding="utf-8")
                continue
            body = download_attachment(client, base, headers, aid, att_id, project)
            (out_dir / f"{slug}.source.md").write_bytes(body)
            meta = {
                "slug": slug,
                "archive_id": aid,
                "title": arch.get("title"),
                "picked_attachment_id": att_id,
                "linked_experience_ids": arch.get("linked_experience_ids") or [],
            }
            (out_dir / f"{slug}.meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"exported {slug} -> {out_dir / f'{slug}.source.md'}")


def _merge_links_from_meta(payload: dict, meta_path: Path) -> None:
    """Preserve TM experience links: empty POST clears links in DB (see archive_repository)."""
    if payload.get("linked_experience_ids"):
        return
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lids = meta.get("linked_experience_ids") or []
    if lids:
        payload["linked_experience_ids"] = lids


def cmd_apply(yaml_path: Path) -> None:
    base = os.environ.get("TM_BASE_URL", "http://localhost:9111").rstrip("/")
    key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if not key:
        sys.exit("TEAM_MEMORY_API_KEY required")
    project = os.environ.get("TEAM_MEMORY_PROJECT", "team_memory")
    headers = {"Authorization": f"Bearer {key}"}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    for entry in data.get("archives", []):
        title = entry["title"]
        payload = {
            "title": title,
            "overview": entry["overview"],
            "solution_doc": entry["solution_doc"],
            "content_type": entry.get("content_type", "tech_design"),
            "value_summary": entry.get("value_summary"),
            "tags": entry.get("tags"),
            "project": project,
            "scope": entry.get("scope", "project"),
            "scope_ref": entry.get("scope_ref"),
        }
        if payload.get("value_summary") is None:
            payload.pop("value_summary", None)
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{base}/api/v1/archives", json=payload, headers=headers)
            r.raise_for_status()
            print(r.json().get("message"), title)


def cmd_apply_json_map(map_path: Path, export_dir: Path) -> None:
    """POST each entry in ``{\"slug\": { ... }}`` (ArchiveCreateRequest-shaped objects)."""
    base = os.environ.get("TM_BASE_URL", "http://localhost:9111").rstrip("/")
    key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if not key:
        sys.exit("TEAM_MEMORY_API_KEY required")
    project = os.environ.get("TEAM_MEMORY_PROJECT", "team_memory")
    headers = {"Authorization": f"Bearer {key}"}

    root = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        sys.exit("JSON root must be an object: {slug: payload, ...}")

    with httpx.Client(timeout=120.0) as client:
        for slug in sorted(root.keys()):
            payload = root[slug]
            if not isinstance(payload, dict):
                raise TypeError(f"Payload for {slug} must be an object")
            _merge_links_from_meta(payload, export_dir / f"{slug}.meta.json")
            payload.setdefault("project", project)
            r = client.post(f"{base}/api/v1/archives", json=payload, headers=headers)
            r.raise_for_status()
            print(r.json().get("message"), payload.get("title", slug))


def cmd_apply_json_dir(
    bodies_dir: Path,
    export_dir: Path,
    *,
    glob_pat: str = "*.json",
) -> None:
    """POST one JSON file per slug (filename stem = slug, e.g. archive.json).

    Each file is the POST body (title, overview, solution_doc, ...).
    If ``linked_experience_ids`` is absent, values are copied from
    ``export_dir / f\"{slug}.meta.json\"`` when present (see _merge_links_from_meta).
    """
    base = os.environ.get("TM_BASE_URL", "http://localhost:9111").rstrip("/")
    key = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if not key:
        sys.exit("TEAM_MEMORY_API_KEY required")
    project = os.environ.get("TEAM_MEMORY_PROJECT", "team_memory")
    headers = {"Authorization": f"Bearer {key}"}

    paths = sorted(bodies_dir.glob(glob_pat))
    if not paths:
        sys.exit(f"No files matching {glob_pat} under {bodies_dir}")

    with httpx.Client(timeout=120.0) as client:
        for path in paths:
            if path.name.startswith("."):
                continue
            slug = path.stem
            payload = json.loads(path.read_text(encoding="utf-8"))
            _merge_links_from_meta(payload, export_dir / f"{slug}.meta.json")
            payload.setdefault("project", project)
            r = client.post(f"{base}/api/v1/archives", json=payload, headers=headers)
            r.raise_for_status()
            print(r.json().get("message"), payload.get("title", slug))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-sources", help="Download best plan-like attachment per slug")
    e.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
    )
    e.add_argument("--out", type=Path, default=_DEFAULT_TM_EXPORT_DIR)
    e.set_defaults(fn=lambda a: cmd_export(a.manifest, a.out))

    apy = sub.add_parser("apply", help="POST bodies from YAML")
    apy.add_argument("yaml_file", type=Path)
    apy.set_defaults(fn=lambda a: cmd_apply(a.yaml_file))

    aj = sub.add_parser(
        "apply-json-dir",
        help="POST one JSON per file (stem=slug); merge linked_experience_ids from export meta",
    )
    aj.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_TM_REMEDIATION_DIR,
        help="Directory with <slug>.json bodies",
    )
    aj.add_argument(
        "--export-dir",
        type=Path,
        default=_DEFAULT_TM_EXPORT_DIR,
        help="Directory with <slug>.meta.json from export-sources",
    )
    aj.add_argument("--glob", default="*.json", dest="glob_pat")
    aj.set_defaults(fn=lambda a: cmd_apply_json_dir(a.dir, a.export_dir, glob_pat=a.glob_pat))

    am = sub.add_parser(
        "apply-json-map",
        help=(
            "POST from one JSON file {slug: payload}; merge linked_experience_ids from export meta"
        ),
    )
    am.add_argument(
        "map_file",
        type=Path,
        default=_DEFAULT_TM_REMEDIATION_DIR / "exec-plan-bodies.json",
        nargs="?",
    )
    am.add_argument(
        "--export-dir",
        type=Path,
        default=_DEFAULT_TM_EXPORT_DIR,
    )
    am.set_defaults(fn=lambda a: cmd_apply_json_map(a.map_file, a.export_dir))

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
