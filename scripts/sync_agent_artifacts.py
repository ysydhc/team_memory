#!/usr/bin/env python3
"""Materialize .claude/.cursor agents, .cursor prompts, and .claude skills from SSOT.

Edit bodies under agents/shared/{bodies,prompts}/ and metadata in agents/manifest.yaml,
then run:  make sync-agent-artifacts
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "agents" / "manifest.yaml"
SHARED = REPO_ROOT / "agents" / "shared"


def load_manifest() -> dict:
    raw = MANIFEST.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise SystemExit("manifest root must be a mapping")
    return data


def render_claude_agent(meta: dict, body: str) -> str:
    claude = meta["claude"]
    lines = ["---", f"name: {claude['name']}", f"description: {claude['description']}"]
    tools = claude.get("tools")
    if tools:
        lines.append("tools:")
        lines.extend(f"  - {t}" for t in tools)
    dis = claude.get("disallowed_tools")
    if dis:
        lines.append("disallowedTools:")
        lines.extend(f"  - {t}" for t in dis)
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.rstrip() + "\n"


def render_cursor_agent(meta: dict, body: str) -> str:
    cur = meta["cursor"]
    lines = ["---", f"name: {cur['name']}"]
    if cur.get("model") is not None:
        lines.append(f"model: {cur['model']}")
    lines.append(f"description: {cur['description']}")
    if "readonly" in cur:
        lines.append(f"readonly: {str(cur['readonly']).lower()}")
    if cur.get("color") is not None:
        lines.append(f"color: {cur['color']}")
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.rstrip() + "\n"


def render_skill(skill_name: str, skill_description: str, body: str) -> str:
    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {skill_description}",
        "---",
    ]
    return "\n".join(lines) + "\n\n" + body.rstrip() + "\n"


def main() -> int:
    m = load_manifest()
    for p in m.get("prompts", []):
        pid = p["id"]
        src = SHARED / "prompts" / f"{pid}.md"
        if not src.is_file():
            raise SystemExit(f"missing prompt source: {src}")
        body = src.read_text(encoding="utf-8")

        out_prompt = REPO_ROOT / ".cursor" / "prompts" / f"{pid}.md"
        out_prompt.parent.mkdir(parents=True, exist_ok=True)
        out_prompt.write_text(body, encoding="utf-8")

        skill_name = str(p["skill_name"])
        skill_dir = REPO_ROOT / ".claude" / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            render_skill(skill_name, str(p["skill_description"]), body),
            encoding="utf-8",
        )

    for a in m.get("agents", []):
        aid = a["id"]
        src = SHARED / "bodies" / f"{aid}.md"
        if not src.is_file():
            raise SystemExit(f"missing agent body: {src}")
        body = src.read_text(encoding="utf-8")

        ca = REPO_ROOT / ".claude" / "agents" / f"{aid}.md"
        ca.parent.mkdir(parents=True, exist_ok=True)
        ca.write_text(render_claude_agent(a, body), encoding="utf-8")

        cua = REPO_ROOT / ".cursor" / "agents" / f"{aid}.md"
        cua.parent.mkdir(parents=True, exist_ok=True)
        cua.write_text(render_cursor_agent(a, body), encoding="utf-8")

    print("sync-agent-artifacts: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
