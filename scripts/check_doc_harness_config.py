"""Validate doc-harness.project.yaml: required keys and referenced paths exist."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "doc-harness.project.yaml"


def _fail(msg: str) -> None:
    print(f"doc-harness config: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not CONFIG.is_file():
        _fail(f"missing {CONFIG.relative_to(REPO_ROOT)}")

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail("root must be a mapping")

    version = raw.get("version")
    if version != 1:
        _fail(f"expected version: 1, got {version!r}")

    if "design_docs" not in raw or not isinstance(raw["design_docs"], dict):
        _fail("missing or invalid section: design_docs")

    dd = raw["design_docs"]
    for k in ("root", "index"):
        if k not in dd or not isinstance(dd[k], str):
            _fail(f"design_docs.{k} must be a non-empty string path")

    if "commands" not in raw or not isinstance(raw["commands"], dict):
        _fail("missing or invalid section: commands")
    cmd = raw["commands"]
    if "doc_check" not in cmd or not cmd["doc_check"]:
        _fail("commands.doc_check must be set")
    if "plan_check" in cmd:
        _fail("commands.plan_check must not be set (removed)")

    if raw.get("guides"):
        _fail("guides section must be absent (maintenance text is in doc-health skill)")

    paths_to_check = [dd["index"]]
    wl = raw.get("whitelists")
    if wl is not None:
        if not isinstance(wl, dict):
            _fail("whitelists must be a mapping")
        if "plan_structure" in wl:
            _fail("whitelists.plan_structure must not be set (removed)")
        for k, p in wl.items():
            if not isinstance(p, str) or not p:
                _fail(f"whitelists.{k} must be a non-empty string")
            paths_to_check.append(p)

    for rel in paths_to_check:
        p = REPO_ROOT / rel
        if not p.is_file():
            _fail(f"path not found: {rel}")

    print(f"OK {CONFIG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
