#!/usr/bin/env python3
"""Doc gardening: scan markdown links for broken refs, deprecated refs, stale markers.

Scans docs/decision, docs/cmd, docs/guide, docs/ops and docs/README.md for:
- DOC_LINK_404: internal link points to non-existent file
- DOC_LINK_BROKEN: broken path or anchor
- DOC_DEPRECATED_REF: new doc references archive/deprecated (violation)
- DOC_STALE_MARKER: contains stale marker text (optional)

Output: file:line: rule_id: message
Exit 0 = no issues, non-zero = issues found.

Usage:
  python scripts/harness_doc_gardening.py [--root PATH] [--whitelist PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Rule IDs: align with .claude/skills/doc-health/SKILL.md (文档扫描与 rule_id)
RULE_LINK_404 = "DOC_LINK_404"
RULE_LINK_BROKEN = "DOC_LINK_BROKEN"
RULE_DEPRECATED_REF = "DOC_DEPRECATED_REF"
RULE_STALE_MARKER = "DOC_STALE_MARKER"

# Stale marker patterns (simple keyword match)
STALE_MARKERS = [
    "部分内容已过时",
    "已废弃",
    "已过时",
]

# Archive/deprecated path segments (for DOC_DEPRECATED_REF)
ARCHIVE_SEGMENTS = ("archive", "deprecated")

# Whitelist path: scripts/doc-gardening-whitelist.txt or next to script
DEFAULT_WHITELIST_NAME = "doc-gardening-whitelist.txt"


def _is_internal_link(url: str) -> bool:
    """True if url is an internal relative link we should validate."""
    url = url.strip()
    if not url:
        return False
    if url.startswith(("#", "http://", "https://", "mailto:", "ftp:")):
        return False
    # Only validate paths that look like doc links (reduce false positives from placeholders)
    if url.startswith(("./", "../", "docs/")):
        return True
    if "/" in url or url.endswith((".md", ".mdc", ".yaml", ".yml")):
        return True
    return False


def _resolve_link_target(source_file: Path, link_url: str, root: Path) -> Path | None:
    """Resolve link URL to absolute path. Returns None if not resolvable."""
    url = link_url.strip()
    # Strip anchor
    if "#" in url:
        url = url.split("#")[0]
    if not url:
        return None

    # docs/... from root (path from project root)
    if url.startswith("docs/"):
        try:
            resolved = (root / url).resolve()
            return resolved
        except (ValueError, OSError):
            return None

    # Relative to source file's directory
    if (
        url.startswith("./")
        or url.startswith("../")
        or (not url.startswith("/") and "://" not in url)
    ):
        base = source_file.parent
        try:
            resolved = (base / url).resolve()
            return resolved
        except (ValueError, OSError):
            return None

    return None


def _is_archive_or_deprecated(path: Path, root: Path) -> bool:
    """True if path is under archive/ or deprecated/."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = rel.parts
    for seg in ARCHIVE_SEGMENTS:
        if seg in parts:
            return True
    return False


def _load_whitelist(whitelist_path: Path | None) -> set[str]:
    """Load whitelist entries. Returns set of 'path' or 'path:line' or 'path:rule_id'."""
    if whitelist_path is None or not whitelist_path.exists():
        return set()
    entries: set[str] = set()
    for line in whitelist_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _is_whitelisted(rel_path: str, line_no: int, rule_id: str, whitelist: set[str]) -> bool:
    """Check if this violation is whitelisted."""
    # Full path
    if rel_path in whitelist:
        return True
    # Line-level
    if f"{rel_path}:{line_no}" in whitelist:
        return True
    # Line+rule-level (path:line:rule_id)
    if f"{rel_path}:{line_no}:{rule_id}" in whitelist:
        return True
    # Rule-level (path:rule_id)
    if f"{rel_path}:{rule_id}" in whitelist:
        return True
    return False


def _extract_links(content: str) -> list[tuple[int, str, str]]:
    """Extract (line_no, url, link_type) from markdown. link_type: 'inline' or 'angle'."""
    results: list[tuple[int, str, str]] = []
    lines = content.splitlines()

    # [text](url)
    inline_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    # <url>
    angle_re = re.compile(r"<([^>\s]+)>")

    for i, line in enumerate(lines, start=1):
        for m in inline_re.finditer(line):
            results.append((i, m.group(2).strip(), "inline"))
        for m in angle_re.finditer(line):
            url = m.group(1).strip()
            if not url.startswith("http") and "://" not in url:
                results.append((i, url, "angle"))
    return results


def _extract_stale_markers(content: str) -> list[tuple[int, str]]:
    """Extract (line_no, marker) for lines containing stale markers."""
    results: list[tuple[int, str]] = []
    for i, line in enumerate(content.splitlines(), start=1):
        for marker in STALE_MARKERS:
            if marker in line:
                results.append((i, marker))
                break
    return results


def check_file(
    file_path: Path,
    root: Path,
    whitelist: set[str],
) -> list[tuple[int, str, str]]:
    """Check a single markdown file. Returns list of (line, rule_id, message)."""
    violations: list[tuple[int, str, str]] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return violations

    try:
        rel_path = file_path.resolve().relative_to(root.resolve())
        rel_str = str(rel_path).replace("\\", "/")
    except ValueError:
        rel_str = str(file_path)

    source_in_archive = _is_archive_or_deprecated(file_path, root)

    # Check links
    for line_no, url, _link_type in _extract_links(content):
        if not _is_internal_link(url):
            continue

        resolved = _resolve_link_target(file_path, url, root)
        if resolved is None:
            violations.append((line_no, RULE_LINK_BROKEN, f"Unresolvable link: {url}"))
            continue

        # Resolve to path relative to root for existence check
        try:
            rel_target = resolved.relative_to(root.resolve())
        except ValueError:
            rel_target = resolved

        # Check if target exists (file or directory with index)
        target_exists = False
        if resolved.exists():
            if resolved.is_file():
                target_exists = True
            elif resolved.is_dir():
                # Directory: check for index.md or README.md
                for name in ("README.md", "index.md"):
                    if (resolved / name).exists():
                        target_exists = True
                        break

        if not target_exists:
            violations.append(
                (
                    line_no,
                    RULE_LINK_404,
                    f"Link target does not exist: {rel_target}",
                )
            )
            continue

        # DOC_DEPRECATED_REF: new doc must not reference archive/deprecated
        if not source_in_archive and _is_archive_or_deprecated(resolved, root):
            violations.append(
                (
                    line_no,
                    RULE_DEPRECATED_REF,
                    f"Should not reference archived doc: {rel_target}",
                )
            )

    # DOC_STALE_MARKER (optional)
    for line_no, marker in _extract_stale_markers(content):
        violations.append((line_no, RULE_STALE_MARKER, f"Contains stale marker: {marker}"))

    # Apply whitelist
    filtered: list[tuple[int, str, str]] = []
    for line_no, rule_id, msg in violations:
        if _is_whitelisted(rel_str, line_no, rule_id, whitelist):
            continue
        filtered.append((line_no, rule_id, msg))

    return filtered


def collect_md_files(root: Path, dirs: list[str]) -> list[Path]:
    """Collect all .md files under given dirs."""
    paths: list[Path] = []
    for d in dirs:
        p = root / d
        if p.exists() and p.is_dir():
            for f in p.rglob("*.md"):
                paths.append(f)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Doc gardening: scan markdown for broken/deprecated refs"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Scan only this dir (for tests); overrides default scan dirs",
    )
    parser.add_argument(
        "--whitelist",
        type=Path,
        default=None,
        help="Whitelist file (default: scripts/doc-gardening-whitelist.txt)",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    whitelist_path = args.whitelist
    if whitelist_path is None:
        whitelist_path = root / "scripts" / DEFAULT_WHITELIST_NAME
    else:
        whitelist_path = whitelist_path.resolve()
    whitelist = _load_whitelist(whitelist_path)

    if args.path is not None:
        path = args.path.resolve()
        if not path.exists():
            print(f"Error: --path {path} does not exist", file=sys.stderr)
            return 2
        root = path  # Use fixture dir as root for path resolution
        md_files = sorted(path.rglob("*.md"))
    else:
        scan_dirs = [
            "docs/decision",
            "docs/cmd",
            "docs/guide",
            "docs/ops",
        ]
        md_files = collect_md_files(root, scan_dirs)
        hub = root / "docs" / "README.md"
        if hub.is_file():
            md_files = sorted({*md_files, hub})

    all_violations: list[tuple[Path, int, str, str]] = []
    for md_path in md_files:
        for line_no, rule_id, msg in check_file(md_path, root, whitelist):
            all_violations.append((md_path, line_no, rule_id, msg))

    for path, line_no, rule_id, msg in all_violations:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        rel_str = str(rel).replace("\\", "/")
        print(f"{rel_str}:{line_no}: {rule_id}: {msg}")

    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
