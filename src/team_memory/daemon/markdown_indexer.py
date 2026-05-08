"""Markdown indexer for Obsidian vaults.

Parses Markdown files into Experience-compatible data,
with frontmatter extraction, project resolution, and index filtering.
"""

from __future__ import annotations

import os
from pathlib import PurePath
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split *text* into (frontmatter_dict, body).

    Returns ({}, text) when no valid frontmatter block is found.
    """
    if not text.startswith("---"):
        return {}, text

    # Find the closing ---
    # The opening --- is at position 0, search for the next one after it.
    rest = text[3:]
    if rest.startswith("\n"):
        rest = rest[1:]
    elif rest.startswith("\r\n"):
        rest = rest[2:]
    else:
        # "---" not followed by newline – not valid frontmatter
        return {}, text

    end = rest.find("\n---")
    if end == -1:
        return {}, text

    fm_text = rest[:end]
    body = rest[end + 4:]  # skip past \n---
    if body.startswith("\n"):
        body = body[1:]
    elif body.startswith("\r\n"):
        body = body[2:]

    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}

    if not isinstance(fm, dict):
        fm = {}

    return fm, body


def _infer_tags(file_path: str) -> list[str]:
    """Infer tags from the directory names of *file_path*."""
    p = PurePath(file_path)
    # Use the parent directory names as tags (excluding root-ish segments)
    parts = p.parent.parts
    # Skip the vault root – just take the last meaningful directory
    tags: list[str] = []
    for part in reversed(parts):
        if part in ("/", ""):
            continue
        tags.append(part)
        if len(tags) >= 3:
            break
    return tags


def _match_pattern(rel_path: str, pattern: str) -> bool:
    """Match *rel_path* against a glob-style *pattern* with ``**`` support.

    ``**`` matches zero or more directory segments.
    ``*`` matches any characters within a single segment.
    ``?`` matches a single character within a segment.
    """
    import re as _re

    # Normalise separators to /
    rel = rel_path.replace(os.sep, "/")
    pat = pattern.replace(os.sep, "/")

    # Convert glob pattern to regex
    i = 0
    regex = ""
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if i + 1 < len(pat) and pat[i + 1] == "*":
                # **/  →  zero or more directory segments
                if i + 2 < len(pat) and pat[i + 2] == "/":
                    regex += "(?:.+/)?"
                    i += 3
                else:
                    # ** at end → match everything
                    regex += ".*"
                    i += 2
            else:
                regex += "[^/]*"
                i += 1
        elif c == "?":
            regex += "[^/]"
            i += 1
        else:
            regex += _re.escape(c)
            i += 1

    return bool(_re.match("^" + regex + "$", rel))


# ---------------------------------------------------------------------------
# MarkdownIndexer
# ---------------------------------------------------------------------------

class MarkdownIndexer:
    """Stateless helper that parses Markdown files and applies vault rules."""

    # -- public API ----------------------------------------------------------

    @staticmethod
    def parse_file(file_path: str) -> dict[str, Any]:
        """Parse a Markdown file and return Experience-compatible data.

        Returns dict with keys: title, description, solution, tags, file_path.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, IOError):
            text = ""

        fm, body = _parse_frontmatter(text)

        # title
        title = fm.get("title")
        if not title:
            title = PurePath(file_path).stem

        # tags
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        if not tags:
            tags = _infer_tags(file_path)

        # description: first 500 chars of body (stripped)
        description = body.strip()[:500]

        # solution: body text truncated to 4000 chars to stay within
        # qwen3-embedding:0.6b's context limit (Chinese ~1.5 char/token)
        solution = body.strip()[:4000]

        return {
            "title": title,
            "description": description,
            "solution": solution,
            "tags": tags,
            "file_path": file_path,
        }

    @staticmethod
    def resolve_project(file_path: str, config: dict[str, Any]) -> str:
        """Determine which vault/project *file_path* belongs to.

        Returns the ``project`` name, or ``"default"`` if no vault matches.
        """
        abs_path = os.path.abspath(file_path)
        for vault in config.get("vaults", []):
            vault_path = os.path.abspath(vault["path"])
            if abs_path.startswith(vault_path + os.sep) or abs_path == vault_path:
                return vault["project"]
        return "default"

    @staticmethod
    def should_index(file_path: str, config: dict[str, Any]) -> bool:
        """Decide whether *file_path* ought to be indexed.

        A file is indexable when:
        1. It ends with ``.md``.
        2. It matches at least one ``index_patterns`` of its vault.
        3. It does **not** match any ``exclude_patterns`` of its vault.
        """
        if not file_path.endswith(".md"):
            return False

        abs_path = os.path.abspath(file_path)

        for vault in config.get("vaults", []):
            vault_path = os.path.abspath(vault["path"])
            if not (abs_path.startswith(vault_path + os.sep) or abs_path == vault_path):
                continue

            try:
                rel = str(PurePath(abs_path).relative_to(vault_path))
            except ValueError:
                return False

            if not rel or rel == ".":
                # File *is* the vault root – skip
                return False

            # Check exclude first
            excluded = any(_match_pattern(rel, pat) for pat in vault.get("exclude_patterns", []))
            if excluded:
                return False

            # Must match at least one index pattern
            matched = any(_match_pattern(rel, pat) for pat in vault.get("index_patterns", []))
            return matched

        # Not under any vault → not indexable
        return False

    @staticmethod
    def load_config(config_path: str) -> dict[str, Any]:
        """Load a YAML configuration file and return the parsed dict."""
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
