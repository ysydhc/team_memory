"""Path safety for archive file uploads (local disk MVP)."""

from __future__ import annotations

import os
from pathlib import Path


def normalized_under_root(candidate: Path, root: Path) -> bool:
    """True if candidate resolves to a path under root (anti path-traversal)."""
    try:
        root_r = root.resolve()
        cand_r = candidate.resolve()
    except OSError:
        return False
    try:
        cand_r.relative_to(root_r)
        return True
    except ValueError:
        return False


def safe_suffix(filename: str | None, allowed: list[str] | None) -> str:
    """Return normalized suffix including dot, or empty string."""
    if not filename:
        return ""
    name = os.path.basename(filename)
    if ".." in name or "/" in name or "\\" in name:
        return ""
    suf = Path(name).suffix.lower()
    if not suf:
        return ""
    if allowed is not None and len(allowed) > 0 and suf not in {x.lower() for x in allowed}:
        raise ValueError(f"File extension not allowed: {suf}")
    return suf
