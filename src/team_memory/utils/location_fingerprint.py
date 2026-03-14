"""Location fingerprint and line-overlap helpers for experience file bindings.

Normalization and fingerprint algorithm, and window_size default (20), are
the single source of truth for the file-location-binding plan; storage and
pipeline call these utils only.
"""

from __future__ import annotations

import hashlib

LOCATION_SCORE_EXACT = 1.0
LOCATION_SCORE_SAME_FILE = 0.7

# Default window size for sliding-window fingerprint search (plan-wide unique default).
DEFAULT_WINDOW_SIZE = 20


def normalize_snippet_for_fingerprint(text: str | None) -> str:
    """Strip and collapse whitespace; return '' for None or empty."""
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""
    return " ".join(s.split())


def content_fingerprint(snippet: str | None) -> str:
    """Normalize snippet then hash (sha256 hex); empty normalized -> 'empty'."""
    normalized = normalize_snippet_for_fingerprint(snippet)
    if not normalized:
        return "empty"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def find_fingerprint_in_lines(
    lines: list[str],
    fingerprint: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[int, int] | None:
    """Find fingerprint in lines via sliding window of window_size lines.

    Each window is normalized and hashed; first matching window returns
    (start_line, end_line) 1-based inclusive. Returns None if lines empty,
    window_size > len(lines), or fingerprint not found.
    """
    if not lines or len(lines) < window_size:
        return None
    for i in range(0, len(lines) - window_size + 1):
        window = lines[i : i + window_size]
        window_text = "\n".join(window)
        fp = content_fingerprint(window_text)
        if fp == fingerprint:
            return (i + 1, i + window_size)
    return None


def lines_overlap(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> bool:
    """True if ranges [a_start, a_end] and [b_start, b_end] overlap (inclusive)."""
    return not (a_end < b_start or b_end < a_start)


def overlap_score(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> float:
    """Score for two line ranges: full containment 1.0, partial overlap 0.7, else 0.0."""
    if not lines_overlap(a_start, a_end, b_start, b_end):
        return 0.0
    a_contains_b = a_start <= b_start and b_end <= a_end
    b_contains_a = b_start <= a_start and a_end <= b_end
    if a_contains_b or b_contains_a:
        return LOCATION_SCORE_EXACT
    return LOCATION_SCORE_SAME_FILE
