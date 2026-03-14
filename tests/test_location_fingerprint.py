"""Tests for location_fingerprint utils: normalize, fingerprint, find in lines, overlap."""

from __future__ import annotations

import pytest

from team_memory.utils.location_fingerprint import (
    LOCATION_SCORE_EXACT,
    LOCATION_SCORE_SAME_FILE,
    content_fingerprint,
    find_fingerprint_in_lines,
    lines_overlap,
    normalize_snippet_for_fingerprint,
    overlap_score,
)


class TestNormalizeSnippetForFingerprint:
    """normalize_snippet_for_fingerprint: strip, collapse whitespace; None/empty -> ''."""

    def test_none_returns_empty(self):
        assert normalize_snippet_for_fingerprint(None) == ""

    def test_empty_string_returns_empty(self):
        assert normalize_snippet_for_fingerprint("") == ""

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_snippet_for_fingerprint("  foo  ") == "foo"

    def test_collapses_internal_whitespace(self):
        assert normalize_snippet_for_fingerprint("a  \t  b\n c") == "a b c"

    def test_preserves_single_space_between_words(self):
        assert normalize_snippet_for_fingerprint("hello world") == "hello world"


class TestContentFingerprint:
    """content_fingerprint: normalize then hash; empty -> fixed constant."""

    def test_empty_string_returns_empty_constant(self):
        assert content_fingerprint("") == "empty"

    def test_none_normalized_returns_empty_constant(self):
        assert content_fingerprint(None) == "empty"

    def test_whitespace_only_normalizes_to_empty_constant(self):
        assert content_fingerprint("   \n\t  ") == "empty"

    def test_same_snippet_same_fingerprint(self):
        fp1 = content_fingerprint("def foo(): pass")
        fp2 = content_fingerprint("def foo(): pass")
        assert fp1 == fp2

    def test_different_snippet_different_fingerprint(self):
        fp1 = content_fingerprint("def foo(): pass")
        fp2 = content_fingerprint("def bar(): pass")
        assert fp1 != fp2

    def test_fingerprint_is_sha256_hex_like(self):
        fp = content_fingerprint("x")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


class TestFindFingerprintInLines:
    """find_fingerprint_in_lines: sliding window; None when empty/oversized/not found."""

    def test_empty_lines_returns_none(self):
        assert find_fingerprint_in_lines([], "any") is None

    def test_window_size_exceeds_lines_returns_none(self):
        lines = ["line1", "line2"]
        fp = content_fingerprint("line1\nline2")
        assert find_fingerprint_in_lines(lines, fp, window_size=20) is None

    def test_fingerprint_not_found_returns_none(self):
        lines = ["a", "b", "c"] * 10
        assert find_fingerprint_in_lines(lines, "nonexistent_fp", window_size=3) is None

    def test_found_returns_one_based_start_end(self):
        lines = ["first", "second", "third", "fourth"]
        snippet = normalize_snippet_for_fingerprint("first\nsecond")
        fp = content_fingerprint(snippet)
        result = find_fingerprint_in_lines(lines, fp, window_size=2)
        assert result == (1, 2)

    def test_window_size_boundary_exactly_twenty_lines(self):
        lines = [f"line{i}" for i in range(20)]
        snippet = "\n".join(lines)
        fp = content_fingerprint(normalize_snippet_for_fingerprint(snippet))
        result = find_fingerprint_in_lines(lines, fp, window_size=20)
        assert result == (1, 20)

    def test_window_size_21_with_20_lines_returns_none(self):
        lines = [f"line{i}" for i in range(20)]
        fp = content_fingerprint("x")
        assert find_fingerprint_in_lines(lines, fp, window_size=21) is None

    def test_default_window_size_is_20(self):
        lines = [f"line{i}" for i in range(20)]
        snippet = "\n".join(lines)
        fp = content_fingerprint(normalize_snippet_for_fingerprint(snippet))
        result = find_fingerprint_in_lines(lines, fp)
        assert result == (1, 20)


class TestLinesOverlap:
    """lines_overlap: true when ranges [a_start,a_end] and [b_start,b_end] overlap."""

    def test_overlap_true(self):
        assert lines_overlap(1, 10, 5, 15) is True
        assert lines_overlap(5, 15, 1, 10) is True

    def test_no_overlap_false(self):
        assert lines_overlap(1, 5, 10, 15) is False
        assert lines_overlap(10, 15, 1, 5) is False

    def test_adjacent_touching_overlap(self):
        assert lines_overlap(1, 5, 5, 10) is True

    def test_single_line_overlap(self):
        assert lines_overlap(3, 3, 3, 3) is True


class TestOverlapScore:
    """overlap_score: full containment 1.0, partial overlap 0.7, no overlap 0.0."""

    def test_full_containment_returns_exact(self):
        assert overlap_score(1, 20, 5, 10) == LOCATION_SCORE_EXACT
        assert overlap_score(5, 10, 1, 20) == LOCATION_SCORE_EXACT

    def test_partial_overlap_returns_same_file(self):
        assert overlap_score(1, 10, 5, 15) == LOCATION_SCORE_SAME_FILE
        assert overlap_score(5, 15, 1, 10) == LOCATION_SCORE_SAME_FILE

    def test_no_overlap_returns_zero(self):
        assert overlap_score(1, 5, 10, 15) == 0.0

    def test_exact_match_returns_exact(self):
        assert overlap_score(3, 7, 3, 7) == LOCATION_SCORE_EXACT


class TestConstants:
    """Exported constants for storage and pipeline."""

    def test_location_score_exact(self):
        assert LOCATION_SCORE_EXACT == 1.0

    def test_location_score_same_file(self):
        assert LOCATION_SCORE_SAME_FILE == 0.7
