"""Unit tests for archive overview fallback and embedding text builder."""

from team_memory.services.archive import (
    EMBEDDING_CONTENT_BUDGET,
    _embedding_text_for_archive,
    derive_overview_fallback,
)


def test_derive_overview_empty() -> None:
    assert derive_overview_fallback("") == ""
    assert derive_overview_fallback("   ") == ""


def test_derive_overview_head() -> None:
    text = "hello world " * 100
    out = derive_overview_fallback(text, max_len=50)
    assert len(out) <= 50
    assert "hello" in out


def test_derive_overview_prefers_section_after_h2() -> None:
    doc = "# Title\n\nintro line\n\n## Section A\n\nbody one\n\n## Section B\n\nnext"
    out = derive_overview_fallback(doc, max_len=500)
    assert "Section A" in out or "body one" in out


# ── _embedding_text_for_archive ──────────────────────────────


def test_embedding_text_overview_only() -> None:
    """When overview fills the budget, solution_doc is not appended."""
    long_ov = "x" * (EMBEDDING_CONTENT_BUDGET + 100)
    text = _embedding_text_for_archive("T", overview=long_ov, solution_doc="SHOULD_NOT_APPEAR")
    assert "SHOULD_NOT_APPEAR" not in text
    assert "x" * EMBEDDING_CONTENT_BUDGET in text


def test_embedding_text_solution_fills_remaining() -> None:
    """Short overview + solution_doc together fill up to budget."""
    ov = "A" * 300
    sol = "B" * 2000
    text = _embedding_text_for_archive("T", overview=ov, solution_doc=sol)
    assert ov in text
    # remaining budget = 1000 - 300 = 700
    expected_sol_len = EMBEDDING_CONTENT_BUDGET - 300
    assert "B" * expected_sol_len in text
    assert "B" * (expected_sol_len + 1) not in text


def test_embedding_text_no_overview_uses_solution() -> None:
    """No overview at all → solution_doc[:budget] used."""
    sol = "S" * 2000
    text = _embedding_text_for_archive("T", solution_doc=sol)
    assert "S" * EMBEDDING_CONTENT_BUDGET in text
    assert "S" * (EMBEDDING_CONTENT_BUDGET + 1) not in text


def test_embedding_text_empty_both() -> None:
    """Neither overview nor solution_doc → only title and metadata."""
    text = _embedding_text_for_archive("T", tags=["a", "b"])
    assert "T" in text
    assert "tags: a, b" in text
