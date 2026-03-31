"""Unit tests for archive overview fallback."""

from team_memory.services.archive import derive_overview_fallback


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
