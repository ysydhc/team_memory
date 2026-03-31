"""Path helpers for archive multipart uploads (traversal + extension whitelist)."""

from __future__ import annotations

from pathlib import Path

import pytest

from team_memory.utils.archive_upload_paths import normalized_under_root, safe_suffix


def test_normalized_under_root_accepts_descendant(tmp_path: Path) -> None:
    root = tmp_path / "up"
    root.mkdir()
    inner = root / "subdir" / "b.txt"
    inner.parent.mkdir(parents=True)
    inner.write_text("x", encoding="utf-8")
    assert normalized_under_root(inner.resolve(), root.resolve()) is True


def test_normalized_under_root_rejects_outside(tmp_path: Path) -> None:
    root = tmp_path / "up"
    root.mkdir()
    outsider = tmp_path / "other" / "f.bin"
    outsider.parent.mkdir(parents=True)
    outsider.write_bytes(b"x")
    assert normalized_under_root(outsider.resolve(), root.resolve()) is False


def test_safe_suffix_whitelist() -> None:
    assert safe_suffix("doc.pdf", [".pdf", ".md"]) == ".pdf"
    with pytest.raises(ValueError, match="not allowed"):
        safe_suffix("x.exe", [".pdf"])


def test_safe_suffix_no_whitelist_when_allowed_is_none() -> None:
    assert safe_suffix("README", None) == ""
    assert safe_suffix("file.TXT", None) == ".txt"
    assert safe_suffix("blob.exe", None) == ".exe"


def test_safe_suffix_rejects_dotdot_or_separators_in_final_name() -> None:
    """basename 会去掉 a/b.md 中的目录；最终段里仍含 .. 或反斜杠的应拒绝。"""
    assert safe_suffix("weird..name.md", [".md"]) == ""
    assert safe_suffix("x\\y.md", [".md"]) == ""
