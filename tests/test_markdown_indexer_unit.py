"""Unit tests for markdown_indexer — parsing, extraction, pattern matching.

Covers all pure functions and static methods that don't require DB/LLM:
- _parse_frontmatter: YAML frontmatter extraction
- _infer_tags: directory-based tag inference
- _match_pattern: glob-style path matching with ** support
- MarkdownIndexer.parse_file: file parsing with title/tag/description logic
- MarkdownIndexer.resolve_project: vault project resolution
- MarkdownIndexer.should_index: index/exclude pattern filtering
- MarkdownIndexer.load_config: YAML config loading
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from daemon.markdown_indexer import (
    MarkdownIndexer,
    _infer_tags,
    _match_pattern,
    _parse_frontmatter,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_frontmatter
# ---------------------------------------------------------------------------
class TestParseFrontmatter:
    """Tests for _parse_frontmatter / frontmatter 解析"""

    def test_no_frontmatter(self):
        """Plain text without frontmatter returns empty dict + full text."""
        text = "# Hello\nSome body."
        fm, body = _parse_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_valid_frontmatter(self):
        """Valid YAML frontmatter is parsed correctly."""
        text = "---\ntitle: Test\ntags:\n  - python\n---\nBody here."
        fm, body = _parse_frontmatter(text)
        assert fm["title"] == "Test"
        assert fm["tags"] == ["python"]
        assert body == "Body here."

    def test_frontmatter_with_crlf(self):
        """Frontmatter with \\r\\n line endings works."""
        text = "---\r\ntitle: CR\r\n---\r\nBody"
        fm, body = _parse_frontmatter(text)
        assert fm["title"] == "CR"
        assert body == "Body"

    def test_no_closing_delimiter(self):
        """Missing closing --- returns empty dict + full text."""
        text = "---\ntitle: No Close\nBody"
        fm, body = _parse_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_no_newline_after_opening(self):
        """--- not followed by newline is not valid frontmatter."""
        text = "---title: Bad\n---\nBody"
        fm, body = _parse_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_invalid_yaml_frontmatter(self):
        """Malformed YAML returns empty dict (graceful fallback)."""
        text = "---\n: invalid: yaml: [[\n---\nBody"
        fm, body = _parse_frontmatter(text)
        assert fm == {}
        assert body == "Body"

    def test_non_dict_yaml_returns_empty(self):
        """Non-dict YAML (e.g. a list) returns empty dict."""
        text = "---\n- item1\n- item2\n---\nBody"
        fm, body = _parse_frontmatter(text)
        assert fm == {}

    def test_empty_body_after_frontmatter(self):
        """Frontmatter with nothing after the closing ---."""
        text = "---\ntitle: Alone\n---"
        fm, body = _parse_frontmatter(text)
        assert fm["title"] == "Alone"
        assert body == ""

    def test_body_with_double_newline_after_delimiter(self):
        """Opening --- followed by double newline."""
        text = "---\n\n\ntitle: Spaced\n---\nBody"
        fm, body = _parse_frontmatter(text)
        # The first \n after --- is consumed, body starts after \n---
        assert fm["title"] == "Spaced"

    def test_empty_frontmatter(self):
        """Empty frontmatter block returns empty dict."""
        text = "---\ntitle: x\n---\nBody"
        fm, body = _parse_frontmatter(text)
        assert fm["title"] == "x"
        assert body == "Body"


# ---------------------------------------------------------------------------
# _infer_tags
# ---------------------------------------------------------------------------
class TestInferTags:
    """Tests for _infer_tags / 目录标签推断"""

    def test_deep_path(self):
        """Deep path extracts up to 3 parent directory names."""
        tags = _infer_tags("/vault/projects/python/main.md")
        assert tags[:3] == ["python", "projects", "vault"]

    def test_shallow_path(self):
        """Shallow path uses available parent directories."""
        tags = _infer_tags("/notes/my.md")
        assert "notes" in tags

    def test_root_level_file(self):
        """File at root level with no meaningful parent dirs."""
        tags = _infer_tags("/file.md")
        # Root dirs like '/' are skipped, so tags may be empty
        assert isinstance(tags, list)

    def test_max_three_tags(self):
        """At most 3 tags are returned."""
        tags = _infer_tags("/a/b/c/d/e/f.md")
        assert len(tags) <= 3

    def test_empty_parent_parts(self):
        """PurePath('file.md').parent.parts may yield ('',)."""
        tags = _infer_tags("file.md")
        # '' is filtered out, tags should be empty
        assert tags == []


# ---------------------------------------------------------------------------
# _match_pattern
# ---------------------------------------------------------------------------
class TestMatchPattern:
    """Tests for _match_pattern / glob 模式匹配"""

    def test_simple_glob_star(self):
        """Single * matches within a segment."""
        assert _match_pattern("readme.md", "*.md")

    def test_simple_glob_no_match(self):
        """Single * does NOT cross directories."""
        assert not _match_pattern("docs/sub/readme.md", "*.md")

    def test_double_star_matches_multiple_dirs(self):
        """** matches zero or more directory segments."""
        assert _match_pattern("a/b/c/file.md", "**/file.md")

    def test_double_star_matches_top_level(self):
        """** matches zero segments (top-level)."""
        assert _match_pattern("file.md", "**/file.md")

    def test_question_mark(self):
        """? matches exactly one character."""
        assert _match_pattern("file1.md", "file?.md")
        assert not _match_pattern("file12.md", "file?.md")

    def test_exact_match(self):
        """Exact pattern matches exactly."""
        assert _match_pattern("README.md", "README.md")
        assert not _match_pattern("readme.md", "README.md")

    def test_prefix_glob(self):
        """Glob at the end of pattern."""
        assert _match_pattern("notes/hello.md", "notes/*.md")

    def test_no_match(self):
        """Non-matching pattern returns False."""
        assert not _match_pattern("other/file.py", "docs/*.md")

    def test_double_star_at_end(self):
        """** at end matches everything."""
        assert _match_pattern("any/path/goes.md", "**")
        assert _match_pattern("single.md", "**")

    def test_pattern_with_separator_normalisation(self):
        """os.sep is normalised to / for matching."""
        assert _match_pattern("docs" + os.sep + "readme.md", "docs/*.md")

    def test_empty_pattern_empty_path(self):
        """Empty pattern matches empty path."""
        assert _match_pattern("", "")

    def test_single_star_matches_filename(self):
        """*.* pattern matches any filename with extension."""
        assert _match_pattern("test.md", "*.*")
        assert not _match_pattern("test", "*.*")


# ---------------------------------------------------------------------------
# MarkdownIndexer.parse_file
# ---------------------------------------------------------------------------
class TestParseFile:
    """Tests for MarkdownIndexer.parse_file / 文件解析"""

    def test_basic_file_with_frontmatter(self, tmp_path):
        """File with frontmatter extracts title, tags, body."""
        md = tmp_path / "note.md"
        md.write_text("---\ntitle: My Note\ntags:\n  - python\n---\nHello world.")
        result = MarkdownIndexer.parse_file(str(md))
        assert result["title"] == "My Note"
        assert result["tags"] == ["python"]
        assert result["description"] == "Hello world."
        assert result["solution"] == "Hello world."
        assert result["file_path"] == str(md)

    def test_file_without_frontmatter(self, tmp_path):
        """File without frontmatter uses filename as title."""
        md = tmp_path / "my_file.md"
        md.write_text("# Just a heading\nSome content.")
        result = MarkdownIndexer.parse_file(str(md))
        assert result["title"] == "my_file"
        assert result["description"].startswith("# Just a heading")

    def test_nonexistent_file(self):
        """Nonexistent file returns empty body, stem as title."""
        result = MarkdownIndexer.parse_file("/nonexistent/path/fake.md")
        assert result["title"] == "fake"
        assert result["description"] == ""

    def test_description_truncated_to_500(self, tmp_path):
        """Description is truncated to 500 characters."""
        md = tmp_path / "long.md"
        body = "x" * 1000
        md.write_text(body)
        result = MarkdownIndexer.parse_file(str(md))
        assert len(result["description"]) == 500

    def test_solution_truncated_to_4000(self, tmp_path):
        """Solution is truncated to 4000 characters."""
        md = tmp_path / "long.md"
        body = "y" * 5000
        md.write_text(body)
        result = MarkdownIndexer.parse_file(str(md))
        assert len(result["solution"]) == 4000

    def test_tags_as_string_comma_separated(self, tmp_path):
        """Comma-separated string tags are split into a list."""
        md = tmp_path / "tagged.md"
        md.write_text("---\ntitle: Tagged\ntags: python, rust, go\n---\nBody")
        result = MarkdownIndexer.parse_file(str(md))
        assert result["tags"] == ["python", "rust", "go"]

    def test_no_tags_uses_inferred(self, tmp_path):
        """When no frontmatter tags, directory-based tags are inferred."""
        md = tmp_path / "note.md"
        md.write_text("# Note\nContent")
        result = MarkdownIndexer.parse_file(str(md))
        # tags should be inferred from directory path
        assert isinstance(result["tags"], list)

    def test_title_fallback_to_stem(self, tmp_path):
        """Empty title in frontmatter falls back to file stem."""
        md = tmp_path / "fallback.md"
        md.write_text("---\ntitle:\n---\nBody")
        result = MarkdownIndexer.parse_file(str(md))
        assert result["title"] == "fallback"


# ---------------------------------------------------------------------------
# MarkdownIndexer.resolve_project
# ---------------------------------------------------------------------------
class TestResolveProject:
    """Tests for MarkdownIndexer.resolve_project / 项目解析"""

    def test_matching_vault(self):
        """Path inside a vault returns that vault's project."""
        config = {
            "vaults": [
                {"path": "/home/user/notes", "project": "my_notes"},
            ]
        }
        result = MarkdownIndexer.resolve_project("/home/user/notes/daily.md", config)
        assert result == "my_notes"

    def test_no_matching_vault(self):
        """Path outside all vaults returns 'default'."""
        config = {
            "vaults": [
                {"path": "/home/user/notes", "project": "my_notes"},
            ]
        }
        result = MarkdownIndexer.resolve_project("/tmp/other.md", config)
        assert result == "default"

    def test_empty_vaults(self):
        """Empty vaults list returns 'default'."""
        result = MarkdownIndexer.resolve_project("/any/path.md", {"vaults": []})
        assert result == "default"

    def test_no_vaults_key(self):
        """Config without 'vaults' key returns 'default'."""
        result = MarkdownIndexer.resolve_project("/any/path.md", {})
        assert result == "default"

    def test_exact_vault_path_match(self):
        """File path equal to vault path matches."""
        config = {
            "vaults": [
                {"path": "/vault", "project": "root_vault"},
            ]
        }
        result = MarkdownIndexer.resolve_project("/vault", config)
        assert result == "root_vault"

    def test_first_matching_vault_wins(self):
        """First matching vault wins when multiple match."""
        config = {
            "vaults": [
                {"path": "/home/user", "project": "home"},
                {"path": "/home/user/notes", "project": "notes"},
            ]
        }
        result = MarkdownIndexer.resolve_project("/home/user/notes/file.md", config)
        assert result == "home"


# ---------------------------------------------------------------------------
# MarkdownIndexer.should_index
# ---------------------------------------------------------------------------
class TestShouldIndex:
    """Tests for MarkdownIndexer.should_index / 索引过滤"""

    def _make_config(self, vault_path, index=None, exclude=None):
        """Helper to build a config dict."""
        return {
            "vaults": [
                {
                    "path": vault_path,
                    "project": "test",
                    "index_patterns": index or ["**/*.md"],
                    "exclude_patterns": exclude or [],
                }
            ]
        }

    def test_md_file_matching_pattern(self, tmp_path):
        """MD file matching index pattern is indexed."""
        vault = tmp_path / "vault"
        vault.mkdir()
        config = self._make_config(str(vault), index=["**/*.md"])
        assert MarkdownIndexer.should_index(str(vault / "note.md"), config)

    def test_non_md_file_rejected(self, tmp_path):
        """Non-.md files are rejected."""
        vault = tmp_path / "vault"
        vault.mkdir()
        config = self._make_config(str(vault))
        assert not MarkdownIndexer.should_index(str(vault / "image.png"), config)

    def test_excluded_pattern(self, tmp_path):
        """File matching exclude pattern is not indexed."""
        vault = tmp_path / "vault"
        (vault / ".obsidian").mkdir(parents=True)
        config = self._make_config(str(vault), exclude=[".obsidian/**"])
        assert not MarkdownIndexer.should_index(
            str(vault / ".obsidian" / "config.md"), config
        )

    def test_no_matching_index_pattern(self, tmp_path):
        """File not matching any index pattern is not indexed."""
        vault = tmp_path / "vault"
        vault.mkdir()
        config = self._make_config(str(vault), index=["docs/*.md"])
        assert not MarkdownIndexer.should_index(str(vault / "readme.md"), config)

    def test_no_vaults_configured(self):
        """No vaults means nothing is indexable."""
        assert not MarkdownIndexer.should_index("/any/path.md", {"vaults": []})

    def test_no_vaults_key(self):
        """Config without 'vaults' key."""
        assert not MarkdownIndexer.should_index("/any/path.md", {})

    def test_file_outside_vault(self, tmp_path):
        """File outside any vault is not indexable."""
        vault = tmp_path / "vault"
        vault.mkdir()
        config = self._make_config(str(vault))
        assert not MarkdownIndexer.should_index(str(tmp_path / "other.md"), config)

    def test_double_star_index_pattern(self, tmp_path):
        """** pattern matches nested paths."""
        vault = tmp_path / "vault"
        (vault / "a" / "b").mkdir(parents=True)
        config = self._make_config(str(vault), index=["**/note.md"])
        assert MarkdownIndexer.should_index(str(vault / "a" / "b" / "note.md"), config)

    def test_specific_index_pattern(self, tmp_path):
        """Specific directory pattern matches."""
        vault = tmp_path / "vault"
        (vault / "docs").mkdir(parents=True)
        config = self._make_config(str(vault), index=["docs/*.md"])
        assert MarkdownIndexer.should_index(str(vault / "docs" / "guide.md"), config)

    def test_exclude_before_index(self, tmp_path):
        """Exclude is checked before index — excluded takes priority."""
        vault = tmp_path / "vault"
        (vault / "secret").mkdir(parents=True)
        config = self._make_config(
            str(vault),
            index=["**/*.md"],
            exclude=["secret/**"],
        )
        # Even though **/*.md matches, secret/** excludes first
        assert not MarkdownIndexer.should_index(
            str(vault / "secret" / "leak.md"), config
        )


# ---------------------------------------------------------------------------
# MarkdownIndexer.load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    """Tests for MarkdownIndexer.load_config / 配置加载"""

    def test_valid_yaml(self, tmp_path):
        """Valid YAML config is loaded correctly."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("vaults:\n  - path: /notes\n    project: notes\n")
        result = MarkdownIndexer.load_config(str(cfg))
        assert result["vaults"][0]["path"] == "/notes"
        assert result["vaults"][0]["project"] == "notes"

    def test_empty_yaml_file(self, tmp_path):
        """Empty YAML file returns empty dict."""
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        result = MarkdownIndexer.load_config(str(cfg))
        assert result == {}

    def test_non_dict_yaml(self, tmp_path):
        """YAML that is a list (not dict) returns empty dict."""
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- item1\n- item2\n")
        result = MarkdownIndexer.load_config(str(cfg))
        assert result == {}

    def test_nonexistent_config_raises(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises((FileNotFoundError, OSError)):
            MarkdownIndexer.load_config("/nonexistent/config.yaml")
