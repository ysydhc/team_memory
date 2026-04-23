"""Tests for scripts/hooks/markdown_indexer.py — MarkdownIndexer."""

from __future__ import annotations

import os
import textwrap
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Import target
# ---------------------------------------------------------------------------
from scripts.hooks.markdown_indexer import MarkdownIndexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_config(tmp_path):
    """Return a config dict and also write a YAML file for load_config tests."""
    cfg = {
        "vaults": [
            {
                "path": "/Users/yeshouyou/Work/ad_learning",
                "project": "ad_learning",
                "index_patterns": ["docs/**/*.md", "ad_literacy/**/*.md", "technology/**/*.md"],
                "exclude_patterns": [".obsidian/**", ".git/**"],
            },
            {
                "path": "/Users/yeshouyou/Work/agent/ai_learning",
                "project": "ai_learning",
                "index_patterns": ["**/*.md"],
                "exclude_patterns": [".obsidian/**", ".git/**"],
            },
            {
                "path": "/Users/yeshouyou/Work/agent/team_doc",
                "project": "team_doc",
                "index_patterns": ["plans/**/*.md"],
                "exclude_patterns": [".obsidian/**", ".git/**"],
            },
        ]
    }
    return cfg


@pytest.fixture()
def config_file(tmp_path, sample_config):
    """Write sample_config as YAML and return the path."""
    import yaml
    p = tmp_path / "obsidian_config.yaml"
    p.write_text(yaml.dump(sample_config, allow_unicode=True), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# parse_file
# ---------------------------------------------------------------------------

class TestParseFile:
    """MarkdownIndexer.parse_file"""

    def test_with_frontmatter(self, tmp_path):
        md = tmp_path / "note.md"
        md.write_text(textwrap.dedent("""\
            ---
            title: My Note
            tags:
              - python
              - obsidian
            ---
            This is the body content.
        """), encoding="utf-8")

        result = MarkdownIndexer.parse_file(str(md))
        assert result["title"] == "My Note"
        assert result["tags"] == ["python", "obsidian"]
        assert result["description"] == "This is the body content."
        assert result["solution"] == "This is the body content."
        assert result["file_path"] == str(md)

    def test_without_frontmatter_uses_filename(self, tmp_path):
        md = tmp_path / "my_awesome_note.md"
        md.write_text("Just some text.", encoding="utf-8")

        result = MarkdownIndexer.parse_file(str(md))
        assert result["title"] == "my_awesome_note"
        assert result["description"] == "Just some text."
        assert result["tags"]  # inferred from directory

    def test_chinese_content(self, tmp_path):
        md = tmp_path / "中文笔记.md"
        md.write_text(textwrap.dedent("""\
            ---
            title: 学习笔记
            tags:
              - 学习
            ---
            这是中文正文内容，用于测试中文解析功能。
        """), encoding="utf-8")

        result = MarkdownIndexer.parse_file(str(md))
        assert result["title"] == "学习笔记"
        assert "学习" in result["tags"]
        assert "中文正文" in result["description"]

    def test_empty_file(self, tmp_path):
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")

        result = MarkdownIndexer.parse_file(str(md))
        assert result["title"] == "empty"
        assert result["description"] == ""
        assert result["solution"] == ""
        assert result["tags"]  # inferred from directory name

    def test_frontmatter_tags_as_string(self, tmp_path):
        md = tmp_path / "comma_tags.md"
        md.write_text(textwrap.dedent("""\
            ---
            title: Comma Tags
            tags: "python, rust, go"
            ---
            Body here.
        """), encoding="utf-8")

        result = MarkdownIndexer.parse_file(str(md))
        assert "python" in result["tags"]
        assert "rust" in result["tags"]
        assert "go" in result["tags"]

    def test_description_truncated_at_500(self, tmp_path):
        long_body = "A" * 600
        md = tmp_path / "long.md"
        md.write_text(long_body, encoding="utf-8")

        result = MarkdownIndexer.parse_file(str(md))
        assert len(result["description"]) == 500
        assert result["solution"] == long_body


# ---------------------------------------------------------------------------
# resolve_project
# ---------------------------------------------------------------------------

class TestResolveProject:
    """MarkdownIndexer.resolve_project"""

    def test_ad_learning(self, sample_config):
        path = "/Users/yeshouyou/Work/ad_learning/docs/something.md"
        assert MarkdownIndexer.resolve_project(path, sample_config) == "ad_learning"

    def test_ai_learning(self, sample_config):
        path = "/Users/yeshouyou/Work/agent/ai_learning/notes/llm.md"
        assert MarkdownIndexer.resolve_project(path, sample_config) == "ai_learning"

    def test_team_doc(self, sample_config):
        path = "/Users/yeshouyou/Work/agent/team_doc/plans/sprint.md"
        assert MarkdownIndexer.resolve_project(path, sample_config) == "team_doc"

    def test_unknown_path_returns_default(self, sample_config):
        path = "/tmp/unknown/path/note.md"
        assert MarkdownIndexer.resolve_project(path, sample_config) == "default"


# ---------------------------------------------------------------------------
# should_index
# ---------------------------------------------------------------------------

class TestShouldIndex:
    """MarkdownIndexer.should_index"""

    def test_matching_index_pattern(self, sample_config):
        path = "/Users/yeshouyou/Work/ad_learning/docs/guide.md"
        assert MarkdownIndexer.should_index(path, sample_config) is True

    def test_matching_index_pattern_ai_learning(self, sample_config):
        path = "/Users/yeshouyou/Work/agent/ai_learning/anything.md"
        assert MarkdownIndexer.should_index(path, sample_config) is True

    def test_matching_index_pattern_team_doc(self, sample_config):
        path = "/Users/yeshouyou/Work/agent/team_doc/plans/2024.md"
        assert MarkdownIndexer.should_index(path, sample_config) is True

    def test_exclude_patterns(self, sample_config):
        path = "/Users/yeshouyou/Work/ad_learning/.obsidian/app.json.md"
        assert MarkdownIndexer.should_index(path, sample_config) is False

    def test_git_excluded(self, sample_config):
        path = "/Users/yeshouyou/Work/agent/ai_learning/.git/COMMIT_EDITMSG.md"
        assert MarkdownIndexer.should_index(path, sample_config) is False

    def test_non_md_file(self, sample_config):
        path = "/Users/yeshouyou/Work/ad_learning/docs/guide.txt"
        assert MarkdownIndexer.should_index(path, sample_config) is False

    def test_no_matching_index_pattern(self, sample_config):
        # ad_learning only indexes docs/**, ad_literacy/**, technology/**
        # A file in root should not match
        path = "/Users/yeshouyou/Work/ad_learning/random.md"
        assert MarkdownIndexer.should_index(path, sample_config) is False

    def test_path_not_in_any_vault(self, sample_config):
        path = "/tmp/outside/note.md"
        assert MarkdownIndexer.should_index(path, sample_config) is False


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """MarkdownIndexer.load_config"""

    def test_load_config(self, config_file, sample_config):
        loaded = MarkdownIndexer.load_config(config_file)
        assert "vaults" in loaded
        assert len(loaded["vaults"]) == 3
        assert loaded["vaults"][0]["project"] == "ad_learning"
        assert loaded["vaults"][1]["project"] == "ai_learning"
        assert loaded["vaults"][2]["project"] == "team_doc"

    def test_load_config_index_patterns(self, config_file):
        loaded = MarkdownIndexer.load_config(config_file)
        assert "docs/**/*.md" in loaded["vaults"][0]["index_patterns"]
        assert "**/*.md" in loaded["vaults"][1]["index_patterns"]
        assert "plans/**/*.md" in loaded["vaults"][2]["index_patterns"]
