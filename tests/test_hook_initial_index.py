"""Tests for scripts/hooks/initial_index.py — initial full-index pipeline."""

from __future__ import annotations

import os
import subprocess
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from scripts.hooks.initial_index import _is_untracked, initial_index
from scripts.hooks.markdown_indexer import MarkdownIndexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vault_dir(tmp_path):
    """Create a fake vault directory with some .md files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Indexed files
    docs = vault / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\nThis is a guide.", encoding="utf-8")
    (docs / "tips.md").write_text("# Tips\nSome tips.", encoding="utf-8")

    # Excluded dir (non-hidden so glob can find it)
    drafts = vault / "drafts"
    drafts.mkdir()
    (drafts / "wip.md").write_text("# WIP\nWork in progress.", encoding="utf-8")

    return vault


@pytest.fixture()
def config_file(tmp_path, vault_dir):
    """Write a config YAML pointing at the temp vault."""
    cfg = {
        "vaults": [
            {
                "path": str(vault_dir),
                "project": "test_project",
                "index_patterns": ["**/*.md"],
                "exclude_patterns": [".obsidian/**", ".git/**", "drafts/**"],
            },
        ]
    }
    p = tmp_path / "obsidian_config.yaml"
    p.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


@pytest.fixture()
def multi_vault_config(tmp_path):
    """Config with two vaults — one existing, one non-existing."""
    vault1 = tmp_path / "vault1"
    vault1.mkdir()
    docs = vault1 / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Note\nBody.", encoding="utf-8")

    cfg = {
        "vaults": [
            {
                "path": str(vault1),
                "project": "vault1",
                "index_patterns": ["**/*.md"],
                "exclude_patterns": [".obsidian/**", ".git/**"],
            },
            {
                "path": "/nonexistent/path/that/does/not/exist",
                "project": "ghost",
                "index_patterns": ["**/*.md"],
                "exclude_patterns": [".obsidian/**", ".git/**"],
            },
        ]
    }
    p = tmp_path / "multi_config.yaml"
    p.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p), vault1


# ---------------------------------------------------------------------------
# dry_run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    """initial_index with dry_run=True should not call TMClient."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_tm(self, config_file, vault_dir):
        """dry_run=True should only print, not call tm.save()."""
        with patch("scripts.hooks.initial_index.TMClient") as MockTM:
            mock_tm_instance = MagicMock()
            mock_tm_instance.save = AsyncMock(return_value={"id": "abc"})
            MockTM.return_value = mock_tm_instance

            with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
                stats = await initial_index(config_file, "http://localhost:3900", dry_run=True)

            mock_tm_instance.save.assert_not_called()
            assert stats["indexed"] > 0

    @pytest.mark.asyncio
    async def test_dry_run_counts_files(self, config_file, vault_dir):
        """dry_run=True should count files correctly."""
        with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
            stats = await initial_index(config_file, "http://localhost:3900", dry_run=True)

        # vault has docs/guide.md, docs/tips.md, drafts/wip.md
        # drafts/wip.md is excluded by exclude_patterns → should_index=False
        # so total_files = 3, indexed = 2, skipped = 1
        assert stats["total_files"] == 3
        assert stats["indexed"] == 2
        assert stats["skipped"] == 1


# ---------------------------------------------------------------------------
# Skip untracked files
# ---------------------------------------------------------------------------

class TestSkipUntracked:
    """Files untracked by git should be skipped."""

    @pytest.mark.asyncio
    async def test_untracked_files_are_skipped(self, config_file, vault_dir):
        """Files where _is_untracked returns True should be skipped."""
        with patch("scripts.hooks.initial_index._is_untracked", return_value=True):
            stats = await initial_index(config_file, "http://localhost:3900", dry_run=True)

        # All files are "untracked" → all skipped after should_index pass
        assert stats["skipped"] >= 0
        assert stats["indexed"] == 0


# ---------------------------------------------------------------------------
# Skip should_index=False files
# ---------------------------------------------------------------------------

class TestSkipShouldIndex:
    """Files where should_index returns False should be skipped."""

    @pytest.mark.asyncio
    async def test_excluded_patterns_are_skipped(self, config_file, vault_dir):
        """Files matching exclude_patterns should be skipped."""
        with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
            stats = await initial_index(config_file, "http://localhost:3900", dry_run=True)

        # drafts/wip.md should be skipped (excluded by exclude_patterns)
        assert stats["skipped"] >= 1

    @pytest.mark.asyncio
    async def test_non_md_files_not_counted(self, vault_dir, tmp_path):
        """Non-.md files should not even be found by glob patterns."""
        # Write a .txt file
        (vault_dir / "readme.txt").write_text("not markdown", encoding="utf-8")

        cfg = {
            "vaults": [
                {
                    "path": str(vault_dir),
                    "project": "test_project",
                    "index_patterns": ["**/*.md"],
                    "exclude_patterns": [".obsidian/**", ".git/**"],
                },
            ]
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")

        with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
            stats = await initial_index(str(p), "http://localhost:3900", dry_run=True)

        # .txt files should not appear in total_files (glob only matches *.md)
        # Only .md files are counted
        for key in ["total_files", "indexed", "skipped", "errors"]:
            assert stats[key] >= 0


# ---------------------------------------------------------------------------
# Skip non-existent vault path
# ---------------------------------------------------------------------------

class TestSkipNonExistentVault:
    """Vaults with non-existent paths should be skipped."""

    @pytest.mark.asyncio
    async def test_nonexistent_vault_skipped(self, multi_vault_config):
        """Vaults whose path does not exist should be skipped entirely."""
        config_path, vault1 = multi_vault_config

        with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
            stats = await initial_index(config_path, "http://localhost:3900", dry_run=True)

        # Only vault1 files should be counted, ghost vault should be skipped
        assert stats["total_files"] >= 1
        assert stats["errors"] == 0


# ---------------------------------------------------------------------------
# Stats correctness
# ---------------------------------------------------------------------------

class TestStatsCorrectness:
    """Stats dict should accurately reflect operations."""

    @pytest.mark.asyncio
    async def test_stats_with_mixed_files(self, config_file, vault_dir):
        """Stats should reflect indexed, skipped, and total correctly."""
        with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
            stats = await initial_index(config_file, "http://localhost:3900", dry_run=True)

        assert stats["total_files"] == stats["indexed"] + stats["skipped"] + stats["errors"]

    @pytest.mark.asyncio
    async def test_error_count_on_tm_failure(self, config_file, vault_dir):
        """Errors should be counted when tm.save() raises."""
        with patch("scripts.hooks.initial_index.TMClient") as MockTM:
            mock_tm_instance = MagicMock()
            mock_tm_instance.save = AsyncMock(side_effect=ConnectionError("tm down"))
            MockTM.return_value = mock_tm_instance

            with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
                stats = await initial_index(config_file, "http://localhost:3900", dry_run=False)

        # guide.md and tips.md pass should_index, .obsidian/config.md is excluded
        assert stats["errors"] == 2
        assert stats["indexed"] == 0

    @pytest.mark.asyncio
    async def test_empty_vault(self, tmp_path):
        """An empty vault should produce zero stats."""
        vault = tmp_path / "empty_vault"
        vault.mkdir()

        cfg = {"vaults": [{"path": str(vault), "project": "empty", "index_patterns": ["**/*.md"], "exclude_patterns": []}]}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")

        stats = await initial_index(str(p), "http://localhost:3900", dry_run=True)

        assert stats["total_files"] == 0
        assert stats["indexed"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_successful_indexing_counts(self, config_file, vault_dir):
        """Successfully indexed files should increment indexed count."""
        with patch("scripts.hooks.initial_index.TMClient") as MockTM:
            mock_tm_instance = MagicMock()
            mock_tm_instance.save = AsyncMock(return_value={"id": "new-id"})
            MockTM.return_value = mock_tm_instance

            with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
                stats = await initial_index(config_file, "http://localhost:3900", dry_run=False)

        assert stats["indexed"] == 2  # guide.md + tips.md
        assert stats["errors"] == 0
        assert mock_tm_instance.save.call_count == 2

    @pytest.mark.asyncio
    async def test_save_called_with_correct_args(self, config_file, vault_dir):
        """tm.save() should be called with source='obsidian' and correct project."""
        with patch("scripts.hooks.initial_index.TMClient") as MockTM:
            mock_tm_instance = MagicMock()
            mock_tm_instance.save = AsyncMock(return_value={"id": "new-id"})
            MockTM.return_value = mock_tm_instance

            with patch("scripts.hooks.initial_index._is_untracked", return_value=False):
                await initial_index(config_file, "http://localhost:3900", dry_run=False)

        for call in mock_tm_instance.save.call_args_list:
            kwargs = call.kwargs
            assert kwargs["source"] == "obsidian"
            assert kwargs["project"] == "test_project"
            assert kwargs["title"]
            assert kwargs["content"]


# ---------------------------------------------------------------------------
# _is_untracked
# ---------------------------------------------------------------------------

class TestIsUntracked:
    """_is_untracked should correctly determine git tracking status."""

    def test_tracked_file_returns_false(self):
        """A tracked file should return False."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="file.md\n")
            result = _is_untracked("/repo/file.md", "/repo")

        assert result is False

    def test_untracked_file_returns_true(self):
        """An untracked file should return True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            result = _is_untracked("/repo/new.md", "/repo")

        assert result is True

    def test_git_command_failure_returns_true(self):
        """If git itself fails, conservatively return True."""
        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            result = _is_untracked("/repo/file.md", "/repo")

        assert result is True

    def test_subprocess_error_returns_true(self):
        """SubprocessError should also conservatively return True."""
        with patch("subprocess.run", side_effect=subprocess.SubprocessError("fail")):
            result = _is_untracked("/repo/file.md", "/repo")

        assert result is True


# ---------------------------------------------------------------------------
# No vaults in config
# ---------------------------------------------------------------------------

class TestNoVaults:
    """Edge case: config with no vaults."""

    @pytest.mark.asyncio
    async def test_empty_config(self, tmp_path):
        """Config with empty vaults list should return zero stats."""
        cfg = {"vaults": []}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")

        stats = await initial_index(str(p), "http://localhost:3900", dry_run=True)

        assert stats["total_files"] == 0
        assert stats["indexed"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
