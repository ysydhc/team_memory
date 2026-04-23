"""Tests for scripts/hooks/git_watcher.py — GitWatcher."""

from __future__ import annotations

import os
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.hooks.git_watcher import GitWatcher
from scripts.hooks.markdown_indexer import MarkdownIndexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_config(tmp_path):
    """Return a config dict with a vault pointing at tmp_path."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
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
    return cfg, str(vault_dir)


@pytest.fixture()
def mock_tm_client():
    """Return a mock TMClient with async methods."""
    tm = MagicMock()
    tm.draft_save = AsyncMock(return_value={"id": "draft-123", "status": "draft"})
    tm.draft_publish = AsyncMock(return_value={"id": "draft-123", "status": "published"})
    tm.recall = AsyncMock(return_value={"results": []})
    return tm


@pytest.fixture()
def watcher(sample_config, mock_tm_client):
    cfg, vault_dir = sample_config
    return GitWatcher(config=cfg, indexer=MarkdownIndexer(), tm_client=mock_tm_client)


# ---------------------------------------------------------------------------
# on_git_add
# ---------------------------------------------------------------------------

class TestOnGitAdd:
    """GitWatcher.on_git_add"""

    @pytest.mark.asyncio
    async def test_md_file_calls_draft_save(self, watcher, mock_tm_client, tmp_path):
        """on_git_add should call draft_save for .md files that should_index."""
        vault_dir = tmp_path / "vault"
        md_file = vault_dir / "note.md"
        md_file.write_text("# Hello\nBody content here.", encoding="utf-8")

        results = await watcher.on_git_add(str(vault_dir), [str(md_file)])

        assert len(results) == 1
        mock_tm_client.draft_save.assert_called_once()
        call_kwargs = mock_tm_client.draft_save.call_args
        assert call_kwargs.kwargs["title"] == "note"
        assert call_kwargs.kwargs["group_key"] == str(md_file)
        assert call_kwargs.kwargs["project"] == "test_project"

    @pytest.mark.asyncio
    async def test_skips_non_md_file(self, watcher, mock_tm_client, tmp_path):
        """on_git_add should skip non-.md files."""
        vault_dir = tmp_path / "vault"
        py_file = vault_dir / "script.py"
        py_file.write_text("print('hi')", encoding="utf-8")

        results = await watcher.on_git_add(str(vault_dir), [str(py_file)])

        assert len(results) == 0
        mock_tm_client.draft_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_should_index_false(self, watcher, mock_tm_client, tmp_path):
        """on_git_add should skip files where should_index returns False."""
        vault_dir = tmp_path / "vault"
        # File in .obsidian dir should be excluded
        obsidian_dir = vault_dir / ".obsidian"
        obsidian_dir.mkdir()
        md_file = obsidian_dir / "config.md"
        md_file.write_text("# Config", encoding="utf-8")

        results = await watcher.on_git_add(str(vault_dir), [str(md_file)])

        assert len(results) == 0
        mock_tm_client.draft_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_files(self, watcher, mock_tm_client, tmp_path):
        """on_git_add should process multiple files."""
        vault_dir = tmp_path / "vault"
        f1 = vault_dir / "a.md"
        f2 = vault_dir / "b.md"
        f1.write_text("# A", encoding="utf-8")
        f2.write_text("# B", encoding="utf-8")

        results = await watcher.on_git_add(str(vault_dir), [str(f1), str(f2)])

        assert len(results) == 2
        assert mock_tm_client.draft_save.call_count == 2


# ---------------------------------------------------------------------------
# on_git_commit
# ---------------------------------------------------------------------------

class TestOnGitCommit:
    """GitWatcher.on_git_commit"""

    @pytest.mark.asyncio
    async def test_with_draft_calls_publish(self, watcher, mock_tm_client, tmp_path):
        """on_git_commit should publish drafts found via recall for committed files."""
        vault_dir = tmp_path / "vault"
        md_file = vault_dir / "note.md"
        md_file.write_text("# Updated content", encoding="utf-8")

        # recall finds a draft for this file
        mock_tm_client.recall = AsyncMock(return_value={
            "results": [{"id": "draft-456", "group_key": str(md_file), "status": "draft"}]
        })

        results = await watcher.on_git_commit(str(vault_dir), committed_files=[str(md_file)])

        mock_tm_client.draft_publish.assert_called_once()
        call_kwargs = mock_tm_client.draft_publish.call_args
        assert call_kwargs.kwargs["draft_id"] == "draft-456"

    @pytest.mark.asyncio
    async def test_no_draft_creates_published(self, watcher, mock_tm_client, tmp_path):
        """on_git_commit should create published directly when no draft exists."""
        vault_dir = tmp_path / "vault"
        md_file = vault_dir / "note.md"
        md_file.write_text("# New file", encoding="utf-8")

        # recall finds no draft for this file
        mock_tm_client.recall = AsyncMock(return_value={"results": []})

        results = await watcher.on_git_commit(str(vault_dir), committed_files=[str(md_file)])

        mock_tm_client.draft_publish.assert_not_called()
        mock_tm_client.draft_save.assert_called_once()
        call_kwargs = mock_tm_client.draft_save.call_args
        assert call_kwargs.kwargs["title"] == "note"
        assert call_kwargs.kwargs["project"] == "test_project"

    @pytest.mark.asyncio
    async def test_skips_non_md_files(self, watcher, mock_tm_client, tmp_path):
        """on_git_commit should skip non-.md files."""
        vault_dir = tmp_path / "vault"
        py_file = vault_dir / "script.py"
        py_file.write_text("print('hi')", encoding="utf-8")

        results = await watcher.on_git_commit(str(vault_dir), committed_files=[str(py_file)])

        assert len(results) == 0
        mock_tm_client.recall.assert_not_called()
        mock_tm_client.draft_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_should_index_false(self, watcher, mock_tm_client, tmp_path):
        """on_git_commit should skip files that should_index=False."""
        vault_dir = tmp_path / "vault"
        obsidian_dir = vault_dir / ".obsidian"
        obsidian_dir.mkdir()
        md_file = obsidian_dir / "config.md"
        md_file.write_text("# Config", encoding="utf-8")

        results = await watcher.on_git_commit(str(vault_dir), committed_files=[str(md_file)])

        assert len(results) == 0
        mock_tm_client.recall.assert_not_called()


# ---------------------------------------------------------------------------
# on_git_rm
# ---------------------------------------------------------------------------

class TestOnGitRm:
    """GitWatcher.on_git_rm"""

    @pytest.mark.asyncio
    async def test_returns_soft_delete_hint(self, watcher, mock_tm_client, tmp_path):
        """on_git_rm should return soft-delete hints since TM has no delete API."""
        vault_dir = tmp_path / "vault"
        md_file = vault_dir / "note.md"

        # recall finds an experience for this file
        mock_tm_client.recall = AsyncMock(return_value={
            "results": [{"id": "exp-789", "group_key": str(md_file), "status": "published"}]
        })

        results = await watcher.on_git_rm(str(vault_dir), [str(md_file)])

        assert len(results) == 1
        assert results[0]["action"] == "soft_delete"
        assert results[0]["id"] == "exp-789"
        assert "no delete API" in results[0]["message"]

    @pytest.mark.asyncio
    async def test_skips_non_md(self, watcher, mock_tm_client, tmp_path):
        """on_git_rm should skip non-.md files."""
        vault_dir = tmp_path / "vault"
        py_file = vault_dir / "script.py"

        results = await watcher.on_git_rm(str(vault_dir), [str(py_file)])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_no_experience_found(self, watcher, mock_tm_client, tmp_path):
        """on_git_rm should handle the case where no experience is found."""
        vault_dir = tmp_path / "vault"
        md_file = vault_dir / "note.md"

        # recall finds nothing
        mock_tm_client.recall = AsyncMock(return_value={"results": []})

        results = await watcher.on_git_rm(str(vault_dir), [str(md_file)])

        assert len(results) == 1
        assert results[0]["action"] == "not_found"


# ---------------------------------------------------------------------------
# get_staged_files
# ---------------------------------------------------------------------------

class TestGetStagedFiles:
    """GitWatcher.get_staged_files"""

    def test_parses_git_diff_output(self, watcher):
        """get_staged_files should parse git diff --name-only --cached output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="plans/sprint.md\nplans/retro.md\nsrc/main.py\n",
                returncode=0,
            )
            result = watcher.get_staged_files("/fake/repo")

        assert "plans/sprint.md" in result
        assert "plans/retro.md" in result
        # .py file should be filtered out
        assert "src/main.py" not in result

    def test_empty_staging(self, watcher):
        """get_staged_files should return empty list when nothing staged."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            result = watcher.get_staged_files("/fake/repo")

        assert result == []


# ---------------------------------------------------------------------------
# get_committed_files
# ---------------------------------------------------------------------------

class TestGetCommittedFiles:
    """GitWatcher.get_committed_files"""

    def test_parses_git_log_output(self, watcher):
        """get_committed_files should parse git log --name-only output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="\nplans/sprint.md\nplans/retro.md\nsrc/main.py\n",
                returncode=0,
            )
            result = watcher.get_committed_files("/fake/repo")

        assert "plans/sprint.md" in result
        assert "plans/retro.md" in result
        assert "src/main.py" not in result

    def test_empty_commit(self, watcher):
        """get_committed_files should return empty for empty commit."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            result = watcher.get_committed_files("/fake/repo")

        assert result == []


# ---------------------------------------------------------------------------
# on_git_mv
# ---------------------------------------------------------------------------

class TestOnGitMv:
    """GitWatcher.on_git_mv"""

    @pytest.mark.asyncio
    async def test_rm_old_and_add_new(self, watcher, mock_tm_client, tmp_path):
        """on_git_mv should rm old path and add new path."""
        vault_dir = tmp_path / "vault"
        new_file = vault_dir / "new_note.md"
        new_file.write_text("# New location", encoding="utf-8")

        # recall finds experience for old path
        mock_tm_client.recall = AsyncMock(return_value={
            "results": [{"id": "exp-old", "group_key": "/old/path.md", "status": "published"}]
        })

        result = await watcher.on_git_mv(str(vault_dir), "/old/path.md", str(new_file))

        assert result["rm"]["action"] == "soft_delete"
        assert result["add"] is not None
        mock_tm_client.draft_save.assert_called_once()
