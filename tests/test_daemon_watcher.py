"""Tests for daemon Obsidian file watcher."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from daemon.config import DaemonConfig, ObsidianSettings, VaultConfig
from daemon.watcher import _find_vault, _process_changes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(vault_path: str = "/tmp/test_vault") -> DaemonConfig:
    return DaemonConfig(
        obsidian=ObsidianSettings(
            vaults=[
                VaultConfig(
                    path=vault_path,
                    project="knowledge",
                    exclude=[".obsidian", ".trash"],
                ),
            ],
        ),
    )


def _make_sink() -> AsyncMock:
    sink = AsyncMock()
    sink.draft_save = AsyncMock(return_value={"id": "exp-1"})
    return sink


def _make_indexer(entry: dict | None = None) -> MagicMock:
    indexer = MagicMock()
    indexer.parse_file = MagicMock(return_value=entry or {
        "title": "Test Note", "description": "test desc", "solution": "test content", "tags": ["test"],
    })
    return indexer


# ---------------------------------------------------------------------------
# _find_vault
# ---------------------------------------------------------------------------

class TestFindVault:
    def test_file_in_vault(self, tmp_path):
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        config = DaemonConfig(
            obsidian=ObsidianSettings(
                vaults=[VaultConfig(path=str(vault_dir), project="test")],
            ),
        )
        file_path = vault_dir / "notes" / "test.md"
        result = _find_vault(file_path, config)
        assert result is not None
        assert result.project == "test"

    def test_file_outside_vault(self, tmp_path):
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        config = DaemonConfig(
            obsidian=ObsidianSettings(
                vaults=[VaultConfig(path=str(vault_dir), project="test")],
            ),
        )
        file_path = other_dir / "test.md"
        result = _find_vault(file_path, config)
        assert result is None

    def test_no_vaults_configured(self):
        config = DaemonConfig(obsidian=ObsidianSettings(vaults=[]))
        result = _find_vault(Path("/any/path.md"), config)
        assert result is None


# ---------------------------------------------------------------------------
# _process_changes
# ---------------------------------------------------------------------------

class TestProcessChanges:
    @pytest.mark.asyncio
    async def test_added_md_file_gets_indexed(self, tmp_path):
        from watchfiles import Change

        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        md_file = vault_dir / "test.md"
        md_file.write_text("# Test Note\ntest content")

        config = _make_config(str(vault_dir))
        sink = _make_sink()
        indexer = _make_indexer()

        changes = {(Change.added, str(md_file))}
        await _process_changes(changes, config, indexer, sink)

        indexer.parse_file.assert_called_once()
        sink.draft_save.assert_called_once_with(
            title="Test Note",
            content=ANY,
            tags=["test"],
            project="knowledge",
            group_key=None,
        )

    @pytest.mark.asyncio
    async def test_modified_md_file_gets_indexed(self, tmp_path):
        from watchfiles import Change

        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        md_file = vault_dir / "test.md"
        md_file.write_text("# Updated Note\nnew content")

        config = _make_config(str(vault_dir))
        sink = _make_sink()
        indexer = _make_indexer()

        changes = {(Change.modified, str(md_file))}
        await _process_changes(changes, config, indexer, sink)

        sink.draft_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_md_file_ignored(self, tmp_path):
        from watchfiles import Change

        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        py_file = vault_dir / "test.py"

        config = _make_config(str(vault_dir))
        sink = _make_sink()
        indexer = _make_indexer()

        changes = {(Change.added, str(py_file))}
        await _process_changes(changes, config, indexer, sink)

        sink.draft_save.assert_not_called()
        indexer.parse_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_excluded_dir_ignored(self, tmp_path):
        from watchfiles import Change

        vault_dir = tmp_path / "vault"
        obsidian_dir = vault_dir / ".obsidian"
        obsidian_dir.mkdir(parents=True)
        md_file = obsidian_dir / "config.md"
        md_file.write_text("config")

        config = _make_config(str(vault_dir))
        sink = _make_sink()
        indexer = _make_indexer()

        changes = {(Change.added, str(md_file))}
        await _process_changes(changes, config, indexer, sink)

        sink.draft_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_deleted_md_logged_no_sink_call(self, tmp_path):
        from watchfiles import Change

        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        md_file = vault_dir / "deleted.md"

        config = _make_config(str(vault_dir))
        sink = _make_sink()
        indexer = _make_indexer()

        changes = {(Change.deleted, str(md_file))}
        await _process_changes(changes, config, indexer, sink)

        sink.draft_save.assert_not_called()
        indexer.parse_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_outside_vault_ignored(self, tmp_path):
        from watchfiles import Change

        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        md_file = other_dir / "test.md"

        config = _make_config(str(vault_dir))
        sink = _make_sink()
        indexer = _make_indexer()

        changes = {(Change.added, str(md_file))}
        await _process_changes(changes, config, indexer, sink)

        sink.draft_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_entry_from_one_file(self, tmp_path):
        from watchfiles import Change

        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        md_file = vault_dir / "single.md"

        config = _make_config(str(vault_dir))
        sink = _make_sink()
        indexer = _make_indexer(entry={
            "title": "Single Entry", "description": "desc", "solution": "body text", "tags": ["x"],
        })

        changes = {(Change.added, str(md_file))}
        await _process_changes(changes, config, indexer, sink)

        assert sink.draft_save.call_count == 1
