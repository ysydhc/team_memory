"""Unit tests for wiki_cli.py — Wiki Compile CLI.

Covers:
  - _default_wiki_root()       pure path logic
  - main()                     argparse dispatch for compile/status/uncompiled
  - _fetch_published_experiences()  async DB fetch (mocked)
  - cmd_compile / cmd_status / cmd_uncompiled  async commands (mocked DB + compiler)

Marked:
  @pytest.mark.unit — no real DB, pure logic / mock-only
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from team_memory.daemon.wiki_cli import (
    _default_wiki_root,
    _fetch_published_experiences,
    cmd_compile,
    cmd_status,
    cmd_uncompiled,
    main,
)

pytestmark = pytest.mark.unit


# ===========================================================================
# _default_wiki_root — pure path logic
# ===========================================================================

class TestDefaultWikiRoot:
    """_default_wiki_root() returns <project_root>/wiki."""

    def test_returns_string(self):
        result = _default_wiki_root()
        assert isinstance(result, str)

    def test_ends_with_wiki(self):
        result = _default_wiki_root()
        assert result.endswith("wiki")

    def test_contains_project_root(self):
        result = _default_wiki_root()
        from team_memory.project_resolver import find_project_root
        project_root = find_project_root()
        assert result == os.path.join(project_root, "wiki")

    def test_is_absolute_path(self):
        result = _default_wiki_root()
        assert os.path.isabs(result)


# ===========================================================================
# main() — argparse dispatch
# ===========================================================================

class TestMainArgparse:
    """Test main() argument parsing and dispatch."""

    def _run_main(self, argv: list[str]):
        """Helper: run main() with mocked sys.argv and mocked asyncio.run."""
        with patch.object(sys, "argv", ["wiki_cli.py"] + argv):
            with patch("team_memory.daemon.wiki_cli.asyncio.run") as mock_run:
                main()
                return mock_run

    def test_no_command_prints_help(self):
        """No subcommand should call parser.print_help."""
        with patch.object(sys, "argv", ["wiki_cli.py"]):
            with patch("team_memory.daemon.wiki_cli.argparse.ArgumentParser.print_help") as mock_help:
                main()
                mock_help.assert_called_once()

    def test_compile_command(self):
        mock_run = self._run_main(["compile"])
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args is not None

    def test_compile_full_flag(self):
        mock_run = self._run_main(["compile", "--full"])
        mock_run.assert_called_once()

    def test_compile_with_wiki_root(self):
        mock_run = self._run_main(["compile", "--wiki-root", "/tmp/testwiki"])
        mock_run.assert_called_once()

    def test_status_command(self):
        mock_run = self._run_main(["status"])
        mock_run.assert_called_once()

    def test_status_with_wiki_root(self):
        mock_run = self._run_main(["status", "--wiki-root", "/tmp/testwiki"])
        mock_run.assert_called_once()

    def test_uncompiled_command(self):
        mock_run = self._run_main(["uncompiled"])
        mock_run.assert_called_once()

    def test_uncompiled_with_wiki_root(self):
        mock_run = self._run_main(["uncompiled", "--wiki-root", "/tmp/testwiki"])
        mock_run.assert_called_once()


# ===========================================================================
# cmd_compile — async with mocked DB and compiler
# ===========================================================================

class TestCmdCompile:
    """cmd_compile() with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_no_experiences_prints_message(self, capsys):
        """When no published experiences, should print message and return."""
        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await cmd_compile(full=False, wiki_root="/tmp/testwiki")
            output = capsys.readouterr().out
            assert "No published experiences found" in output

    @pytest.mark.asyncio
    async def test_with_experiences_calls_compiler(self):
        """With experiences, should invoke WikiCompiler methods."""
        fake_experiences = [{"id": "1", "title": "test"}]
        fake_result = SimpleNamespace(created=1, updated=0, skipped=0, deleted=0, errors=0)

        mock_compiler = AsyncMock()
        mock_compiler.compile_incremental = AsyncMock(return_value=fake_result)
        mock_compiler.full_rebuild = AsyncMock(return_value=fake_result)
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        mock_settings = SimpleNamespace(database=SimpleNamespace(url="postgresql://test"))

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_experiences,
        ):
            with patch("team_memory.config.load_settings", return_value=mock_settings):
                with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                    await cmd_compile(full=False, wiki_root="/tmp/testwiki")
                    mock_compiler.compile_incremental.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_rebuild_calls_full_rebuild(self):
        """With --full flag, should call full_rebuild."""
        fake_experiences = [{"id": "1", "title": "test"}]
        fake_result = SimpleNamespace(created=1, updated=0, skipped=0, deleted=0, errors=0)

        mock_compiler = AsyncMock()
        mock_compiler.full_rebuild = AsyncMock(return_value=fake_result)
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        mock_settings = SimpleNamespace(database=SimpleNamespace(url="postgresql://test"))

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_experiences,
        ):
            with patch("team_memory.config.load_settings", return_value=mock_settings):
                with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                    await cmd_compile(full=True, wiki_root="/tmp/testwiki")
                    mock_compiler.full_rebuild.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_wiki_root_used_when_none(self):
        """When wiki_root is None, _default_wiki_root() should be used."""
        fake_experiences = [{"id": "1"}]
        fake_result = SimpleNamespace(created=0, updated=0, skipped=0, deleted=0, errors=0)

        mock_compiler = AsyncMock()
        mock_compiler.compile_incremental = AsyncMock(return_value=fake_result)
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        mock_settings = SimpleNamespace(database=SimpleNamespace(url="postgresql://test"))

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_experiences,
        ):
            with patch("team_memory.config.load_settings", return_value=mock_settings):
                with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler) as mock_cls:
                    await cmd_compile(full=False, wiki_root=None)
                    # WikiCompiler should have been called with default wiki_root
                    call_kwargs = mock_cls.call_args
                    assert call_kwargs[1].get("wiki_root") or call_kwargs[0][0] == _default_wiki_root()


# ===========================================================================
# cmd_status — async with mocked compiler
# ===========================================================================

class TestCmdStatus:
    """cmd_status() with mocked compiler."""

    @pytest.mark.asyncio
    async def test_displays_stats(self, capsys):
        fake_stats = {"total": 5, "by_project": {"proj_a": 3}, "by_status": {"active": 5}}

        mock_compiler = AsyncMock()
        mock_compiler.get_stats = AsyncMock(return_value=fake_stats)
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                await cmd_status(wiki_root="/tmp/testwiki")
                output = capsys.readouterr().out
                assert "Wiki Status" in output
                assert "5" in output

    @pytest.mark.asyncio
    async def test_displays_by_project(self, capsys):
        fake_stats = {"total": 10, "by_project": {"alpha": 7, "beta": 3}, "by_status": {}}

        mock_compiler = AsyncMock()
        mock_compiler.get_stats = AsyncMock(return_value=fake_stats)
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                await cmd_status(wiki_root="/tmp/testwiki")
                output = capsys.readouterr().out
                assert "By Project" in output
                assert "alpha" in output
                assert "beta" in output

    @pytest.mark.asyncio
    async def test_shows_uncompiled_count(self, capsys):
        """When compiled < pg count, shows warning."""
        fake_stats = {"total": 3, "by_project": {}, "by_status": {}}
        fake_exps = [{"id": i} for i in range(5)]

        mock_compiler = AsyncMock()
        mock_compiler.get_stats = AsyncMock(return_value=fake_stats)
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_exps,
        ):
            with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                await cmd_status(wiki_root="/tmp/testwiki")
                output = capsys.readouterr().out
                assert "not yet compiled" in output

    @pytest.mark.asyncio
    async def test_all_compiled_message(self, capsys):
        """When compiled == pg count, shows all compiled."""
        fake_stats = {"total": 3, "by_project": {}, "by_status": {}}
        fake_exps = [{"id": i} for i in range(3)]

        mock_compiler = AsyncMock()
        mock_compiler.get_stats = AsyncMock(return_value=fake_stats)
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_exps,
        ):
            with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                await cmd_status(wiki_root="/tmp/testwiki")
                output = capsys.readouterr().out
                assert "All" in output and "compiled" in output


# ===========================================================================
# cmd_uncompiled — async with mocked compiler
# ===========================================================================

class TestCmdUncompiled:
    """cmd_uncompiled() with mocked compiler."""

    @pytest.mark.asyncio
    async def test_all_compiled(self, capsys):
        fake_exps = [{"id": "1"}, {"id": "2"}]

        mock_compiler = AsyncMock()
        mock_compiler.get_uncompiled = AsyncMock(return_value=[])
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_exps,
        ):
            with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                await cmd_uncompiled(wiki_root="/tmp/testwiki")
                output = capsys.readouterr().out
                assert "All experiences are compiled" in output

    @pytest.mark.asyncio
    async def test_some_uncompiled(self, capsys):
        fake_exps = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

        mock_compiler = AsyncMock()
        mock_compiler.get_uncompiled = AsyncMock(return_value=["2", "3"])
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_exps,
        ):
            with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                await cmd_uncompiled(wiki_root="/tmp/testwiki")
                output = capsys.readouterr().out
                assert "2 uncompiled" in output
                assert "2" in output
                assert "3" in output

    @pytest.mark.asyncio
    async def test_passes_ids_to_compiler(self):
        fake_exps = [{"id": "aaa"}, {"id": "bbb"}]

        mock_compiler = AsyncMock()
        mock_compiler.get_uncompiled = AsyncMock(return_value=[])
        mock_compiler.__aenter__ = AsyncMock(return_value=mock_compiler)
        mock_compiler.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "team_memory.daemon.wiki_cli._fetch_published_experiences",
            new_callable=AsyncMock,
            return_value=fake_exps,
        ):
            with patch("team_memory.daemon.wiki_compiler.WikiCompiler", return_value=mock_compiler):
                await cmd_uncompiled(wiki_root="/tmp/testwiki")
                mock_compiler.get_uncompiled.assert_called_once_with(["aaa", "bbb"])


# ===========================================================================
# _fetch_published_experiences — async with mocked DB
# ===========================================================================

class TestFetchPublishedExperiences:
    """_fetch_published_experiences() with mocked DB."""

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        """On any exception, should return empty list."""
        with patch("team_memory.config.load_settings", side_effect=ImportError("no module")):
            result = await _fetch_published_experiences()
            assert result == []

    @pytest.mark.asyncio
    async def test_returns_list_on_success(self):
        """On success, should return list of dicts."""
        mock_exp = MagicMock()
        mock_exp.to_dict.return_value = {"id": "1", "title": "Test"}

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_exp]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql://test"

        with patch("team_memory.config.load_settings", return_value=mock_settings):
            with patch("team_memory.storage.database.get_session", return_value=mock_cm):
                with patch("team_memory.storage.database.close_db", new_callable=AsyncMock):
                    result = await _fetch_published_experiences()
                    assert len(result) == 1
                    assert result[0]["id"] == "1"
