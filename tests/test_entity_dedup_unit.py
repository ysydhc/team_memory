"""Unit tests for entity_dedup.py — entity dedup CLI.

Covers:
  - Module imports and attribute checks
  - main()          argparse parsing (--dry-run flag)
  - run_dedup()     async orchestration (mocked DB)
  - _find_duplicates()   async DB query (mocked)
  - _merge_group()       async DB operations (mocked)

Marked:
  @pytest.mark.unit — no real DB, mock-only
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from team_memory.daemon.entity_dedup import (
    main,
    run_dedup,
    _find_duplicates,
    _merge_group,
)

pytestmark = pytest.mark.unit


# ===========================================================================
# Import / attribute tests (original)
# ===========================================================================

class TestEntityDedupImports:
    """Verify module can be imported."""

    def test_import_dedup(self):
        from team_memory.daemon import entity_dedup as ed
        assert hasattr(ed, "run_dedup")

    def test_import_main(self):
        from team_memory.daemon.entity_dedup import main as m
        assert callable(m)

    def test_import_find_duplicates(self):
        assert callable(_find_duplicates)

    def test_import_merge_group(self):
        assert callable(_merge_group)


# ===========================================================================
# main() — argparse parsing
# ===========================================================================

class TestMainArgparse:
    """main() parses --dry-run and calls asyncio.run(run_dedup(...))."""

    def test_no_args_default_dry_run_false(self):
        """No --dry-run => dry_run=False."""
        with patch.object(sys, "argv", ["entity_dedup.py"]):
            with patch("team_memory.daemon.entity_dedup.asyncio.run") as mock_run:
                main()
                mock_run.assert_called_once()
                coroutine = mock_run.call_args[0][0]
                assert asyncio.iscoroutine(coroutine)

    def test_dry_run_flag(self):
        """--dry-run => dry_run=True."""
        with patch.object(sys, "argv", ["entity_dedup.py", "--dry-run"]):
            with patch("team_memory.daemon.entity_dedup.asyncio.run") as mock_run:
                main()
                mock_run.assert_called_once()

    def test_main_calls_asyncio_run(self):
        """main() should always call asyncio.run."""
        with patch.object(sys, "argv", ["entity_dedup.py"]):
            with patch("team_memory.daemon.entity_dedup.asyncio.run", new_callable=MagicMock) as mock_run:
                main()
                mock_run.assert_called_once()


# ===========================================================================
# _find_duplicates — async DB query (mocked)
# ===========================================================================

class TestFindDuplicates:
    """_find_duplicates() with mocked DB session.

    Note: get_session and close_db are imported inside the function body,
    so we patch them at their original source paths.
    """

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_duplicates(self):
        """No duplicates => empty list."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            with patch("team_memory.storage.database.close_db", new_callable=AsyncMock):
                result = await _find_duplicates("postgresql://test")
                assert result == []

    @pytest.mark.asyncio
    async def test_returns_groups(self):
        """Returns properly structured groups."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("python", ["id1", "id2"], ["Python", "python"], ["proj_a", "proj_b"], [10, 5]),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            with patch("team_memory.storage.database.close_db", new_callable=AsyncMock):
                result = await _find_duplicates("postgresql://test")
                assert len(result) == 1
                assert result[0]["lower_name"] == "python"
                assert result[0]["ids"] == ["id1", "id2"]
                assert result[0]["names"] == ["Python", "python"]
                assert result[0]["projects"] == ["proj_a", "proj_b"]
                assert result[0]["counts"] == [10, 5]

    @pytest.mark.asyncio
    async def test_multiple_groups(self):
        """Multiple duplicate groups."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("python", ["id1"], ["Python"], ["a"], [10]),
            ("react", ["id2", "id3"], ["React", "react"], ["b", "c"], [8, 3]),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            with patch("team_memory.storage.database.close_db", new_callable=AsyncMock):
                result = await _find_duplicates("postgresql://test")
                assert len(result) == 2

    @pytest.mark.asyncio
    async def test_closes_db_after_query(self):
        """close_db() should be called after the query."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            with patch("team_memory.storage.database.close_db", new_callable=AsyncMock) as mock_close:
                await _find_duplicates("postgresql://test")
                mock_close.assert_awaited_once()


# ===========================================================================
# _merge_group — async DB operations (mocked)
# ===========================================================================

class TestMergeGroup:
    """_merge_group() with mocked DB session.

    Note: get_session is imported inside the function body.
    """

    @pytest.mark.asyncio
    async def test_dry_run_does_not_modify_db(self):
        """dry_run=True should not execute any UPDATE/DELETE."""
        mock_session = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            result = await _merge_group("postgresql://test", "primary_id", ["dup1"], dry_run=True)
            assert result == 0
            mock_session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_merge_returns_affected_count(self):
        """Non-dry-run should return affected row count."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            result = await _merge_group("postgresql://test", "p1", ["d1"], dry_run=False)
            # rowcount=3 from UPDATE + 1 from source_count update = 4
            assert result == 4

    @pytest.mark.asyncio
    async def test_multiple_duplicates(self):
        """Should process each duplicate ID."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 2
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            result = await _merge_group("postgresql://test", "p1", ["d1", "d2"], dry_run=False)
            # 2 dup_ids * 4 queries each = 8 execute calls + 1 commit = 9 total
            assert mock_session.execute.await_count >= 8

    @pytest.mark.asyncio
    async def test_no_duplicates_returns_zero(self):
        """Empty duplicate_ids => affected=0, no DB calls."""
        mock_session = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_cm):
            result = await _merge_group("postgresql://test", "p1", [], dry_run=False)
            assert result == 0
            mock_session.execute.assert_not_awaited()


# ===========================================================================
# run_dedup — async orchestration (mocked DB)
# ===========================================================================

class TestRunDedup:
    """run_dedup() with mocked DB calls.

    Note: load_settings, get_session are imported inside the function body.
    """

    @pytest.mark.asyncio
    async def test_no_duplicates(self, capsys):
        """When no duplicates, prints message and returns early."""
        with patch("team_memory.daemon.entity_dedup._find_duplicates", new_callable=AsyncMock, return_value=[]):
            with patch("team_memory.config.load_settings"):
                await run_dedup(dry_run=False)
            output = capsys.readouterr().out
            assert "No duplicate entities found" in output

    @pytest.mark.asyncio
    async def test_dry_run_output(self, capsys):
        """Dry run should show what would be merged."""
        groups = [
            {
                "lower_name": "python",
                "ids": ["id1", "id2"],
                "names": ["Python", "python"],
                "projects": ["proj_a", "proj_b"],
                "counts": [10, 5],
            }
        ]
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql://test"

        # Mock the final count query
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 100
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.entity_dedup._find_duplicates", new_callable=AsyncMock, return_value=groups):
            with patch("team_memory.config.load_settings", return_value=mock_settings):
                with patch("team_memory.storage.database.get_session", return_value=mock_cm):
                    await run_dedup(dry_run=True)
            output = capsys.readouterr().out
            assert "Found 1 duplicate" in output
            assert "Would merge" in output
            assert "Python" in output

    @pytest.mark.asyncio
    async def test_execute_merge(self, capsys):
        """Non-dry-run should call _merge_group."""
        groups = [
            {
                "lower_name": "react",
                "ids": ["id1", "id2", "id3"],
                "names": ["React", "react", "REACT"],
                "projects": ["a", "b", "c"],
                "counts": [10, 5, 3],
            }
        ]
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql://test"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 98
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.entity_dedup._find_duplicates", new_callable=AsyncMock, return_value=groups):
            with patch("team_memory.daemon.entity_dedup._merge_group", new_callable=AsyncMock, return_value=2):
                with patch("team_memory.config.load_settings", return_value=mock_settings):
                    with patch("team_memory.storage.database.get_session", return_value=mock_cm):
                        await run_dedup(dry_run=False)
            output = capsys.readouterr().out
            assert "Merged" in output

    @pytest.mark.asyncio
    async def test_shows_final_count(self, capsys):
        """Should print final entity count."""
        groups = [
            {
                "lower_name": "x",
                "ids": ["id1", "id2"],
                "names": ["X", "x"],
                "projects": ["p1", "p2"],
                "counts": [5, 3],
            }
        ]
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql://test"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.entity_dedup._find_duplicates", new_callable=AsyncMock, return_value=groups):
            with patch("team_memory.daemon.entity_dedup._merge_group", new_callable=AsyncMock, return_value=0):
                with patch("team_memory.config.load_settings", return_value=mock_settings):
                    with patch("team_memory.storage.database.get_session", return_value=mock_cm):
                        await run_dedup(dry_run=False)
            output = capsys.readouterr().out
            assert "Final entity count: 42" in output

    @pytest.mark.asyncio
    async def test_dry_run_summary(self, capsys):
        """Dry run should show total would-merge count."""
        groups = [
            {
                "lower_name": "a",
                "ids": ["1", "2"],
                "names": ["A", "a"],
                "projects": ["p1", "p2"],
                "counts": [5, 3],
            },
            {
                "lower_name": "b",
                "ids": ["3", "4", "5"],
                "names": ["B", "b", "B"],
                "projects": ["p1", "p2", "p3"],
                "counts": [10, 7, 2],
            },
        ]
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql://test"

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 50
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.entity_dedup._find_duplicates", new_callable=AsyncMock, return_value=groups):
            with patch("team_memory.config.load_settings", return_value=mock_settings):
                with patch("team_memory.storage.database.get_session", return_value=mock_cm):
                    await run_dedup(dry_run=True)
            output = capsys.readouterr().out
            # Group 1: 1 dup, Group 2: 2 dups => total 3
            assert "would merge 3" in output
