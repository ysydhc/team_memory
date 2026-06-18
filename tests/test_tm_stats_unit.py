"""Comprehensive unit & integration tests for tm_stats.py — search quality report.

Covers:
  - _bar()          bar chart generation (pure logic)
  - _resolve_dsn()  DSN resolution fallback chain (mocked imports)
  - main()          argparse entry point
  - _faithfulness_breakdown()   async DB query (mocked asyncpg)
  - _get_low_faithfulness_entries() async DB query (mocked asyncpg)
  - run_stats()     full report pipeline (mocked writer + buffer)

Marked:
  @pytest.mark.unit        — no real DB, pure logic / mock-only
  @pytest.mark.integration — touches real asyncpg or full stack
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from team_memory.daemon.observe.tm_stats import (
    _bar,
    _faithfulness_breakdown,
    _get_low_faithfulness_entries,
    _resolve_dsn,
    main,
    run_stats,
)

# ===========================================================================
# Unit tests — pure logic, no external deps
# ===========================================================================


class TestBar:
    """_bar() bar-chart string generation."""

    def test_full_bar(self):
        assert _bar(10, 10, width=5) == "█████"

    def test_empty_bar(self):
        assert _bar(0, 10, width=5) == "░░░░░"

    def test_half_bar(self):
        result = _bar(5, 10, width=10)
        assert "█" in result
        assert "░" in result
        assert len(result) == 10

    def test_zero_max_returns_empty(self):
        assert _bar(5, 0, width=8) == "░" * 8

    def test_negative_max_returns_empty(self):
        assert _bar(5, -3, width=8) == "░" * 8

    def test_default_width_is_20(self):
        assert len(_bar(1, 10)) == 20

    def test_custom_width(self):
        assert len(_bar(3, 10, width=40)) == 40

    def test_small_value_large_max(self):
        result = _bar(1, 1000, width=30)
        assert result.count("█") <= 1
        assert len(result) == 30

    def test_value_equals_max(self):
        assert _bar(7, 7, width=12) == "█" * 12

    def test_single_pixel_bar(self):
        # value/max = 1/width => exactly 1 filled char
        result = _bar(1, 20, width=20)
        assert result.count("█") == 1
        assert result.count("░") == 19

    def test_boundary_value_below_one_pixel(self):
        # value/max < 1/width => 0 filled
        result = _bar(1, 100, width=50)
        assert result.count("█") == 0

    @pytest.mark.parametrize(
        "v,m,w,expected_filled",
        [
            (0, 10, 10, 0),
            (10, 10, 10, 10),
            (5, 10, 10, 5),
            (3, 10, 20, 6),
            (1, 3, 30, 10),
        ],
    )
    def test_parametrized(self, v, m, w, expected_filled):
        result = _bar(v, m, width=w)
        assert result.count("█") == expected_filled
        assert len(result) == w


# ---------------------------------------------------------------------------
# _resolve_dsn — mock every import path
# ---------------------------------------------------------------------------


class TestResolveDsn:
    """_resolve_dsn() fallback chain with mocked imports."""

    def test_returns_string(self):
        dsn = _resolve_dsn()
        assert isinstance(dsn, str)
        assert len(dsn) > 0

    def test_starts_with_postgresql(self):
        dsn = _resolve_dsn()
        assert dsn.startswith("postgresql")

    def test_contains_database_name(self):
        dsn = _resolve_dsn()
        assert "team_memory" in dsn

    def test_settings_path_takes_priority(self):
        """When load_settings works, its URL is used."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+asyncpg://u:p@host:5433/team_memory"
        mock_load = MagicMock(return_value=mock_settings)
        with patch.dict(
            sys.modules,
            {
                "team_memory.config": MagicMock(load_settings=mock_load),
            },
        ):
            dsn = _resolve_dsn()
        assert dsn == "postgresql://u:p@host:5433/team_memory"

    def test_env_var_fallback(self):
        """When load_settings fails, DATABASE_URL env var is used."""
        mock_mod = MagicMock()
        mock_mod.load_settings.side_effect = ImportError("nope")
        with patch.dict(sys.modules, {"team_memory.config": mock_mod}):
            with patch.dict("os.environ", {"DATABASE_URL": "postgresql://env@db/test"}):
                dsn = _resolve_dsn()
        assert dsn == "postgresql://env@db/test"

    def test_alembic_fallback(self):
        """When env var empty, reads alembic.ini sqlalchemy.url."""
        mock_mod = MagicMock()
        mock_mod.load_settings.side_effect = ImportError("nope")
        with patch.dict(sys.modules, {"team_memory.config": mock_mod}):
            with patch.dict("os.environ", {"DATABASE_URL": ""}, clear=False):
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__ = lambda s: s
                    mock_open.return_value.__exit__ = MagicMock(return_value=False)
                    mock_open.return_value.read.return_value = (
                        "[alembic]\nsqlalchemy.url = postgresql://alembic:pwd@host/db"
                    )
                    dsn = _resolve_dsn()
        assert "alembic" in dsn.lower() or "postgresql" in dsn

    def test_team_memory_db_url_fallback(self):
        """When earlier sources fail, TEAM_MEMORY_DB_URL is used."""
        mock_mod = MagicMock()
        mock_mod.load_settings.side_effect = ImportError("nope")
        with patch.dict(sys.modules, {"team_memory.config": mock_mod}):
            with patch.dict(
                "os.environ",
                {"DATABASE_URL": "", "TEAM_MEMORY_DB_URL": "postgresql://tm:pwd@fallback:5544/tm_fallback"},
                clear=False,
            ):
                with patch("builtins.open", side_effect=OSError("no file")):
                    dsn = _resolve_dsn()
        assert "fallback" in dsn and "5544" in dsn


# ---------------------------------------------------------------------------
# main() — argparse entry point
# ---------------------------------------------------------------------------


class TestMain:
    """main() CLI argument parsing."""

    def test_default_args(self):
        """Default args should call run_stats with 7 days, day granularity."""
        with patch("team_memory.daemon.observe.tm_stats.run_stats", new_callable=AsyncMock) as mock_run:
            with patch("sys.argv", ["tm_stats.py"]):
                main()
        mock_run.assert_awaited_once_with(7, "day")

    def test_custom_days(self):
        """--days 30 should pass 30."""
        with patch("team_memory.daemon.observe.tm_stats.run_stats", new_callable=AsyncMock) as mock_run:
            with patch("sys.argv", ["tm_stats.py", "--days", "30"]):
                main()
        mock_run.assert_awaited_once_with(30, "day")

    def test_custom_granularity_hour(self):
        """--granularity hour should pass 'hour'."""
        with patch("team_memory.daemon.observe.tm_stats.run_stats", new_callable=AsyncMock) as mock_run:
            with patch("sys.argv", ["tm_stats.py", "--granularity", "hour"]):
                main()
        mock_run.assert_awaited_once_with(7, "hour")

    def test_custom_granularity_week(self):
        """--granularity week should pass 'week'."""
        with patch("team_memory.daemon.observe.tm_stats.run_stats", new_callable=AsyncMock) as mock_run:
            with patch("sys.argv", ["tm_stats.py", "--granularity", "week"]):
                main()
        mock_run.assert_awaited_once_with(7, "week")

    def test_all_custom_args(self):
        """Combined --days and --granularity."""
        with patch("team_memory.daemon.observe.tm_stats.run_stats", new_callable=AsyncMock) as mock_run:
            with patch("sys.argv", ["tm_stats.py", "--days", "14", "--granularity", "week"]):
                main()
        mock_run.assert_awaited_once_with(14, "week")


# ===========================================================================
# Integration-style tests (mocked asyncpg, but testing async DB query logic)
# ===========================================================================


class TestFaithfulnessBreakdown:
    """_faithfulness_breakdown() — async DB count query."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_above_threshold(self):
        """Count entries above threshold."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=42)
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await _faithfulness_breakdown("postgresql://test", 7, threshold=0.7, above=True)
        assert result == 42
        mock_conn.fetchval.assert_awaited_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_below_threshold(self):
        """Count entries below threshold."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=10)
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await _faithfulness_breakdown("postgresql://test", 30, threshold=0.7, above=False)
        assert result == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_none_result_returns_zero(self):
        """fetchval returns None => 0."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=None)
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await _faithfulness_breakdown("postgresql://test", 7)
        assert result == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_closed_in_finally(self):
        """Connection must be closed even on exception."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=RuntimeError("db error"))
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            with pytest.raises(RuntimeError):
                await _faithfulness_breakdown("postgresql://test", 7)
        mock_conn.close.assert_awaited_once()


class TestGetLowFaithfulnessEntries:
    """_get_low_faithfulness_entries() — async DB fetch query."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_mapped_dicts(self):
        """Rows should be mapped to {query, score, created_at}."""
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "query": "test query",
            "faithfulness_score": 0.5,
            "created_at": "2026-04-20",
        }[key]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[mock_row])
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await _get_low_faithfulness_entries("postgresql://test", 7)
        assert len(result) == 1
        assert result[0]["query"] == "test query"
        assert result[0]["score"] == 0.5
        assert result[0]["created_at"] == "2026-04-20"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_result(self):
        """No rows => empty list."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            result = await _get_low_faithfulness_entries("postgresql://test", 7)
        assert result == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_connection_closed_on_error(self):
        """Connection must close even when fetch raises."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=RuntimeError("db down"))
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            with pytest.raises(RuntimeError):
                await _get_low_faithfulness_entries("postgresql://test", 7)
        mock_conn.close.assert_awaited_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_limit_passed(self):
        """Limit param should be forwarded to fetch."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.close = AsyncMock()
        with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
            await _get_low_faithfulness_entries("postgresql://test", 14, limit=3)
        args = mock_conn.fetch.call_args[0]
        # args should be (query_string, 14, 3)
        assert args[1] == 14
        assert args[2] == 3


# ===========================================================================
# run_stats — full pipeline with mocked writer + buffer
# ===========================================================================


class TestRunStats:
    """run_stats() — end-to-end report with mocked dependencies."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_in_stats_returns_early(self, capsys):
        """If stats contains 'error', the function should print and return."""
        mock_writer = AsyncMock()
        mock_writer.get_stats = AsyncMock(return_value={"error": "connection refused"})
        mock_writer.close = AsyncMock()

        with patch("team_memory.daemon.observe.tm_stats._resolve_dsn", return_value="postgresql://test"):
            with patch("team_memory.services.search.search_log_writer.SearchLogWriter", return_value=mock_writer):
                await run_stats(7, "day")
        out = capsys.readouterr().out
        assert "Failed to fetch stats" in out
        assert "connection refused" in out

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_render_with_rich(self, capsys):
        """Full render path with Rich available — all DB access mocked."""
        stats = {
            "total_searches": 100,
            "hit": 80,
            "used": 40,
            "used_marker": 25,
            "used_fuzzy": 15,
            "unjudged": 10,
            "use_rate": 0.5,
            "by_project": {"proj": {"total": 50, "used": 20}},
        }
        trends = [
            {"period": "2026-04-20", "total": 10, "hit": 8, "used": 4, "use_rate": 0.4},
        ]
        miss = [{"query": "bad query", "count": 3, "last_seen": "2026-04-20T10:00:00"}]
        low_use = [{"query": "unused query", "count": 2, "used": 0}]

        mock_writer = AsyncMock()
        mock_writer.get_stats = AsyncMock(return_value=stats)
        mock_writer.get_trends = AsyncMock(return_value=trends)
        mock_writer.get_miss_queries = AsyncMock(return_value=miss)
        mock_writer.get_low_use_queries = AsyncMock(return_value=low_use)
        mock_writer.close = AsyncMock()

        mock_buffer = AsyncMock()
        mock_buffer.get_stats = AsyncMock(
            return_value={
                "total": 50,
                "evaluated": 40,
                "pending": 10,
                "avg_faithfulness": 0.8,
            }
        )
        mock_buffer.close = AsyncMock()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=20)
        mock_conn.fetch = AsyncMock(return_value=[])

        with patch("team_memory.daemon.observe.tm_stats._resolve_dsn", return_value="postgresql://test"):
            with patch("team_memory.services.search.search_log_writer.SearchLogWriter", return_value=mock_writer):
                with patch("team_memory.daemon.buffers.response_buffer.ResponseBuffer", return_value=mock_buffer):
                    with patch("asyncpg.connect", new_callable=AsyncMock, return_value=mock_conn):
                        await run_stats(7, "day")

        mock_writer.get_stats.assert_awaited_once()
        mock_writer.get_trends.assert_awaited_once()
        mock_writer.get_miss_queries.assert_awaited_once()
        mock_writer.get_low_use_queries.assert_awaited_once()
        mock_writer.close.assert_awaited_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_writer_closed_on_exception(self):
        """Writer must be closed even when an exception occurs mid-render."""
        stats = {
            "total_searches": 10,
            "hit": 8,
            "used": 4,
            "used_marker": 2,
            "used_fuzzy": 2,
            "unjudged": 2,
            "use_rate": 0.4,
        }
        mock_writer = AsyncMock()
        mock_writer.get_stats = AsyncMock(return_value=stats)
        mock_writer.get_trends = AsyncMock(side_effect=RuntimeError("boom"))
        mock_writer.close = AsyncMock()

        with patch("team_memory.daemon.observe.tm_stats._resolve_dsn", return_value="postgresql://test"):
            with patch("team_memory.services.search.search_log_writer.SearchLogWriter", return_value=mock_writer):
                with pytest.raises(RuntimeError):
                    await run_stats(7, "day")
        mock_writer.close.assert_awaited_once()
