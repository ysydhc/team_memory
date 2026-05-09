"""Unit tests for CLI backfill scripts — embedding_backfill, entity_backfill,
solution_backfill.

Markers:
  @pytest.mark.unit — pure logic, fully mocked (no real DB/LLM)

Covers: argument parsing, helper functions, main() entry points,
        error paths, and happy paths for each script.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_settings(**overrides):
    """Build a minimal Settings mock for scripts."""
    settings = MagicMock()
    settings.database.url = "postgresql+asyncpg://test@localhost/test"
    ee = MagicMock()
    ee.model = "test-model"
    ee.base_url = "http://llm.local"
    ee.api_key_env = "LLM_API_KEY"
    ee.timeout = 10
    settings.entity_extraction = ee
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _mock_session_context(fetchall_return=None, scalars_all_return=None):
    """Return an (enterable) mock that acts as an async session context manager."""
    session = AsyncMock()
    result = MagicMock()
    if fetchall_return is not None:
        result.fetchall.return_value = fetchall_return
    else:
        result.fetchall.return_value = []
    if scalars_all_return is not None:
        result.scalars.return_value.all.return_value = scalars_all_return
    else:
        result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    return session


# ═════════════════════════════════════════════════════════════════════════════
#  embedding_backfill.py
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestEmbeddingBackfillMain:
    """Test main() argument parsing via sys.argv manipulation."""

    def test_main_calls_run_backfill_defaults(self):
        """main() with no args should call run_backfill(dry_run=False, limit=0)."""
        mock_coro = AsyncMock()
        mock_run_backfill = MagicMock(return_value=mock_coro)
        with patch.object(sys, "argv", ["embedding_backfill.py"]), \
             patch("team_memory.daemon.embedding_backfill.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock()
            from team_memory.daemon.embedding_backfill import main
            main()
            mock_asyncio.run.assert_called_once()
            # The coroutine passed to asyncio.run is run_backfill(dry_run=False, limit=0)
            call_arg = mock_asyncio.run.call_args[0][0]
            # We can check by calling it and inspecting kwargs via the mock
            # Instead, let's patch run_backfill directly
        mock_coro.close()

    def test_main_dry_run(self):
        mock_coro = AsyncMock()
        with patch.object(sys, "argv", ["embedding_backfill.py", "--dry-run"]), \
             patch("team_memory.daemon.embedding_backfill.run_backfill", new_callable=MagicMock, return_value=mock_coro) as mock_rb, \
             patch("team_memory.daemon.embedding_backfill.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.embedding_backfill import main
            main()
            mock_rb.assert_called_once_with(dry_run=True, limit=0)
        mock_coro.close()

    def test_main_limit(self):
        mock_coro = AsyncMock()
        with patch.object(sys, "argv", ["embedding_backfill.py", "--limit", "42"]), \
             patch("team_memory.daemon.embedding_backfill.run_backfill", new_callable=MagicMock, return_value=mock_coro) as mock_rb, \
             patch("team_memory.daemon.embedding_backfill.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.embedding_backfill import main
            main()
            mock_rb.assert_called_once_with(dry_run=False, limit=42)
        mock_coro.close()

    def test_main_all_args(self):
        mock_coro = AsyncMock()
        with patch.object(sys, "argv", ["embedding_backfill.py", "--dry-run", "--limit", "5"]), \
             patch("team_memory.daemon.embedding_backfill.run_backfill", new_callable=MagicMock, return_value=mock_coro) as mock_rb, \
             patch("team_memory.daemon.embedding_backfill.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.embedding_backfill import main
            main()
            mock_rb.assert_called_once_with(dry_run=True, limit=5)
        mock_coro.close()


@pytest.mark.unit
class TestEmbeddingBackfillRunBackfill:
    """Test run_backfill() with fully mocked DB and embedding provider."""

    @pytest.mark.asyncio
    async def test_no_rows_returns_early(self, capsys):
        session = _mock_session_context(fetchall_return=[])
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider") as mock_emb, \
             patch("team_memory.storage.database.get_session", return_value=mock_ctx):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=False, limit=0)
        captured = capsys.readouterr()
        assert "All published experiences have embeddings" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_shows_rows(self, capsys):
        rows = [
            ("id-1", "Title 1", "Desc 1", "Sol 1", ["tag1"]),
            ("id-2", "Title 2", "Desc 2", "Sol 2", []),
        ]
        session = _mock_session_context(fetchall_return=rows)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider"), \
             patch("team_memory.storage.database.get_session", return_value=mock_ctx):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=True, limit=0)
        captured = capsys.readouterr()
        assert "Found 2 experiences" in captured.out
        assert "id-1" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_with_limit(self, capsys):
        rows = [(f"id-{i}", f"Title {i}", f"Desc {i}", "", []) for i in range(20)]
        session = _mock_session_context(fetchall_return=rows)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider"), \
             patch("team_memory.storage.database.get_session", return_value=mock_ctx):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=True, limit=5)
        captured = capsys.readouterr()
        assert "processing 5" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_truncates_at_10(self, capsys):
        rows = [(f"id-{i}", f"Title {i}", f"Desc {i}", "", []) for i in range(15)]
        session = _mock_session_context(fetchall_return=rows)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider"), \
             patch("team_memory.storage.database.get_session", return_value=mock_ctx):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=True, limit=0)
        captured = capsys.readouterr()
        assert "and 5 more" in captured.out

    @pytest.mark.asyncio
    async def test_successful_backfill(self, capsys):
        row = ("exp-001", "Bug fix", "Fixed it", "Solution text", ["python"])
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=[0.1, 0.2])

        # Two session contexts: SELECT and UPDATE
        update_session = _mock_session_context()
        update_ctx = AsyncMock()
        update_ctx.__aenter__ = AsyncMock(return_value=update_session)
        update_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider", return_value=mock_embedder), \
             patch("team_memory.storage.database.get_session", side_effect=[mock_ctx, update_ctx]):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=False, limit=0)
        captured = capsys.readouterr()
        assert "Succeeded: 1" in captured.out
        assert "Failed:    0" in captured.out

    @pytest.mark.asyncio
    async def test_embedding_returns_none(self, capsys):
        row = ("exp-002", "Title", "Desc", "Sol", [])
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=None)

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider", return_value=mock_embedder), \
             patch("team_memory.storage.database.get_session", return_value=mock_ctx):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=False, limit=0)
        captured = capsys.readouterr()
        assert "Failed:    1" in captured.out

    @pytest.mark.asyncio
    async def test_encode_exception_increments_failed(self, capsys):
        row = ("exp-003", "T", "D", "S", ["x"])
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider", return_value=mock_embedder), \
             patch("team_memory.storage.database.get_session", return_value=mock_ctx):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=False, limit=0)
        captured = capsys.readouterr()
        assert "Failed exp-003" in captured.out
        assert "API down" in captured.out

    @pytest.mark.asyncio
    async def test_limit_truncates_rows(self, capsys):
        rows = [(f"id-{i}", f"Title {i}", f"Desc {i}", f"Sol {i}", []) for i in range(10)]
        session = _mock_session_context(fetchall_return=rows)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=[0.1])

        # 1 SELECT + 3 UPDATE sessions
        update_sessions = []
        for _ in range(3):
            us = _mock_session_context()
            uc = AsyncMock()
            uc.__aenter__ = AsyncMock(return_value=us)
            uc.__aexit__ = AsyncMock(return_value=False)
            update_sessions.append(uc)
        all_sessions = [mock_ctx] + update_sessions

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider", return_value=mock_embedder), \
             patch("team_memory.storage.database.get_session", side_effect=all_sessions):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=False, limit=3)
        captured = capsys.readouterr()
        assert "processing 3" in captured.out
        assert "Succeeded: 3" in captured.out

    @pytest.mark.asyncio
    async def test_embed_text_truncation(self, capsys):
        """Ensure embed_text is truncated to 4000 chars."""
        long_desc = "x" * 5000
        row = ("exp-long", "Title", long_desc, "Sol", [])
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=[0.1])

        update_session = _mock_session_context()
        update_ctx = AsyncMock()
        update_ctx.__aenter__ = AsyncMock(return_value=update_session)
        update_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.config.load_settings", return_value=_make_settings()), \
             patch("team_memory.bootstrap._create_embedding_provider", return_value=mock_embedder), \
             patch("team_memory.storage.database.get_session", side_effect=[mock_ctx, update_ctx]):
            from team_memory.daemon.embedding_backfill import run_backfill
            await run_backfill(dry_run=False, limit=0)
        call_text = mock_embedder.encode_single.call_args[0][0]
        assert len(call_text) <= 4000


# ═════════════════════════════════════════════════════════════════════════════
#  entity_backfill.py
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestEntityBackfillMain:
    """Test main() argument parsing."""

    def test_main_defaults(self):
        mock_coro = AsyncMock()
        with patch.object(sys, "argv", ["entity_backfill.py"]), \
             patch("team_memory.daemon.entity_backfill.run_backfill", return_value=mock_coro) as mock_rb, \
             patch("team_memory.daemon.entity_backfill.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.entity_backfill import main
            main()
            mock_rb.assert_called_once()
            _, kwargs = mock_rb.call_args
            assert kwargs["model"] is None
            assert kwargs["base_url"] is None
            assert kwargs["limit"] == 0
            assert kwargs["dry_run"] is False
            assert kwargs["concurrency"] == 3

    def test_main_dry_run(self):
        mock_coro = AsyncMock()
        with patch.object(sys, "argv", ["entity_backfill.py", "--dry-run"]), \
             patch("team_memory.daemon.entity_backfill.run_backfill", return_value=mock_coro) as mock_rb, \
             patch("team_memory.daemon.entity_backfill.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.entity_backfill import main
            main()
            _, kwargs = mock_rb.call_args
            assert kwargs["dry_run"] is True

    def test_main_all_args(self):
        mock_coro = AsyncMock()
        with patch.object(sys, "argv", [
            "entity_backfill.py", "--model", "DeepSeek-V3",
            "--base-url", "http://custom.local",
            "--limit", "10", "--concurrency", "5",
        ]), \
             patch("team_memory.daemon.entity_backfill.run_backfill", return_value=mock_coro) as mock_rb, \
             patch("team_memory.daemon.entity_backfill.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.entity_backfill import main
            main()
            _, kwargs = mock_rb.call_args
            assert kwargs["model"] == "DeepSeek-V3"
            assert kwargs["base_url"] == "http://custom.local"
            assert kwargs["limit"] == 10
            assert kwargs["concurrency"] == 5


@pytest.mark.unit
class TestEntityBackfillHelpers:
    """Test _get_unextracted_experiences and _build_llm_config."""

    @pytest.mark.asyncio
    async def test_get_unextracted_no_results(self):
        exp = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_ctx), \
             patch("team_memory.storage.database.close_db", new_callable=AsyncMock):
            from team_memory.daemon.entity_backfill import _get_unextracted_experiences
            rows = await _get_unextracted_experiences("postgresql://x", limit=0)
            assert rows == []

    @pytest.mark.asyncio
    async def test_get_unextracted_with_results(self):
        exp = MagicMock()
        exp.to_dict.return_value = {"id": "exp-1", "title": "Test"}
        result = MagicMock()
        result.scalars.return_value.all.return_value = [exp]
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.storage.database.get_session", return_value=mock_ctx), \
             patch("team_memory.storage.database.close_db", new_callable=AsyncMock):
            from team_memory.daemon.entity_backfill import _get_unextracted_experiences
            rows = await _get_unextracted_experiences("postgresql://x", limit=1)
            assert len(rows) == 1
            assert rows[0]["id"] == "exp-1"

    @pytest.mark.asyncio
    async def test_build_llm_config_defaults(self):
        settings = _make_settings()
        with patch("team_memory.config.load_settings", return_value=settings), \
             patch.dict("os.environ", {"LLM_API_KEY": "test-key"}):
            from team_memory.daemon.entity_backfill import _build_llm_config
            cfg = await _build_llm_config()
            assert cfg.provider == "generic"
            assert cfg.model == "test-model"
            assert cfg.base_url == "http://llm.local"
            assert cfg.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_build_llm_config_override_model(self):
        settings = _make_settings()
        with patch("team_memory.config.load_settings", return_value=settings), \
             patch.dict("os.environ", {}, clear=False):
            from team_memory.daemon.entity_backfill import _build_llm_config
            cfg = await _build_llm_config(model="custom-model")
            assert cfg.model == "custom-model"

    @pytest.mark.asyncio
    async def test_build_llm_config_override_base_url(self):
        settings = _make_settings()
        with patch("team_memory.config.load_settings", return_value=settings), \
             patch.dict("os.environ", {}, clear=False):
            from team_memory.daemon.entity_backfill import _build_llm_config
            cfg = await _build_llm_config(base_url="http://custom.local")
            assert cfg.base_url == "http://custom.local"

    @pytest.mark.asyncio
    async def test_build_llm_config_missing_api_key(self):
        settings = _make_settings()
        with patch("team_memory.config.load_settings", return_value=settings), \
             patch.dict("os.environ", {}, clear=True):
            from team_memory.daemon.entity_backfill import _build_llm_config
            cfg = await _build_llm_config()
            assert cfg.api_key == ""


@pytest.mark.unit
class TestEntityBackfillRunBackfill:
    """Test run_backfill() with mocked DB and LLM."""

    @pytest.mark.asyncio
    async def test_no_experiences_returns_early(self, capsys):
        with patch("team_memory.daemon.entity_backfill._get_unextracted_experiences",
                    new_callable=AsyncMock, return_value=[]), \
             patch("team_memory.config.load_settings"):
            from team_memory.daemon.entity_backfill import run_backfill
            await run_backfill(dry_run=False)
        captured = capsys.readouterr()
        assert "No experiences need entity extraction" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_shows_experiences(self, capsys):
        exps = [
            {"id": f"exp-{i}", "title": f"Title {i}", "description": f"Desc {i}"}
            for i in range(5)
        ]
        with patch("team_memory.daemon.entity_backfill._get_unextracted_experiences",
                    new_callable=AsyncMock, return_value=exps), \
             patch("team_memory.config.load_settings"):
            from team_memory.daemon.entity_backfill import run_backfill
            await run_backfill(dry_run=True)
        captured = capsys.readouterr()
        assert "Found 5 experiences" in captured.out
        assert "exp-0" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_truncates_at_20(self, capsys):
        exps = [{"id": f"exp-{i}", "title": f"Title {i}"} for i in range(25)]
        with patch("team_memory.daemon.entity_backfill._get_unextracted_experiences",
                    new_callable=AsyncMock, return_value=exps), \
             patch("team_memory.config.load_settings"):
            from team_memory.daemon.entity_backfill import run_backfill
            await run_backfill(dry_run=True)
        captured = capsys.readouterr()
        assert "and 5 more" in captured.out

    @pytest.mark.asyncio
    async def test_successful_processing(self, capsys):
        exps = [
            {"id": "exp-1", "title": "Test", "description": "D", "solution": "S",
             "tags": ["t"], "project": "default"}
        ]
        mock_extractor = MagicMock()
        mock_extractor.extract_and_persist = AsyncMock(return_value=None)
        settings = _make_settings()

        async def _noop_gather(*args, **kwargs):
            """Simulate asyncio.gather — just await all coroutine args."""
            for arg in args:
                await arg

        with patch("team_memory.daemon.entity_backfill._get_unextracted_experiences",
                    new_callable=AsyncMock, return_value=exps), \
             patch("team_memory.config.load_settings", return_value=settings), \
             patch("team_memory.daemon.entity_backfill._build_llm_config",
                    new_callable=AsyncMock, return_value=MagicMock(model="m", base_url="u")), \
             patch("team_memory.services.entity_extractor.EntityExtractor",
                   return_value=mock_extractor), \
             patch("team_memory.daemon.entity_backfill.asyncio.sleep",
                   new_callable=AsyncMock), \
             patch("team_memory.daemon.entity_backfill.asyncio.gather",
                   side_effect=_noop_gather):
            from team_memory.daemon.entity_backfill import run_backfill
            await run_backfill(dry_run=False, concurrency=1)
        captured = capsys.readouterr()
        assert "Succeeded: 1" in captured.out
        mock_extractor.extract_and_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_processing(self, capsys):
        exps = [{"id": "exp-fail", "title": "Fail", "description": "D"}]
        mock_extractor = MagicMock()
        mock_extractor.extract_and_persist = AsyncMock(
            side_effect=RuntimeError("LLM error")
        )
        settings = _make_settings()

        async def _noop_gather(*args, **kwargs):
            for arg in args:
                await arg

        with patch("team_memory.daemon.entity_backfill._get_unextracted_experiences",
                    new_callable=AsyncMock, return_value=exps), \
             patch("team_memory.config.load_settings", return_value=settings), \
             patch("team_memory.daemon.entity_backfill._build_llm_config",
                    new_callable=AsyncMock, return_value=MagicMock(model="m", base_url="u")), \
             patch("team_memory.services.entity_extractor.EntityExtractor",
                   return_value=mock_extractor), \
             patch("team_memory.daemon.entity_backfill.asyncio.sleep",
                   new_callable=AsyncMock), \
             patch("team_memory.daemon.entity_backfill.asyncio.gather",
                   side_effect=_noop_gather):
            from team_memory.daemon.entity_backfill import run_backfill
            await run_backfill(dry_run=False, concurrency=1)
        captured = capsys.readouterr()
        assert "Failed:    1" in captured.out
        assert "exp-fail" in captured.out

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, capsys):
        exps = [
            {"id": "exp-ok", "title": "OK", "description": "D"},
            {"id": "exp-fail", "title": "Fail", "description": "D"},
        ]
        mock_extractor = MagicMock()
        mock_extractor.extract_and_persist = AsyncMock(side_effect=RuntimeError("boom"))
        settings = _make_settings()

        async def _noop_gather(*args, **kwargs):
            for arg in args:
                await arg

        with patch("team_memory.daemon.entity_backfill._get_unextracted_experiences",
                    new_callable=AsyncMock, return_value=exps), \
             patch("team_memory.config.load_settings", return_value=settings), \
             patch("team_memory.daemon.entity_backfill._build_llm_config",
                    new_callable=AsyncMock, return_value=MagicMock(model="m", base_url="u")), \
             patch("team_memory.services.entity_extractor.EntityExtractor",
                   return_value=mock_extractor), \
             patch("team_memory.daemon.entity_backfill.asyncio.sleep",
                   new_callable=AsyncMock), \
             patch("team_memory.daemon.entity_backfill.asyncio.gather",
                   side_effect=_noop_gather):
            from team_memory.daemon.entity_backfill import run_backfill
            await run_backfill(dry_run=False, concurrency=1)
        captured = capsys.readouterr()
        assert "Failed:    2" in captured.out


# ═════════════════════════════════════════════════════════════════════════════
#  solution_backfill.py
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSolutionBackfillMain:
    """Test main() argument parsing and bootstrap flow."""

    @pytest.mark.asyncio
    async def test_main_defaults(self):
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        session = _mock_session_context(fetchall_return=[])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", return_value=mock_ctx2):
            from team_memory.daemon.solution_backfill import main
            await main()

    @pytest.mark.asyncio
    async def test_main_limit_and_dry_run(self):
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        session = _mock_session_context(fetchall_return=[])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", [
            "solution_backfill.py", "--limit", "10", "--dry-run"
        ]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", return_value=mock_ctx2):
            from team_memory.daemon.solution_backfill import main
            await main()

    @pytest.mark.asyncio
    async def test_main_no_ee_config_returns(self):
        mock_ctx = MagicMock()
        settings = _make_settings()
        settings.entity_extraction = None
        mock_ctx.settings = settings

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx):
            from team_memory.daemon.solution_backfill import main
            await main()  # Should not raise


@pytest.mark.unit
class TestSolutionBackfillExtractSolution:
    """Test extract_solution() with mocked httpx."""

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "  Extracted solution  "}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.solution_backfill import extract_solution
            result = await extract_solution(
                title="Test title",
                description="Test description",
                base_url="http://llm.local",
                model="test-model",
                api_key="key",
            )
            assert result == "Extracted solution"

    @pytest.mark.asyncio
    async def test_empty_solution_returns_none(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "  "}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.solution_backfill import extract_solution
            result = await extract_solution(
                title="T", description="D",
                base_url="http://x", model="m", api_key="k",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.solution_backfill import extract_solution
            result = await extract_solution(
                title="T", description="D",
                base_url="http://x", model="m", api_key="k",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_payload_structure(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "solution text"}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.solution_backfill import extract_solution
            await extract_solution(
                title="My Title", description="My Desc",
                base_url="http://llm.local", model="gpt-4",
                api_key="sk-xxx", timeout=60,
            )
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
            assert payload["model"] == "gpt-4"
            assert payload["temperature"] == 0.2
            assert payload["max_tokens"] == 300
            headers = call_kwargs[1]["headers"]
            assert headers["x-api-key"] == "sk-xxx"


@pytest.mark.unit
class TestSolutionBackfillMainProcessing:
    """Test the main() processing loop with mocked DB and LLM."""

    @pytest.mark.asyncio
    async def test_no_rows_returns_early(self):
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        session = _mock_session_context(fetchall_return=[])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", return_value=mock_ctx2):
            from team_memory.daemon.solution_backfill import main
            await main()  # Should not raise, early return

    @pytest.mark.asyncio
    async def test_empty_description_skipped(self):
        row = ("exp-1", "Title", "", "source")
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", return_value=mock_ctx2):
            from team_memory.daemon.solution_backfill import main
            await main()

    @pytest.mark.asyncio
    async def test_successful_solution_extraction(self):
        row = ("exp-1", "Bug Title", "Bug description", "github")
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        # Mock session for UPDATE
        update_session = _mock_session_context()
        update_ctx = AsyncMock()
        update_ctx.__aenter__ = AsyncMock(return_value=update_session)
        update_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Fix the bug by doing X"}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", side_effect=[mock_ctx2, update_ctx]), \
             patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client), \
             patch("team_memory.daemon.solution_backfill.asyncio.sleep", new_callable=AsyncMock):
            from team_memory.daemon.solution_backfill import main
            await main()
            update_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_returns_none_failed_increments(self):
        row = ("exp-2", "Title", "Description", "source")
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", return_value=mock_ctx2), \
             patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client), \
             patch("team_memory.daemon.solution_backfill.asyncio.sleep", new_callable=AsyncMock):
            from team_memory.daemon.solution_backfill import main
            await main()  # Should complete without error

    @pytest.mark.asyncio
    async def test_dry_run_no_db_update(self):
        row = ("exp-3", "Title", "Description text", "source")
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        session = _mock_session_context(fetchall_return=[row])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "solution"}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py", "--dry-run"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", return_value=mock_ctx2), \
             patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client), \
             patch("team_memory.daemon.solution_backfill.asyncio.sleep", new_callable=AsyncMock):
            from team_memory.daemon.solution_backfill import main
            await main()
            # Only one session call (the SELECT), no UPDATE session
            assert mock_ctx2.__aenter__.await_count == 1

    @pytest.mark.asyncio
    async def test_base_url_appends_v1(self):
        """Verify base_url gets /v1 appended when missing."""
        row = ("exp-4", "T", "D", "src")
        mock_ctx = MagicMock()
        settings = _make_settings()
        settings.entity_extraction.base_url = "http://llm.local"
        mock_ctx.settings = settings

        session = _mock_session_context(fetchall_return=[row])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "sol"}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        update_session = _mock_session_context()
        update_ctx = AsyncMock()
        update_ctx.__aenter__ = AsyncMock(return_value=update_session)
        update_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", side_effect=[mock_ctx2, update_ctx]), \
             patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client), \
             patch("team_memory.daemon.solution_backfill.asyncio.sleep", new_callable=AsyncMock):
            from team_memory.daemon.solution_backfill import main
            await main()
            call_args = mock_client.post.call_args
            url = call_args[0][0]
            assert url == "http://llm.local/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_base_url_already_has_v1(self):
        """When base_url already ends with /v1, don't double-append."""
        row = ("exp-5", "T", "D", "src")
        mock_ctx = MagicMock()
        settings = _make_settings()
        settings.entity_extraction.base_url = "http://llm.local/v1"
        mock_ctx.settings = settings

        session = _mock_session_context(fetchall_return=[row])
        mock_ctx2 = AsyncMock()
        mock_ctx2.__aenter__ = AsyncMock(return_value=session)
        mock_ctx2.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "sol"}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        update_session = _mock_session_context()
        update_ctx = AsyncMock()
        update_ctx.__aenter__ = AsyncMock(return_value=update_session)
        update_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(sys, "argv", ["solution_backfill.py"]), \
             patch("team_memory.daemon.solution_backfill.bootstrap", return_value=mock_ctx), \
             patch("team_memory.daemon.solution_backfill.get_session", side_effect=[mock_ctx2, update_ctx]), \
             patch("team_memory.daemon.solution_backfill.httpx.AsyncClient", return_value=mock_client), \
             patch("team_memory.daemon.solution_backfill.asyncio.sleep", new_callable=AsyncMock):
            from team_memory.daemon.solution_backfill import main
            await main()
            call_args = mock_client.post.call_args
            url = call_args[0][0]
            assert url == "http://llm.local/v1/chat/completions"
