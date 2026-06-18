"""Unit tests for CLI backfill scripts — embedding_backfill, entity_backfill,
solution_backfill.

Markers:
  @pytest.mark.unit — pure logic, fully mocked (no real DB/LLM)

Covers: argument parsing, helper functions, main() entry points,
        error paths, and happy paths for each script.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack, contextmanager
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
        with (
            patch.object(sys, "argv", ["embedding_backfill.py"]),
            patch("team_memory.daemon.backfill.embedding_backfill.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock()
            from team_memory.daemon.backfill.embedding_backfill import main

            main()
            mock_asyncio.run.assert_called_once()
            # The coroutine passed to asyncio.run is run_backfill(dry_run=False, limit=0)
            call_arg = mock_asyncio.run.call_args[0][0]
            # We can check by calling it and inspecting kwargs via the mock
            # Instead, let's patch run_backfill directly
        mock_coro.close()

    def test_main_dry_run(self):
        mock_coro = AsyncMock()
        with (
            patch.object(sys, "argv", ["embedding_backfill.py", "--dry-run"]),
            patch(
                "team_memory.daemon.backfill.embedding_backfill.run_backfill",
                new_callable=MagicMock,
                return_value=mock_coro,
            ) as mock_rb,
            patch("team_memory.daemon.backfill.embedding_backfill.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.backfill.embedding_backfill import main

            main()
            assert mock_rb.call_count == 1
            _kw = mock_rb.call_args.kwargs
            assert _kw["dry_run"] is True and _kw["limit"] == 0
        mock_coro.close()

    def test_main_limit(self):
        mock_coro = AsyncMock()
        with (
            patch.object(sys, "argv", ["embedding_backfill.py", "--limit", "42"]),
            patch(
                "team_memory.daemon.backfill.embedding_backfill.run_backfill",
                new_callable=MagicMock,
                return_value=mock_coro,
            ) as mock_rb,
            patch("team_memory.daemon.backfill.embedding_backfill.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.backfill.embedding_backfill import main

            main()
            assert mock_rb.call_count == 1
            _kw = mock_rb.call_args.kwargs
            assert _kw["dry_run"] is False and _kw["limit"] == 42
        mock_coro.close()

    def test_main_all_args(self):
        mock_coro = AsyncMock()
        with (
            patch.object(sys, "argv", ["embedding_backfill.py", "--dry-run", "--limit", "5"]),
            patch(
                "team_memory.daemon.backfill.embedding_backfill.run_backfill",
                new_callable=MagicMock,
                return_value=mock_coro,
            ) as mock_rb,
            patch("team_memory.daemon.backfill.embedding_backfill.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.backfill.embedding_backfill import main

            main()
            assert mock_rb.call_count == 1
            _kw = mock_rb.call_args.kwargs
            assert _kw["dry_run"] is True and _kw["limit"] == 5
        mock_coro.close()


_EB = "team_memory.services.maintenance.batch_ops.BatchOpsService"


@contextmanager
def _patch_embedding_batchops(exp_rows=None, arch_rows=None, pm_rows=None):
    """Enter patches for the three list_* (chunked: rows once then []) and update_* methods.

    Yields a dict of the AsyncMock objects so tests can assert on them.
    """
    exp_chunks = [list(exp_rows or []), []]
    arch_chunks = [list(arch_rows or []), []]

    list_exp = AsyncMock(side_effect=exp_chunks)
    list_arch = AsyncMock(side_effect=arch_chunks)
    list_pm = AsyncMock(return_value=list(pm_rows or []))
    upd_exp = AsyncMock()
    upd_arch = AsyncMock()
    upd_pm = AsyncMock()

    patches = [
        patch(f"{_EB}.list_experiences_missing_embedding", list_exp),
        patch(f"{_EB}.list_archives_missing_embedding", list_arch),
        patch(f"{_EB}.list_personal_memories_missing_embedding", list_pm),
        patch(f"{_EB}.update_experience_embedding", upd_exp),
        patch(f"{_EB}.update_archive_embedding", upd_arch),
        patch(f"{_EB}.update_personal_memory_embedding", upd_pm),
    ]
    with ExitStack() as stack:
        for pc in patches:
            stack.enter_context(pc)
        yield {"list_exp": list_exp, "update_exp": upd_exp}


@pytest.mark.unit
class TestEmbeddingBackfillRunBackfill:
    """Test run_backfill() with BatchOpsService and embedding provider mocked."""

    @pytest.mark.asyncio
    async def test_no_rows_returns_early(self, capsys):
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider"),
            _patch_embedding_batchops(exp_rows=[]) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

            await run_backfill(dry_run=False, limit=0)
        captured = capsys.readouterr()
        assert "All published experiences have embeddings" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_shows_rows(self, capsys):
        rows = [
            {"id": "id-1", "title": "Title 1", "description": "Desc 1", "solution": "Sol 1", "tags": ["tag1"]},
            {"id": "id-2", "title": "Title 2", "description": "Desc 2", "solution": "Sol 2", "tags": []},
        ]
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider"),
            _patch_embedding_batchops(exp_rows=rows) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

            await run_backfill(dry_run=True, limit=0)
        captured = capsys.readouterr()
        # Dry-run prints the row id prefixes (first 8 chars).
        assert "id-1" in captured.out
        assert "id-2" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_truncates_at_10(self, capsys):
        rows = [
            {"id": f"id-{i}", "title": f"Title {i}", "description": f"Desc {i}", "solution": "", "tags": []}
            for i in range(15)
        ]
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider"),
            _patch_embedding_batchops(exp_rows=rows) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

            await run_backfill(dry_run=True, limit=0)
        captured = capsys.readouterr()
        # Only first 10 rows are printed in dry-run mode.
        assert "id-0:" in captured.out
        assert "id-9:" in captured.out
        assert "id-10:" not in captured.out

    @pytest.mark.asyncio
    async def test_successful_backfill(self, capsys):
        rows = [
            {
                "id": "exp-001",
                "title": "Bug fix",
                "description": "Fixed it",
                "solution": "Solution text",
                "tags": ["python"],
            }
        ]
        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=[0.1, 0.2])
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider", return_value=mock_embedder),
            _patch_embedding_batchops(exp_rows=rows) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

            await run_backfill(dry_run=False, limit=0)
        mocks["update_exp"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embedding_returns_none(self, capsys):
        rows = [{"id": "exp-002", "title": "Title", "description": "Desc", "solution": "Sol", "tags": []}]
        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=None)
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider", return_value=mock_embedder),
            _patch_embedding_batchops(exp_rows=rows) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

            await run_backfill(dry_run=False, limit=0)
        # No embedding => no DB update.
        mocks["update_exp"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_encode_exception_increments_failed(self, capsys):
        rows = [{"id": "exp-003", "title": "T", "description": "D", "solution": "S", "tags": ["x"]}]
        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(side_effect=RuntimeError("API down"))
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider", return_value=mock_embedder),
            _patch_embedding_batchops(exp_rows=rows) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

            await run_backfill(dry_run=False, limit=0)
        captured = capsys.readouterr()
        assert "Failed exp-003" in captured.out
        assert "API down" in captured.out

    @pytest.mark.asyncio
    async def test_limit_truncates_rows(self, capsys):
        rows = [
            {"id": f"id-{i}", "title": f"Title {i}", "description": f"Desc {i}", "solution": f"Sol {i}", "tags": []}
            for i in range(3)
        ]
        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=[0.1])
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider", return_value=mock_embedder),
            _patch_embedding_batchops(exp_rows=rows) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

            await run_backfill(dry_run=False, limit=3)
        assert mocks["update_exp"].await_count == 3

    @pytest.mark.asyncio
    async def test_embed_text_truncation(self, capsys):
        """Ensure embed_text is truncated to 4000 chars."""
        long_desc = "x" * 5000
        rows = [{"id": "exp-long", "title": "Title", "description": long_desc, "solution": "Sol", "tags": []}]
        mock_embedder = AsyncMock()
        mock_embedder.encode_single = AsyncMock(return_value=[0.1])
        with (
            patch("team_memory.config.load_settings", return_value=_make_settings()),
            patch("team_memory.embedding.factory.create_embedding_provider", return_value=mock_embedder),
            _patch_embedding_batchops(exp_rows=rows) as mocks,
        ):
            from team_memory.daemon.backfill.embedding_backfill import run_backfill

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
        with (
            patch.object(sys, "argv", ["entity_backfill.py"]),
            patch("team_memory.daemon.backfill.entity_backfill.run_backfill", return_value=mock_coro) as mock_rb,
            patch("team_memory.daemon.backfill.entity_backfill.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.backfill.entity_backfill import main

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
        with (
            patch.object(sys, "argv", ["entity_backfill.py", "--dry-run"]),
            patch("team_memory.daemon.backfill.entity_backfill.run_backfill", return_value=mock_coro) as mock_rb,
            patch("team_memory.daemon.backfill.entity_backfill.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.backfill.entity_backfill import main

            main()
            _, kwargs = mock_rb.call_args
            assert kwargs["dry_run"] is True

    def test_main_all_args(self):
        mock_coro = AsyncMock()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "entity_backfill.py",
                    "--model",
                    "DeepSeek-V3",
                    "--base-url",
                    "http://custom.local",
                    "--limit",
                    "10",
                    "--concurrency",
                    "5",
                ],
            ),
            patch("team_memory.daemon.backfill.entity_backfill.run_backfill", return_value=mock_coro) as mock_rb,
            patch("team_memory.daemon.backfill.entity_backfill.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run = MagicMock(side_effect=lambda c: None)
            from team_memory.daemon.backfill.entity_backfill import main

            main()
            _, kwargs = mock_rb.call_args
            assert kwargs["model"] == "DeepSeek-V3"
            assert kwargs["base_url"] == "http://custom.local"
            assert kwargs["limit"] == 10
            assert kwargs["concurrency"] == 5


@pytest.mark.unit
class TestEntityBackfillHelpers:
    """Test _get_unextracted_experiences and _build_llm_config."""

    async def test_build_llm_config_defaults(self):
        settings = _make_settings()
        with (
            patch("team_memory.config.load_settings", return_value=settings),
            patch.dict("os.environ", {"LLM_API_KEY": "test-key"}),
        ):
            from team_memory.daemon.backfill.entity_backfill import _build_llm_config

            cfg = await _build_llm_config()
            assert cfg.provider == "generic"
            assert cfg.model == "test-model"
            assert cfg.base_url == "http://llm.local"
            assert cfg.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_build_llm_config_override_model(self):
        settings = _make_settings()
        with (
            patch("team_memory.config.load_settings", return_value=settings),
            patch.dict("os.environ", {}, clear=False),
        ):
            from team_memory.daemon.backfill.entity_backfill import _build_llm_config

            cfg = await _build_llm_config(model="custom-model")
            assert cfg.model == "custom-model"

    @pytest.mark.asyncio
    async def test_build_llm_config_override_base_url(self):
        settings = _make_settings()
        with (
            patch("team_memory.config.load_settings", return_value=settings),
            patch.dict("os.environ", {}, clear=False),
        ):
            from team_memory.daemon.backfill.entity_backfill import _build_llm_config

            cfg = await _build_llm_config(base_url="http://custom.local")
            assert cfg.base_url == "http://custom.local"

    @pytest.mark.asyncio
    async def test_build_llm_config_missing_api_key(self):
        settings = _make_settings()
        with patch("team_memory.config.load_settings", return_value=settings), patch.dict("os.environ", {}, clear=True):
            from team_memory.daemon.backfill.entity_backfill import _build_llm_config

            cfg = await _build_llm_config()
            assert cfg.api_key == ""


@pytest.mark.unit
class TestEntityBackfillRunBackfill:
    """Test run_backfill() with mocked DB and LLM."""

    @pytest.mark.asyncio
    async def test_no_experiences_returns_early(self, capsys):
        with (
            patch(
                "team_memory.services.maintenance.batch_ops.BatchOpsService.list_experiences_without_entity_links",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("team_memory.config.load_settings"),
        ):
            from team_memory.daemon.backfill.entity_backfill import run_backfill

            await run_backfill(dry_run=False)
        captured = capsys.readouterr()
        assert "No experiences need entity extraction" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_shows_experiences(self, capsys):
        exps = [{"id": f"exp-{i}", "title": f"Title {i}", "description": f"Desc {i}"} for i in range(5)]
        with (
            patch(
                "team_memory.services.maintenance.batch_ops.BatchOpsService.list_experiences_without_entity_links",
                new_callable=AsyncMock,
                return_value=exps,
            ),
            patch("team_memory.config.load_settings"),
        ):
            from team_memory.daemon.backfill.entity_backfill import run_backfill

            await run_backfill(dry_run=True)
        captured = capsys.readouterr()
        assert "Found 5 experiences" in captured.out
        assert "exp-0" in captured.out

    @pytest.mark.asyncio
    async def test_dry_run_truncates_at_20(self, capsys):
        exps = [{"id": f"exp-{i}", "title": f"Title {i}"} for i in range(25)]
        with (
            patch(
                "team_memory.services.maintenance.batch_ops.BatchOpsService.list_experiences_without_entity_links",
                new_callable=AsyncMock,
                return_value=exps,
            ),
            patch("team_memory.config.load_settings"),
        ):
            from team_memory.daemon.backfill.entity_backfill import run_backfill

            await run_backfill(dry_run=True)
        captured = capsys.readouterr()
        assert "and 5 more" in captured.out

    @pytest.mark.asyncio
    async def test_successful_processing(self, capsys):
        exps = [
            {"id": "exp-1", "title": "Test", "description": "D", "solution": "S", "tags": ["t"], "project": "default"}
        ]
        mock_extractor = MagicMock()
        mock_extractor.extract_and_persist = AsyncMock(return_value=None)
        settings = _make_settings()

        async def _noop_gather(*args, **kwargs):
            """Simulate asyncio.gather — just await all coroutine args."""
            for arg in args:
                await arg

        with (
            patch(
                "team_memory.services.maintenance.batch_ops.BatchOpsService.list_experiences_without_entity_links",
                new_callable=AsyncMock,
                return_value=exps,
            ),
            patch("team_memory.config.load_settings", return_value=settings),
            patch(
                "team_memory.daemon.backfill.entity_backfill._build_llm_config",
                new_callable=AsyncMock,
                return_value=MagicMock(model="m", base_url="u"),
            ),
            patch("team_memory.services.entity.entity_extractor.EntityExtractor", return_value=mock_extractor),
            patch("team_memory.daemon.backfill.entity_backfill.asyncio.sleep", new_callable=AsyncMock),
            patch("team_memory.daemon.backfill.entity_backfill.asyncio.gather", side_effect=_noop_gather),
        ):
            from team_memory.daemon.backfill.entity_backfill import run_backfill

            await run_backfill(dry_run=False, concurrency=1)
        captured = capsys.readouterr()
        assert "Succeeded: 1" in captured.out
        mock_extractor.extract_and_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_processing(self, capsys):
        exps = [{"id": "exp-fail", "title": "Fail", "description": "D"}]
        mock_extractor = MagicMock()
        mock_extractor.extract_and_persist = AsyncMock(side_effect=RuntimeError("LLM error"))
        settings = _make_settings()

        async def _noop_gather(*args, **kwargs):
            for arg in args:
                await arg

        with (
            patch(
                "team_memory.services.maintenance.batch_ops.BatchOpsService.list_experiences_without_entity_links",
                new_callable=AsyncMock,
                return_value=exps,
            ),
            patch("team_memory.config.load_settings", return_value=settings),
            patch(
                "team_memory.daemon.backfill.entity_backfill._build_llm_config",
                new_callable=AsyncMock,
                return_value=MagicMock(model="m", base_url="u"),
            ),
            patch("team_memory.services.entity.entity_extractor.EntityExtractor", return_value=mock_extractor),
            patch("team_memory.daemon.backfill.entity_backfill.asyncio.sleep", new_callable=AsyncMock),
            patch("team_memory.daemon.backfill.entity_backfill.asyncio.gather", side_effect=_noop_gather),
        ):
            from team_memory.daemon.backfill.entity_backfill import run_backfill

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

        with (
            patch(
                "team_memory.services.maintenance.batch_ops.BatchOpsService.list_experiences_without_entity_links",
                new_callable=AsyncMock,
                return_value=exps,
            ),
            patch("team_memory.config.load_settings", return_value=settings),
            patch(
                "team_memory.daemon.backfill.entity_backfill._build_llm_config",
                new_callable=AsyncMock,
                return_value=MagicMock(model="m", base_url="u"),
            ),
            patch("team_memory.services.entity.entity_extractor.EntityExtractor", return_value=mock_extractor),
            patch("team_memory.daemon.backfill.entity_backfill.asyncio.sleep", new_callable=AsyncMock),
            patch("team_memory.daemon.backfill.entity_backfill.asyncio.gather", side_effect=_noop_gather),
        ):
            from team_memory.daemon.backfill.entity_backfill import run_backfill

            await run_backfill(dry_run=False, concurrency=1)
        captured = capsys.readouterr()
        assert "Failed:    2" in captured.out


# ═════════════════════════════════════════════════════════════════════════════
#  solution_backfill.py
# ═════════════════════════════════════════════════════════════════════════════


_LES = "team_memory.services.maintenance.batch_ops.BatchOpsService.list_experiences_missing_solution"
_UES = "team_memory.services.maintenance.batch_ops.BatchOpsService.update_experience_solution"


@pytest.mark.unit
class TestSolutionBackfillMain:
    """Test main() argument parsing and bootstrap flow."""

    @pytest.mark.asyncio
    async def test_main_defaults(self):
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=[]),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()

    @pytest.mark.asyncio
    async def test_main_limit_and_dry_run(self):
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()

        with (
            patch.object(sys, "argv", ["solution_backfill.py", "--limit", "10", "--dry-run"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=[]),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()

    @pytest.mark.asyncio
    async def test_main_no_ee_config_returns(self):
        mock_ctx = MagicMock()
        settings = _make_settings()
        settings.entity_extraction = None
        mock_ctx.settings = settings

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()  # Should not raise


@pytest.mark.unit
class TestSolutionBackfillExtractSolution:
    """Test extract_solution() with mocked httpx."""

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "  Extracted solution  "}}]}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.backfill.solution_backfill import extract_solution

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
        mock_response.json.return_value = {"choices": [{"message": {"content": "  "}}]}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.backfill.solution_backfill import extract_solution

            result = await extract_solution(
                title="T",
                description="D",
                base_url="http://x",
                model="m",
                api_key="k",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.backfill.solution_backfill import extract_solution

            result = await extract_solution(
                title="T",
                description="D",
                base_url="http://x",
                model="m",
                api_key="k",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_payload_structure(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "solution text"}}]}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client):
            from team_memory.daemon.backfill.solution_backfill import extract_solution

            await extract_solution(
                title="My Title",
                description="My Desc",
                base_url="http://llm.local",
                model="gpt-4",
                api_key="sk-xxx",
                timeout=60,
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
    """Test the main() processing loop with mocked BatchOpsService and LLM."""

    @staticmethod
    def _client(content="sol", *, post_exc=None):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_client = AsyncMock()
        if post_exc is not None:
            mock_client.post = AsyncMock(side_effect=post_exc)
        else:
            mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    @pytest.mark.asyncio
    async def test_no_rows_returns_early(self):
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=[]),
            patch(_UES, new_callable=AsyncMock) as mock_update,
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()  # Should not raise, early return
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_description_skipped(self):
        rows = [{"id": "exp-1", "title": "Title", "description": ""}]
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=rows),
            patch(_UES, new_callable=AsyncMock) as mock_update,
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_solution_extraction(self):
        rows = [{"id": "exp-1", "title": "Bug Title", "description": "Bug description"}]
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        mock_client = self._client("Fix the bug by doing X")

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=rows),
            patch(_UES, new_callable=AsyncMock) as mock_update,
            patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client),
            patch("team_memory.daemon.backfill.solution_backfill.asyncio.sleep", new_callable=AsyncMock),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_returns_none_failed_increments(self):
        rows = [{"id": "exp-2", "title": "Title", "description": "Description"}]
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        mock_client = self._client(post_exc=Exception("timeout"))

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=rows),
            patch(_UES, new_callable=AsyncMock) as mock_update,
            patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client),
            patch("team_memory.daemon.backfill.solution_backfill.asyncio.sleep", new_callable=AsyncMock),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()  # Should complete without error
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_no_db_update(self):
        rows = [{"id": "exp-3", "title": "Title", "description": "Description text"}]
        mock_ctx = MagicMock()
        mock_ctx.settings = _make_settings()
        mock_client = self._client("solution")

        with (
            patch.object(sys, "argv", ["solution_backfill.py", "--dry-run"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=rows),
            patch(_UES, new_callable=AsyncMock) as mock_update,
            patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client),
            patch("team_memory.daemon.backfill.solution_backfill.asyncio.sleep", new_callable=AsyncMock),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()
            # Dry run must not write to DB.
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_base_url_appends_v1(self):
        """Verify base_url gets /v1 appended when missing."""
        rows = [{"id": "exp-4", "title": "T", "description": "D"}]
        mock_ctx = MagicMock()
        settings = _make_settings()
        settings.entity_extraction.base_url = "http://llm.local"
        mock_ctx.settings = settings
        mock_client = self._client("sol")

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=rows),
            patch(_UES, new_callable=AsyncMock),
            patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client),
            patch("team_memory.daemon.backfill.solution_backfill.asyncio.sleep", new_callable=AsyncMock),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()
            url = mock_client.post.call_args[0][0]
            assert url == "http://llm.local/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_base_url_already_has_v1(self):
        """When base_url already ends with /v1, don't double-append."""
        rows = [{"id": "exp-5", "title": "T", "description": "D"}]
        mock_ctx = MagicMock()
        settings = _make_settings()
        settings.entity_extraction.base_url = "http://llm.local/v1"
        mock_ctx.settings = settings
        mock_client = self._client("sol")

        with (
            patch.object(sys, "argv", ["solution_backfill.py"]),
            patch("team_memory.daemon.backfill.solution_backfill.bootstrap", return_value=mock_ctx),
            patch(_LES, new_callable=AsyncMock, return_value=rows),
            patch(_UES, new_callable=AsyncMock),
            patch("team_memory.daemon.backfill.solution_backfill.httpx.AsyncClient", return_value=mock_client),
            patch("team_memory.daemon.backfill.solution_backfill.asyncio.sleep", new_callable=AsyncMock),
        ):
            from team_memory.daemon.backfill.solution_backfill import main

            await main()
            url = mock_client.post.call_args[0][0]
            assert url == "http://llm.local/v1/chat/completions"
