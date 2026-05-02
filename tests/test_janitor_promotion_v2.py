"""Tests for Janitor run_promotion with Markdown compilation + Obsidian write + git commit.

Covers:
- Promoted Experience is compiled to Markdown via PromotionCompiler
- Markdown file written to configured output directory
- git add + git commit executed
- No output directory configured → skip file write
- Output directory auto-created if missing
- Experience status set to 'promoted' after compilation
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.config.janitor import JanitorConfig
from team_memory.services.janitor import MemoryJanitor


# ============================================================
# Helpers
# ============================================================


def _make_experience(
    *,
    id: str | None = None,
    project: str = "default",
    group_key: str | None = None,
    recall_count: int = 0,
    exp_status: str = "published",
    title: str = "Test Experience",
    description: str = "A problem description",
    solution: str = "A solution",
    tags: list[str] | None = None,
    is_deleted: bool = False,
) -> MagicMock:
    """Build a mock Experience ORM object."""
    exp = MagicMock()
    exp.id = uuid.UUID(id) if id else uuid.uuid4()
    exp.project = project
    exp.group_key = group_key
    exp.recall_count = recall_count
    exp.exp_status = exp_status
    exp.title = title
    exp.description = description
    exp.solution = solution
    exp.tags = tags or ["test"]
    exp.is_deleted = is_deleted
    exp.created_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    # to_dict must return a dict usable by PromotionCompiler
    exp.to_dict = MagicMock(return_value={
        "id": str(exp.id),
        "title": title,
        "description": description,
        "solution": solution,
        "tags": tags or ["test"],
        "group_key": group_key,
        "project": project,
        "recall_count": recall_count,
        "created_at": "2026-01-15T10:00:00+00:00",
    })
    return exp


@pytest.fixture
def mock_session():
    """AsyncMock session."""
    s = AsyncMock()
    s.commit = AsyncMock()
    return s


def _patch_session(mock_session):
    """Return a patch for team_memory.storage.database.get_session that yields mock_session."""
    @asynccontextmanager
    async def _cm(_url):
        yield mock_session
    return patch("team_memory.storage.database.get_session", _cm)


@pytest.fixture
def janitor_no_config():
    """Janitor with no config (no output dirs)."""
    return MemoryJanitor(db_url="sqlite+aiosqlite://", config=None)


@pytest.fixture
def janitor_with_dirs():
    """Janitor configured with promotion_output_dirs."""
    config = JanitorConfig(
        promotion_output_dirs={"default": "/tmp/obsidian/default"},
    )
    return MemoryJanitor(db_url="sqlite+aiosqlite://", config=config)


def _setup_execute(mock_session, *, recall_count_exps=None, qualifying_groups=None, group_exps=None):
    """Set up mock_session.execute with side_effect for the three queries."""
    # recall_count query result
    uc_result = MagicMock()
    uc_result.scalars.return_value.all.return_value = recall_count_exps or []

    # group count query result
    gc_result = MagicMock()
    gc_result.all.return_value = qualifying_groups or []

    side_effects = [uc_result, gc_result]

    # group exp query result (only if qualifying groups)
    if qualifying_groups:
        ge_result = MagicMock()
        ge_result.scalars.return_value.all.return_value = group_exps or []
        side_effects.append(ge_result)

    mock_session.execute = AsyncMock(side_effect=side_effects)


# ============================================================
# Tests
# ============================================================


class TestPromotionMarkdownCompilation:
    """Promoted experiences are compiled to Markdown via PromotionCompiler."""

    @pytest.mark.asyncio
    async def test_compiler_called_for_use_count_promotion(self, janitor_with_dirs, mock_session):
        """PromotionCompiler.compile is called for each use_count-promoted exp."""
        exp = _make_experience(recall_count=5, group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess"),
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown output")
            MockCompiler.return_value = mock_compiler_instance

            result = await janitor_with_dirs.run_promotion()

        # Compiler should have been called with the experience dict and group_key
        mock_compiler_instance.compile.assert_called_once()
        call_args = mock_compiler_instance.compile.call_args
        experiences_arg = call_args[0][0]
        assert len(experiences_arg) == 1
        assert experiences_arg[0]["title"] == "Test Experience"

    @pytest.mark.asyncio
    async def test_compiler_called_for_group_key_promotion(self, janitor_with_dirs, mock_session):
        """PromotionCompiler.compile is called for group_key-promoted exps."""
        exp1 = _make_experience(id="00000000-0000-0000-0000-000000000001", group_key="g1", recall_count=0)
        exp2 = _make_experience(id="00000000-0000-0000-0000-000000000002", group_key="g1", recall_count=0)

        _setup_execute(
            mock_session,
            qualifying_groups=[("g1",)],
            group_exps=[exp1, exp2],
        )

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess"),
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Group Markdown")
            MockCompiler.return_value = mock_compiler_instance

            result = await janitor_with_dirs.run_promotion()

        mock_compiler_instance.compile.assert_called_once()
        call_args = mock_compiler_instance.compile.call_args
        experiences_arg = call_args[0][0]
        assert len(experiences_arg) == 2


class TestPromotionFileWrite:
    """Markdown files are written to configured output directory."""

    @pytest.mark.asyncio
    async def test_file_written_to_output_dir(self, mock_session, tmp_path):
        """Markdown is written to the configured Obsidian directory."""
        output_dir = str(tmp_path / "obsidian")
        config = JanitorConfig(
            promotion_output_dirs={"default": output_dir},
        )
        janitor = MemoryJanitor(db_url="sqlite+aiosqlite://", config=config)

        exp = _make_experience(recall_count=5, group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess") as mock_subprocess,
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Promoted Markdown")
            MockCompiler.return_value = mock_compiler_instance
            mock_subprocess.run = MagicMock()

            result = await janitor.run_promotion()

        # Check that a file was created in output_dir
        import os
        files = os.listdir(output_dir)
        assert len(files) == 1
        assert files[0].startswith("promoted-")
        assert files[0].endswith(".md")

        with open(os.path.join(output_dir, files[0])) as f:
            content = f.read()
        assert content == "# Promoted Markdown"

    @pytest.mark.asyncio
    async def test_output_dir_auto_created(self, mock_session, tmp_path):
        """Non-existent output directory is created automatically."""
        output_dir = str(tmp_path / "new_dir" / "obsidian")
        config = JanitorConfig(
            promotion_output_dirs={"default": output_dir},
        )
        janitor = MemoryJanitor(db_url="sqlite+aiosqlite://", config=config)

        exp = _make_experience(recall_count=5, group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess") as mock_subprocess,
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown")
            MockCompiler.return_value = mock_compiler_instance
            mock_subprocess.run = MagicMock()

            result = await janitor.run_promotion()

        import os
        assert os.path.isdir(output_dir)


class TestPromotionNoOutputDir:
    """When no output directory is configured, file writing is skipped."""

    @pytest.mark.asyncio
    async def test_no_file_write_without_config(self, janitor_no_config, mock_session, tmp_path):
        """No file is written when promotion_output_dirs is empty."""
        exp = _make_experience(recall_count=5, group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        written_files = []

        original_open = open

        def tracking_open(*args, **kwargs):
            if args and isinstance(args[0], str) and "promoted-" in args[0]:
                written_files.append(args[0])
            return original_open(*args, **kwargs)

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("builtins.open", side_effect=tracking_open),
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown")
            MockCompiler.return_value = mock_compiler_instance

            result = await janitor_no_config.run_promotion()

        assert len(written_files) == 0

    @pytest.mark.asyncio
    async def test_no_file_write_for_unmapped_project(self, mock_session):
        """No file written when the project has no mapping in promotion_output_dirs."""
        config = JanitorConfig(
            promotion_output_dirs={"other-project": "/tmp/obsidian/other"},
        )
        janitor = MemoryJanitor(db_url="sqlite+aiosqlite://", config=config)

        exp = _make_experience(recall_count=5, project="unmapped", group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        written_files = []

        original_open = open

        def tracking_open(*args, **kwargs):
            if args and isinstance(args[0], str) and "promoted-" in args[0]:
                written_files.append(args[0])
            return original_open(*args, **kwargs)

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("builtins.open", side_effect=tracking_open),
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown")
            MockCompiler.return_value = mock_compiler_instance

            result = await janitor.run_promotion()

        assert len(written_files) == 0


class TestPromotionGitCommit:
    """git add + git commit executed after file write."""

    @pytest.mark.asyncio
    async def test_git_add_and_commit_called(self, mock_session, tmp_path):
        """git add and git commit are called after writing the promoted file."""
        output_dir = str(tmp_path / "obsidian")
        config = JanitorConfig(
            promotion_output_dirs={"default": output_dir},
        )
        janitor = MemoryJanitor(db_url="sqlite+aiosqlite://", config=config)

        exp = _make_experience(recall_count=5, group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess") as mock_subprocess,
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown")
            MockCompiler.return_value = mock_compiler_instance
            mock_subprocess.run = MagicMock()

            result = await janitor.run_promotion()

        # subprocess.run should have been called for git add and git commit
        calls = mock_subprocess.run.call_args_list
        assert len(calls) == 2

        # git add
        add_call = calls[0]
        assert add_call[0][0][0] == "git"
        assert add_call[0][0][1] == "add"
        assert "promoted-" in add_call[0][0][2]

        # git commit
        commit_call = calls[1]
        assert commit_call[0][0][0] == "git"
        assert commit_call[0][0][1] == "commit"
        assert "-m" in commit_call[0][0]

    @pytest.mark.asyncio
    async def test_git_not_called_without_output_dir(self, janitor_no_config, mock_session):
        """git commands are not called when no output directory configured."""
        exp = _make_experience(recall_count=5, group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess") as mock_subprocess,
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown")
            MockCompiler.return_value = mock_compiler_instance
            mock_subprocess.run = MagicMock()

            result = await janitor_no_config.run_promotion()

        mock_subprocess.run.assert_not_called()


class TestPromotionStatusUpdate:
    """Experience is marked as promoted after compilation."""

    @pytest.mark.asyncio
    async def test_exp_status_set_to_promoted(self, janitor_with_dirs, mock_session):
        """exp_status is set to 'promoted' after Markdown compilation."""
        exp = _make_experience(recall_count=5, group_key="g1")
        _setup_execute(mock_session, recall_count_exps=[exp])

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess"),
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown")
            MockCompiler.return_value = mock_compiler_instance

            result = await janitor_with_dirs.run_promotion()

        assert exp.exp_status == "promoted"
        assert result["promoted_by_recall_count"] == 1
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_return_stats_include_total(self, janitor_with_dirs, mock_session):
        """Return stats include promoted_by_recall_count, promoted_by_group, total."""
        exp1 = _make_experience(id="00000000-0000-0000-0000-000000000001", recall_count=5, group_key="g1")
        exp2 = _make_experience(id="00000000-0000-0000-0000-000000000002", group_key="g1", recall_count=0)

        _setup_execute(
            mock_session,
            recall_count_exps=[exp1],
            qualifying_groups=[("g1",)],
            group_exps=[exp2],
        )

        with (
            _patch_session(mock_session),
            patch("team_memory.services.janitor.PromotionCompiler") as MockCompiler,
            patch("team_memory.services.janitor.subprocess"),
        ):
            mock_compiler_instance = AsyncMock()
            mock_compiler_instance.compile = AsyncMock(return_value="# Markdown")
            MockCompiler.return_value = mock_compiler_instance

            result = await janitor_with_dirs.run_promotion()

        assert result["promoted_by_recall_count"] == 1
        assert result["promoted_by_group"] == 1
        assert result["total"] == 2
