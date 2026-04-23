"""Tests for Janitor promotion task (L2→L3 auto-promotion).

Covers:
- use_count >= threshold triggers promotion
- use_count below threshold does NOT trigger promotion
- Same group_key with count >= threshold triggers promotion
- Same group_key below threshold does NOT trigger promotion
- Already-promoted Experiences are excluded (no double promotion)
- run_all includes promotion in results
- promotion_enabled=False skips promotion
- JanitorConfig has new fields with correct defaults
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.config.janitor import JanitorConfig
from team_memory.services.janitor import MemoryJanitor

# ============================================================
# Helpers
# ============================================================


def _make_experience(
    *,
    exp_status: str = "published",
    use_count: int = 0,
    group_key: str | None = None,
    is_deleted: bool = False,
    exp_id: uuid.UUID | None = None,
    project: str = "default",
    title: str = "Test Experience",
    description: str = "Test description",
    solution: str = "Test solution",
    tags: list[str] | None = None,
    created_at: str = "2026-04-23T00:00:00",
) -> MagicMock:
    """Create a mock Experience object."""
    exp = MagicMock()
    exp.id = exp_id or uuid.uuid4()
    exp.exp_status = exp_status
    exp.use_count = use_count
    exp.group_key = group_key
    exp.is_deleted = is_deleted
    exp.project = project
    exp.title = title
    exp.description = description
    exp.solution = solution
    exp.tags = tags or []
    exp.created_at = created_at
    exp.to_dict = lambda: {
        "id": str(exp.id),
        "title": title,
        "description": description,
        "solution": solution,
        "tags": tags or [],
        "project": project,
        "group_key": group_key,
        "created_at": created_at,
        "exp_status": exp_status,
    }
    return exp


class _FakeScalarResult:
    """Mimics the result of scalars().all()."""

    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items

    def scalars(self):
        return self


@asynccontextmanager
async def _mock_session_ctx(mock_session):
    yield mock_session


def _setup_janitor(config=None):
    """Create a MemoryJanitor with a mocked _session."""
    janitor = MemoryJanitor(db_url="sqlite+aiosqlite://", config=config)
    return janitor


# ============================================================
# Test: JanitorConfig new fields with correct defaults
# ============================================================


class TestJanitorConfigPromotionFields:
    """Test JanitorConfig has promotion-related fields with correct defaults."""

    def test_default_promotion_enabled(self):
        cfg = JanitorConfig()
        assert cfg.promotion_enabled is True

    def test_default_promotion_use_count_threshold(self):
        cfg = JanitorConfig()
        assert cfg.promotion_use_count_threshold == 3

    def test_default_promotion_group_key_threshold(self):
        cfg = JanitorConfig()
        assert cfg.promotion_group_key_threshold == 5

    def test_custom_promotion_values(self):
        cfg = JanitorConfig(
            promotion_enabled=False,
            promotion_use_count_threshold=10,
            promotion_group_key_threshold=8,
        )
        assert cfg.promotion_enabled is False
        assert cfg.promotion_use_count_threshold == 10
        assert cfg.promotion_group_key_threshold == 8


# ============================================================
# Test: run_promotion logic
# ============================================================


class TestRunPromotion:
    """Tests for the run_promotion() method."""

    @pytest.mark.asyncio
    async def test_use_count_above_threshold_gets_promoted(self):
        """Published experience with use_count >= 3 should be promoted."""
        exp = _make_experience(use_count=3, exp_status="published")
        janitor = _setup_janitor(JanitorConfig(promotion_use_count_threshold=3))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: returns 1 experience
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult([exp]))),
                # 2. group_key count query: no qualifying groups
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        assert exp.exp_status == "promoted"
        assert result["promoted_by_use_count"] == 1
        assert result["promoted_by_group"] == 0
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_use_count_below_threshold_not_promoted(self):
        """Published experience with use_count=2 should NOT be promoted."""
        exp = _make_experience(use_count=2, exp_status="published")
        janitor = _setup_janitor(JanitorConfig(promotion_use_count_threshold=3))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: empty (no one meets threshold)
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult([]))),
                # 2. group_key count query: no qualifying groups
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        assert exp.exp_status == "published"
        assert result["promoted_by_use_count"] == 0
        assert result["promoted_by_group"] == 0
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_group_key_above_threshold_gets_promoted(self):
        """5+ published experiences with the same group_key should all be promoted."""
        group_key = "python-error-handling"
        exps = [
            _make_experience(use_count=0, exp_status="published", group_key=group_key)
            for _ in range(5)
        ]
        janitor = _setup_janitor(JanitorConfig(promotion_group_key_threshold=5))

        mock_session = AsyncMock()
        # Row type for group count query
        group_row = MagicMock()
        group_row.__getitem__ = lambda self_, idx: group_key
        group_row_0 = (group_key,)

        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: none
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult([]))),
                # 2. group_key count query: returns qualifying group_key
                MagicMock(all=MagicMock(return_value=[group_row_0])),
                # 3. group_key experiences query: returns 5 experiences
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult(exps))),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        for exp in exps:
            assert exp.exp_status == "promoted"
        assert result["promoted_by_use_count"] == 0
        assert result["promoted_by_group"] == 5
        assert result["total"] == 5

    @pytest.mark.asyncio
    async def test_group_key_below_threshold_not_promoted(self):
        """4 published experiences with the same group_key should NOT be promoted."""
        group_key = "python-error-handling"
        exps = [
            _make_experience(use_count=0, exp_status="published", group_key=group_key)
            for _ in range(4)
        ]
        janitor = _setup_janitor(JanitorConfig(promotion_group_key_threshold=5))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: empty
                MagicMock(scalars=MagicMock(return_value=_FakeScalarResult([]))),
                # 2. group_key count query: no qualifying groups (4 < 5)
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        for exp in exps:
            assert exp.exp_status == "published"
        assert result["promoted_by_use_count"] == 0
        assert result["promoted_by_group"] == 0
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_already_promoted_excluded(self):
        """Already-promoted experiences should NOT be double-promoted."""
        exp_promoted = _make_experience(use_count=5, exp_status="promoted")
        exp_published = _make_experience(use_count=3, exp_status="published")

        janitor = _setup_janitor(JanitorConfig(promotion_use_count_threshold=3))

        mock_session = AsyncMock()
        # use_count query returns only the published one (promoted excluded by query)
        # group_key query returns nothing
        mock_session.execute = AsyncMock(
            side_effect=[
                # 1. use_count query: returns only the published one
                MagicMock(
                    scalars=MagicMock(return_value=_FakeScalarResult([exp_published]))
                ),
                # 2. group_key count query: no qualifying groups
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        assert exp_promoted.exp_status == "promoted"  # unchanged
        assert exp_published.exp_status == "promoted"  # newly promoted
        assert result["promoted_by_use_count"] == 1
        assert result["total"] == 1


# ============================================================
# Test: run_all includes promotion
# ============================================================


class TestRunAllPromotion:
    """Test that run_all includes promotion in results."""

    @pytest.mark.asyncio
    async def test_run_all_includes_promotion(self):
        """run_all should include promotion in results when enabled."""
        janitor = _setup_janitor(JanitorConfig(promotion_enabled=True))

        # Mock all methods
        janitor.run_score_decay = AsyncMock(return_value={"updated_count": 0})
        janitor.sweep_outdated = AsyncMock(
            return_value={"found_count": 0, "deleted_count": 0}
        )
        janitor.purge_soft_deleted = AsyncMock(return_value={"purged_count": 0})
        janitor.expire_drafts = AsyncMock(return_value={"expired_count": 0})
        janitor.prune_personal_memory = AsyncMock(return_value={"pruned_count": 0})
        janitor.run_promotion = AsyncMock(
            return_value={
                "promoted_by_use_count": 2,
                "promoted_by_group": 3,
                "total": 5,
            }
        )

        result = await janitor.run_all()

        assert "promotion" in result["operations"]
        assert result["operations"]["promotion"]["total"] == 5
        janitor.run_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_promotion_disabled_skips_promotion(self):
        """run_all should NOT run promotion when promotion_enabled=False."""
        janitor = _setup_janitor(JanitorConfig(promotion_enabled=False))

        janitor.run_score_decay = AsyncMock(return_value={"updated_count": 0})
        janitor.sweep_outdated = AsyncMock(
            return_value={"found_count": 0, "deleted_count": 0}
        )
        janitor.purge_soft_deleted = AsyncMock(return_value={"purged_count": 0})
        janitor.expire_drafts = AsyncMock(return_value={"expired_count": 0})
        janitor.prune_personal_memory = AsyncMock(return_value={"pruned_count": 0})
        janitor.run_promotion = AsyncMock(
            return_value={
                "promoted_by_use_count": 0,
                "promoted_by_group": 0,
                "total": 0,
            }
        )

        result = await janitor.run_all()

        assert "promotion" not in result["operations"]
        janitor.run_promotion.assert_not_called()
