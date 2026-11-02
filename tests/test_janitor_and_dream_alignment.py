"""Unit tests for MemoryJanitor and AutoDreamWorker bi-temporal & utility alignment."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from team_memory.config.janitor import JanitorConfig
from team_memory.daemon.integration.auto_dream import AutoDreamWorker
from team_memory.daemon.runtime.config import AutoDreamSettings, DaemonConfig
from team_memory.services.maintenance.janitor import MemoryJanitor


def _make_experience(
    *,
    exp_status: str = "published",
    recall_count: int = 0,
    used_count: int = 0,
    negative_feedback_count: int = 0,
    valid_until: datetime | None = None,
    group_key: str | None = None,
    is_deleted: bool = False,
    exp_id: uuid.UUID | None = None,
    project: str = "default",
    title: str = "Test Experience",
    description: str = "Test description",
    solution: str = "Test solution",
    tags: list[str] | None = None,
) -> MagicMock:
    """Create a mock Experience object with bi-temporal and utility fields."""
    exp = MagicMock()
    exp.id = exp_id or uuid.uuid4()
    exp.exp_status = exp_status
    exp.recall_count = recall_count
    exp.used_count = used_count
    exp.negative_feedback_count = negative_feedback_count
    exp.valid_until = valid_until
    exp.group_key = group_key
    exp.is_deleted = is_deleted
    exp.project = project
    exp.title = title
    exp.description = description
    exp.solution = solution
    exp.tags = tags or []
    exp.created_at = "2026-04-23T00:00:00"
    exp.to_dict = lambda: {
        "id": str(exp.id),
        "title": title,
        "description": description,
        "solution": solution,
        "tags": tags or [],
        "project": project,
        "group_key": group_key,
        "created_at": exp.created_at,
        "exp_status": exp_status,
        "recall_count": recall_count,
        "used_count": used_count,
        "negative_feedback_count": negative_feedback_count,
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


def _make_dream_worker(**overrides) -> tuple[AutoDreamWorker, AsyncMock]:
    defaults = dict(
        enabled=True,
        min_hours_since_last=24,
        min_new_experiences=5,
        merge_similarity_threshold=0.85,
        prune_max_age_days=30,
        scan_interval_minutes=1,
    )
    defaults.update(overrides)
    cfg = DaemonConfig()
    cfg.auto_dream = AutoDreamSettings(**defaults)
    sink = AsyncMock()
    sink.recall = AsyncMock(return_value=[])
    sink.update_experience = AsyncMock(return_value={"id": "exp-1", "status": "published"})
    return AutoDreamWorker(config=cfg, sink=sink), sink


class TestJanitorPromotionAlignment:
    """Tests for MemoryJanitor.run_promotion() bi-temporal & utility filtering."""

    @pytest.mark.asyncio
    async def test_promotion_skips_superseded_and_negative_feedback_experiences(self):
        """Superseded experiences (valid_until set) and negatively rated experiences are not promoted."""
        superseded_exp = _make_experience(
            recall_count=10,
            used_count=5,
            valid_until=datetime.now(timezone.utc),
        )
        negative_exp = _make_experience(
            recall_count=10,
            used_count=5,
            negative_feedback_count=1,
        )
        valid_exp = _make_experience(
            recall_count=4,
            used_count=2,
            negative_feedback_count=0,
            valid_until=None,
        )

        janitor = MemoryJanitor(
            db_url="sqlite+aiosqlite://",
            config=JanitorConfig(promotion_use_count_threshold=3),
        )

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                MagicMock(
                    scalars=MagicMock(
                        return_value=_FakeScalarResult([superseded_exp, negative_exp, valid_exp])
                    )
                ),
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        assert superseded_exp.exp_status == "published"
        assert negative_exp.exp_status == "published"
        assert valid_exp.exp_status == "promoted"
        assert result["promoted_by_recall_count"] == 1
        assert result["promoted_by_used_count"] == 1
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_promotion_require_used_filters_zero_adoption_noise(self):
        """When promotion_require_used=True, used_count >= threshold is required."""
        adopted_exp = _make_experience(used_count=3, recall_count=3)
        noise_exp = _make_experience(used_count=0, recall_count=10)

        janitor = MemoryJanitor(
            db_url="sqlite+aiosqlite://",
            config=JanitorConfig(
                promotion_use_count_threshold=3,
                promotion_require_used=True,
            ),
        )

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            side_effect=[
                MagicMock(
                    scalars=MagicMock(return_value=_FakeScalarResult([adopted_exp, noise_exp]))
                ),
                MagicMock(all=MagicMock(return_value=[])),
            ]
        )
        mock_session.commit = AsyncMock()

        with patch.object(janitor, "_session", lambda: _mock_session_ctx(mock_session)):
            result = await janitor.run_promotion()

        assert adopted_exp.exp_status == "promoted"
        assert noise_exp.exp_status == "published"
        assert result["promoted_by_recall_count"] == 1
        assert result["promoted_by_used_count"] == 1
        assert result["total"] == 1


class TestAutoDreamAlignment:
    """Tests for AutoDreamWorker utility tiebreaker, stale pruning, and active filtering."""

    @pytest.mark.asyncio
    async def test_merge_cluster_prioritizes_used_count_on_quality_tie(self):
        """When quality_score is tied, experience with higher used_count is chosen as primary."""
        worker, sink = _make_dream_worker()

        cluster = [
            {
                "id": "high-recall-zero-use",
                "title": "High Recall Zero Use",
                "solution": "Solution A",
                "quality_score": 50.0,
                "used_count": 0,
                "recall_count": 20,
                "tags": ["a"],
            },
            {
                "id": "high-use-low-recall",
                "title": "High Use Low Recall",
                "solution": "Solution B",
                "quality_score": 50.0,
                "used_count": 4,
                "recall_count": 4,
                "tags": ["b"],
            },
        ]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_delete:
            merged = await worker._merge_cluster(cluster)

        assert merged == 1
        sink.update_experience.assert_called_once()
        call_kwargs = sink.update_experience.call_args[1]
        assert call_kwargs["experience_id"] == "high-use-low-recall"
        mock_delete.assert_called_once_with("high-recall-zero-use")

    @pytest.mark.asyncio
    async def test_prune_stale_preserves_used_experiences(self):
        """Experiences with used_count > 0 are never pruned even when recall_count == 0."""
        worker, _ = _make_dream_worker(prune_max_age_days=30)
        old_ts = datetime.now(timezone.utc) - timedelta(days=60)

        mock_ctx = MagicMock()
        mock_ctx.service.list_published = AsyncMock(
            return_value=[
                {
                    "id": "used-exp",
                    "recall_count": 0,
                    "used_count": 2,
                    "is_pinned": False,
                    "created_at": old_ts,
                },
                {
                    "id": "stale-exp",
                    "recall_count": 0,
                    "used_count": 0,
                    "is_pinned": False,
                    "created_at": old_ts,
                },
            ]
        )
        mock_ctx.service.soft_delete_by_id = AsyncMock()

        with patch("team_memory.daemon.integration.auto_dream._daemon_context", return_value=mock_ctx):
            pruned = await worker._prune_stale()

        assert pruned == 1
        mock_ctx.service.soft_delete_by_id.assert_called_once_with("stale-exp")
        called_tiers = [
            call.kwargs.get("quality_tier")
            for call in mock_ctx.service.list_published.call_args_list
        ]
        assert called_tiers == ["Bronze", "Outdated"]

    @pytest.mark.asyncio
    async def test_get_dream_candidates_filters_superseded(self):
        """_get_dream_candidates filters out candidates where valid_until is not None."""
        worker, _ = _make_dream_worker()
        mock_ctx = MagicMock()
        mock_ctx.service.list_published = AsyncMock(
            return_value=[
                {"id": "active-1", "valid_until": None},
                {"id": "superseded-1", "valid_until": "2026-04-01T00:00:00+00:00"},
            ]
        )

        with patch("team_memory.daemon.integration.auto_dream._daemon_context", return_value=mock_ctx):
            candidates = await worker._get_dream_candidates(None)

        assert [c["id"] for c in candidates] == ["active-1"]
