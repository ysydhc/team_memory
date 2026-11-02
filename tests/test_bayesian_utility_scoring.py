"""Unit tests for Phase 1A: Eliminate Matthew Effect & Bayesian Utility Scoring Loop."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.memory.experience import ExperienceService
from team_memory.services.search.scoring import apply_popularity_boost, compute_bayesian_utility_rate
from team_memory.services.search.search_orchestrator import SearchOrchestrator
from team_memory.services.search.search_pipeline import SearchResultItem


def _make_item(
    exp_id: str,
    score: float,
    recall_count: int = 0,
    used_count: int = 0,
    negative_feedback_count: int = 0,
    updated_at: str | None = None,
    created_at: str | None = None,
) -> SearchResultItem:
    """Create a SearchResultItem with popularity and feedback fields."""
    now_iso = datetime.now(timezone.utc).isoformat()
    data = {
        "id": exp_id,
        "title": f"Experience {exp_id}",
        "recall_count": recall_count,
        "used_count": used_count,
        "negative_feedback_count": negative_feedback_count,
        "updated_at": updated_at or now_iso,
        "created_at": created_at or now_iso,
    }
    return SearchResultItem(data=data, score=score, similarity=score)


class TestSearchOrchestratorNoMatthewEffect:
    """Verify SearchOrchestrator does not inflate quality_score on mere recall."""

    @pytest.mark.asyncio
    async def test_search_increments_recall_count_only_and_forwards_telemetry(self):
        exp_id = str(uuid.uuid4())
        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.results = [
            {"id": exp_id, "title": "Recalled Hit", "score": 0.85, "status": "published"},
        ]
        mock_pipeline_result.search_type = "hybrid"
        mock_pipeline_result.reranked = True
        mock_pipeline_result.cached = False
        mock_pipeline_result.stage_metrics = {"vector_ms": 10, "fts_ms": 5, "total_ms": 18}
        mock_pipeline_result.stage_counts = {"vector": 8, "fts": 4, "final": 1}
        mock_pipeline.search = AsyncMock(return_value=mock_pipeline_result)

        orchestrator = SearchOrchestrator(
            search_pipeline=mock_pipeline,
            embedding_provider=MagicMock(),
            db_url="sqlite+aiosqlite://",
        )

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_recall_count = AsyncMock()
        mock_repo.increment_quality_score = AsyncMock()

        with (
            patch("team_memory.storage.database.get_session") as mock_gs,
            patch(
                "team_memory.services.search.search_orchestrator.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            res = await orchestrator.search(query="fix connection leak", max_results=5)

        mock_repo.increment_recall_count.assert_awaited_once_with(uuid.UUID(exp_id))
        mock_repo.increment_quality_score.assert_not_called()
        assert res.stage_metrics == {"vector_ms": 10, "fts_ms": 5, "total_ms": 18}
        assert res.stage_counts == {"vector": 8, "fts": 4, "final": 1}


class TestBayesianUtilityScoring:
    """Verify Bayesian utility rate and anti-noise popularity boost."""

    def test_bayesian_utility_rate_prior_and_smoothing(self):
        # Prior with 0 recall, 0 used, 0 neg -> 1 / (1 + 4) = 0.2
        assert compute_bayesian_utility_rate(0, 0, 0) == pytest.approx(0.2)
        # High recall, 0 used -> drops well below prior
        assert compute_bayesian_utility_rate(30, 0, 0) == pytest.approx(1.0 / 35.0)
        # High adoption (3 recall, 3 used) -> (1 + 3) / (5 + 3) = 0.5
        assert compute_bayesian_utility_rate(3, 3, 0) == pytest.approx(0.5)
        # Negative feedback decreases utility rate
        assert compute_bayesian_utility_rate(5, 1, 2) < compute_bayesian_utility_rate(5, 1, 0)

    def test_anti_noise_reversal_verified_use_beats_high_recall_zero_use(self):
        """Experience with recall=3, used=3 strictly beats recall=30, used=0 at equal base score."""
        noise_item = _make_item("noise-30-0", score=0.75, recall_count=30, used_count=0)
        verified_item = _make_item("verified-3-3", score=0.75, recall_count=3, used_count=3)

        ranked = apply_popularity_boost([noise_item, verified_item])

        assert ranked[0].data["id"] == "verified-3-3"
        assert ranked[1].data["id"] == "noise-30-0"
        assert ranked[0].data["popularity_boost"] > ranked[1].data["popularity_boost"]
        assert ranked[0].score > ranked[1].score
        assert ranked[0].data["utility_rate"] > ranked[1].data["utility_rate"]

    def test_negative_feedback_penalty_reduces_boost_below_one(self):
        """Experience with negative_feedback_count >= 2 and used_count == 0 gets boost < 1.0."""
        bad_item = _make_item(
            "bad-exp",
            score=0.80,
            recall_count=10,
            used_count=0,
            negative_feedback_count=2,
        )
        ranked = apply_popularity_boost([bad_item])
        assert ranked[0].data["popularity_boost"] < 1.0
        assert ranked[0].score < 0.80


class TestFeedbackAndAdoptionLoop:
    """Verify feedback(rating <= 2) and increment_used_count update quality_score & counters."""

    def _make_service(self) -> ExperienceService:
        mock_embedding = MagicMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.0] * 768)
        mock_auth = MagicMock()
        return ExperienceService(
            embedding_provider=mock_embedding,
            auth_provider=mock_auth,
            db_url="sqlite+aiosqlite://",
        )

    @pytest.mark.asyncio
    async def test_negative_feedback_increments_negative_count_and_penalizes_quality(self):
        svc = self._make_service()
        exp_id = str(uuid.uuid4())
        exp_uuid = uuid.UUID(exp_id)

        mock_session = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=MagicMock(id=exp_uuid))
        mock_repo.add_feedback = AsyncMock()
        mock_repo.increment_quality_score = AsyncMock(return_value=True)
        mock_repo.increment_negative_feedback_count = AsyncMock()
        mock_repo.increment_used_count = AsyncMock()

        @asynccontextmanager
        async def _mock_gs(_db_url):
            yield mock_session

        with (
            patch("team_memory.storage.database.get_session", _mock_gs),
            patch(
                "team_memory.services.memory.experience.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            ok = await svc.feedback(
                experience_id=exp_id,
                rating=1,
                feedback_by="bob",
                comment="Outdated advice",
            )

        assert ok is True
        mock_repo.increment_quality_score.assert_awaited_once_with(exp_uuid, -3.0)
        mock_repo.increment_negative_feedback_count.assert_awaited_once_with(exp_uuid)
        mock_repo.increment_used_count.assert_not_called()

    @pytest.mark.asyncio
    async def test_increment_used_count_boosts_quality_score(self):
        svc = self._make_service()
        exp_id = str(uuid.uuid4())
        exp_uuid = uuid.UUID(exp_id)

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_repo = MagicMock()
        mock_repo.increment_quality_score = AsyncMock(return_value=True)

        @asynccontextmanager
        async def _mock_gs(_db_url):
            yield mock_session

        with (
            patch("team_memory.storage.database.get_session", _mock_gs),
            patch(
                "team_memory.services.memory.experience.ExperienceRepository",
                return_value=mock_repo,
            ),
        ):
            await svc.increment_used_count(exp_id)

        mock_session.execute.assert_awaited_once()
        mock_repo.increment_quality_score.assert_awaited_once_with(exp_uuid, 2.0)
        mock_session.commit.assert_awaited_once()
