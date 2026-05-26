"""Unit tests for AutoDreamWorker — covers lifecycle, trigger conditions,
4-phase dream process (orient, gather, consolidate, prune), and persistence.

All tests are fully mocked (no real DB, no real LLM).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from team_memory.daemon.integration.auto_dream import AutoDreamWorker, _read_last_dream, _write_last_dream
from team_memory.daemon.runtime.config import AutoDreamSettings, DaemonConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> DaemonConfig:
    """Build a DaemonConfig with sensible test defaults."""
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
    return cfg


def _make_sink() -> AsyncMock:
    """Create a mock TMSink."""
    sink = AsyncMock()
    sink.recall = AsyncMock(return_value=[])
    sink.update_experience = AsyncMock(return_value={"id": "exp-1", "status": "published"})
    return sink


def _make_experience(
    exp_id: str = "exp-001",
    title: str = "Test Experience",
    description: str = "A test description",
    solution: str = "A test solution",
    project: str = "test_project",
    quality_score: float = 50.0,
    quality_tier: str = "Bronze",
    recall_count: int = 0,
    tags: list[str] | None = None,
    similarity: float = 0.0,
) -> dict:
    return {
        "id": exp_id,
        "title": title,
        "description": description,
        "solution": solution,
        "project": project,
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "recall_count": recall_count,
        "tags": tags or ["test"],
        "similarity": similarity,
    }


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestDreamPersistence:
    """Tests for last_dream.txt read/write."""

    def test_read_nonexistent_file(self, tmp_path):
        """Reading a non-existent file returns None."""
        with patch("team_memory.daemon.integration.auto_dream._LAST_DREAM_FILE", tmp_path / "nope.txt"):
            assert _read_last_dream() is None

    def test_read_empty_file(self, tmp_path):
        """Reading an empty file returns None."""
        f = tmp_path / "last_dream.txt"
        f.write_text("")
        with patch("team_memory.daemon.integration.auto_dream._LAST_DREAM_FILE", f):
            assert _read_last_dream() is None

    def test_roundtrip(self, tmp_path):
        """Write then read returns the same timestamp."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        f = tmp_path / "last_dream.txt"
        with patch("team_memory.daemon.integration.auto_dream._LAST_DREAM_FILE", f):
            _write_last_dream(ts)
            result = _read_last_dream()
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.hour == 10


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestAutoDreamLifecycle:
    """Tests for worker start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_disabled_worker_returns_noop_task(self):
        """When disabled, start() returns a task that completes immediately."""
        cfg = _make_config(enabled=False)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)
        task = worker.start()
        await task  # Should complete immediately
        assert task.done()

    @pytest.mark.asyncio
    async def test_enabled_worker_starts_loop(self):
        """When enabled, start() returns a running task."""
        cfg = _make_config(scan_interval_minutes=60)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)
        task = worker.start()
        assert not task.done()
        worker.stop()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """stop() cancels the running task."""
        cfg = _make_config(scan_interval_minutes=60)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)
        task = worker.start()
        worker.stop()
        with pytest.raises((asyncio.CancelledError, asyncio.InvalidStateError)):
            await asyncio.wait_for(task, timeout=2.0)


# ---------------------------------------------------------------------------
# Trigger condition tests
# ---------------------------------------------------------------------------


class TestTriggerConditions:
    """Tests for the trigger condition logic in _tick."""

    @pytest.mark.asyncio
    async def test_skip_when_too_soon(self, tmp_path):
        """Skip dream when not enough time has passed since last run."""
        cfg = _make_config(min_hours_since_last=24, min_new_experiences=1)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        # Last dream was 1 hour ago
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        f = tmp_path / "last_dream.txt"
        f.write_text(recent.isoformat())

        with patch("team_memory.daemon.integration.auto_dream._LAST_DREAM_FILE", f):
            await worker._tick()

        # Should not have called any sink methods
        sink.recall.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_too_few_new_experiences(self, tmp_path):
        """Skip dream when not enough new experiences."""
        cfg = _make_config(min_hours_since_last=0, min_new_experiences=100)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        # Set last dream to long ago
        old = datetime.now(timezone.utc) - timedelta(hours=48)
        f = tmp_path / "last_dream.txt"
        f.write_text(old.isoformat())

        with patch("team_memory.daemon.integration.auto_dream._LAST_DREAM_FILE", f):
            with patch.object(worker, "_count_new_experiences", return_value=2):
                await worker._tick()

        sink.recall.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_when_conditions_met(self, tmp_path):
        """Dream cycle runs when both conditions are met."""
        cfg = _make_config(min_hours_since_last=1, min_new_experiences=1)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        old = datetime.now(timezone.utc) - timedelta(hours=2)
        f = tmp_path / "last_dream.txt"
        f.write_text(old.isoformat())

        with patch("team_memory.daemon.integration.auto_dream._LAST_DREAM_FILE", f):
            with patch.object(worker, "_count_new_experiences", return_value=10):
                with patch.object(
                    worker, "_dream", return_value={"candidates": 5, "clusters": 2, "merged": 3, "pruned": 1}
                ) as mock_dream:
                    await worker._tick()
                    mock_dream.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 1: Orient tests
# ---------------------------------------------------------------------------


class TestPhaseOrient:
    """Tests for the orient phase (scanning for candidates)."""

    @pytest.mark.asyncio
    async def test_orient_falls_back_to_recall(self):
        """When DB query fails, orient falls back to recall-based scan."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        # DB query fails, recall returns some results
        sink.recall.return_value = [
            _make_experience(exp_id="r1", quality_tier="Bronze", recall_count=0),
            _make_experience(exp_id="r2", quality_tier="Silver", recall_count=5),  # Not a candidate
        ]

        with patch.object(worker, "_get_dream_candidates", side_effect=Exception("DB down")):
            candidates = await worker._phase_orient(None)

        # Should have found the Bronze experience
        assert len(candidates) == 1
        assert candidates[0]["id"] == "r1"


# ---------------------------------------------------------------------------
# Phase 2: Gather tests
# ---------------------------------------------------------------------------


class TestPhaseGather:
    """Tests for the gather phase (clustering similar experiences)."""

    @pytest.mark.asyncio
    async def test_gather_builds_clusters(self):
        """Gather builds clusters when similar experiences are found."""
        cfg = _make_config(merge_similarity_threshold=0.85)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        candidates = [
            _make_experience(exp_id="c1", title="Docker connection timeout", description="Docker PG connection fails"),
            _make_experience(exp_id="c2", title="Docker network issue", description="Docker network problem"),
        ]

        # recall returns similar experiences for c1
        sink.recall.return_value = [
            _make_experience(exp_id="c2", title="Docker network issue", similarity=0.92),
        ]

        clusters = await worker._phase_gather(candidates)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    @pytest.mark.asyncio
    async def test_gather_no_cluster_below_threshold(self):
        """No cluster formed when similarity is below threshold."""
        cfg = _make_config(merge_similarity_threshold=0.85)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        candidates = [
            _make_experience(exp_id="c1", title="Docker connection timeout"),
        ]

        # Similarity too low
        sink.recall.return_value = [
            _make_experience(exp_id="c2", title="Unrelated topic", similarity=0.6),
        ]

        clusters = await worker._phase_gather(candidates)
        assert len(clusters) == 0

    @pytest.mark.asyncio
    async def test_gather_skips_already_processed(self):
        """Experiences already in a cluster are not reused."""
        cfg = _make_config(merge_similarity_threshold=0.85)
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        candidates = [
            _make_experience(exp_id="c1", title="Docker timeout A"),
            _make_experience(exp_id="c2", title="Docker timeout B"),
        ]

        # c1 finds c2 as similar → c2 is processed. When c2 is checked, it's already in a cluster.
        call_count = 0

        async def mock_recall(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_make_experience(exp_id="c2", similarity=0.90)]
            return []

        sink.recall.side_effect = mock_recall

        clusters = await worker._phase_gather(candidates)
        # Should form one cluster with c1 and c2
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# Phase 3: Consolidate tests
# ---------------------------------------------------------------------------


class TestPhaseConsolidate:
    """Tests for the consolidate phase (merging near-duplicates)."""

    @pytest.mark.asyncio
    async def test_consolidate_picks_highest_quality_as_primary(self):
        """The experience with highest quality_score becomes the primary."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        cluster = [
            _make_experience(exp_id="low", quality_score=30.0, title="Low quality", solution="Solution A"),
            _make_experience(exp_id="high", quality_score=80.0, title="High quality", solution="Solution B"),
        ]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock):
            merged = await worker._merge_cluster(cluster)

        # Should update the high-quality one
        sink.update_experience.assert_called_once()
        call_kwargs = sink.update_experience.call_args[1]
        assert call_kwargs["experience_id"] == "high"

    @pytest.mark.asyncio
    async def test_consolidate_merges_tags(self):
        """Tags from secondaries are merged into primary."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        cluster = [
            _make_experience(exp_id="p1", quality_score=80.0, tags=["docker", "postgres"]),
            _make_experience(exp_id="s1", quality_score=30.0, tags=["network", "docker"]),
        ]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock):
            merged = await worker._merge_cluster(cluster)

        # Tags should be merged
        call_kwargs = sink.update_experience.call_args[1]
        assert "tags" in call_kwargs
        assert "docker" in call_kwargs["tags"]
        assert "network" in call_kwargs["tags"]
        assert "postgres" in call_kwargs["tags"]

    @pytest.mark.asyncio
    async def test_consolidate_soft_deletes_secondaries(self):
        """Secondary experiences are soft-deleted."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        cluster = [
            _make_experience(exp_id="p1", quality_score=80.0),
            _make_experience(exp_id="s1", quality_score=30.0),
            _make_experience(exp_id="s2", quality_score=20.0),
        ]

        with patch.object(worker, "_soft_delete", new_callable=AsyncMock) as mock_delete:
            merged = await worker._merge_cluster(cluster)
            assert mock_delete.call_count == 2

    @pytest.mark.asyncio
    async def test_consolidate_no_secondary_no_delete(self):
        """If cluster has only one member (somehow), nothing is deleted."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        cluster = [_make_experience(exp_id="p1", quality_score=80.0)]
        merged = await worker._merge_cluster(cluster)
        assert merged == 0


# ---------------------------------------------------------------------------
# Phase 4: Prune tests
# ---------------------------------------------------------------------------


class TestPhasePrune:
    """Tests for the prune phase (removing stale experiences)."""

    @pytest.mark.asyncio
    async def test_prune_returns_zero_on_db_failure(self):
        """If DB query fails, prune returns 0."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        with patch.object(worker, "_prune_stale", side_effect=Exception("DB error")):
            pruned = await worker._phase_prune()
            assert pruned == 0


# ---------------------------------------------------------------------------
# Full dream cycle integration test (mocked)
# ---------------------------------------------------------------------------


class TestDreamCycle:
    """Integration-style test for the full 4-phase dream cycle."""

    @pytest.mark.asyncio
    async def test_full_dream_cycle(self):
        """End-to-end dream cycle with mocked DB and sink."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        # Mock orient to return candidates
        candidates = [
            _make_experience(exp_id="c1", title="Docker timeout", quality_tier="Bronze"),
            _make_experience(exp_id="c2", title="Docker connection issue", quality_tier="Bronze"),
        ]

        # Mock gather to return clusters
        clusters = [
            [
                _make_experience(exp_id="c1", quality_score=80.0, title="Docker timeout"),
                _make_experience(exp_id="c2", quality_score=30.0, title="Docker connection issue"),
            ]
        ]

        # Mock consolidate
        with patch.object(worker, "_phase_orient", return_value=candidates):
            with patch.object(worker, "_phase_gather", return_value=clusters):
                with patch.object(worker, "_phase_prune", return_value=3):
                    with patch.object(worker, "_soft_delete", new_callable=AsyncMock):
                        stats = await worker._dream(None)

        assert stats["candidates"] == 2
        assert stats["clusters"] == 1
        assert stats["merged"] == 1  # c2 merged into c1
        assert stats["pruned"] == 3

    @pytest.mark.asyncio
    async def test_dream_with_no_candidates_still_prunes(self):
        """Even with no merge candidates, prune phase still runs."""
        cfg = _make_config()
        sink = _make_sink()
        worker = AutoDreamWorker(config=cfg, sink=sink)

        with patch.object(worker, "_phase_orient", return_value=[]):
            with patch.object(worker, "_phase_prune", return_value=5):
                stats = await worker._dream(None)

        assert stats["candidates"] == 0
        assert stats["merged"] == 0
        assert stats["pruned"] == 5


# ---------------------------------------------------------------------------
# Config integration test
# ---------------------------------------------------------------------------


class TestAutoDreamConfig:
    """Tests for AutoDreamSettings in DaemonConfig."""

    def test_default_config(self):
        """AutoDreamSettings has correct defaults."""
        cfg = DaemonConfig()
        assert cfg.auto_dream.enabled is True
        assert cfg.auto_dream.min_hours_since_last == 24
        assert cfg.auto_dream.min_new_experiences == 5
        assert cfg.auto_dream.merge_similarity_threshold == 0.85
        assert cfg.auto_dream.prune_max_age_days == 30
        assert cfg.auto_dream.scan_interval_minutes == 60

    def test_config_from_yaml_dict(self):
        """AutoDreamSettings can be loaded from YAML dict."""
        from team_memory.daemon.runtime.config import _dict_to_dataclass

        data = {
            "enabled": False,
            "min_hours_since_last": 12,
            "min_new_experiences": 3,
        }
        settings = _dict_to_dataclass(AutoDreamSettings, data)
        assert settings.enabled is False
        assert settings.min_hours_since_last == 12
        assert settings.min_new_experiences == 3
        # Unspecified fields keep defaults
        assert settings.merge_similarity_threshold == 0.85
