"""Tests for P0 Core Experience features.

Covers:
  - P0-1: Draft mode (create draft, list drafts, publish)
  - P0-3: Dedup-on-save (duplicate detection, skip dedup)
  - P0-4: Summary generation (single, batch)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from team_memory.auth.provider import ApiKeyAuth
from team_memory.config import LifecycleConfig, MemoryConfig
from team_memory.web import app as web_module
from team_memory.web.app import app

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def setup_app():
    """Set up app globals with mock service and real auth for each test."""
    auth = ApiKeyAuth()
    auth.register_key("test-key-123", "test_admin", "admin")
    web_module._auth = auth
    web_module._settings = MagicMock()
    web_module._settings.database.url = "postgresql+asyncpg://test:test@localhost:5432/team_memory"
    web_module._settings.lifecycle = LifecycleConfig(dedup_on_save=True, dedup_on_save_threshold=0.90)
    web_module._settings.memory = MemoryConfig(auto_summarize=True, summary_threshold_tokens=500)
    web_module._settings.retrieval = MagicMock()
    web_module._settings.retrieval.max_count = 20
    web_module._settings.retrieval.top_k_children = 3
    web_module._settings.retrieval.min_avg_rating = 0.0
    web_module._settings.retrieval.rating_weight = 0.3

    mock_service = MagicMock()
    exp_id = str(uuid.uuid4())
    mock_service.save = AsyncMock(return_value={
        "id": exp_id,
        "title": "Test Exp",
        "exp_status": "published",
        "visibility": "project",
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    mock_service.search = AsyncMock(return_value=[])
    mock_service.feedback = AsyncMock(return_value=True)
    mock_service.get_stats = AsyncMock(return_value={
        "total_experiences": 5,
        "tag_distribution": {},
        "recent_7days": 2,
        "pending_reviews": 1,
        "stale_count": 0,
    })
    mock_service.get_drafts = AsyncMock(return_value=[])
    mock_service.publish_experience = AsyncMock(return_value={
        "id": exp_id,
        "title": "Test Exp",
        "exp_status": "published",
        "visibility": "project",
    })
    mock_service.generate_summary = AsyncMock(return_value={
        "id": exp_id,
        "title": "Test Exp",
        "summary": "这是一个测试摘要。",
    })
    mock_service.batch_generate_summaries = AsyncMock(return_value={
        "generated": 3,
        "total_candidates": 5,
        "errors": [],
    })
    mock_service.invalidate_search_cache = AsyncMock()
    mock_service._search_pipeline = None
    mock_service._event_bus = MagicMock()
    mock_service._embedding_queue = None
    web_module._service = mock_service

    # Patch bootstrap so lifespan runs and /api/v1 routes are registered
    mock_ctx = MagicMock()
    mock_ctx.settings = web_module._settings
    mock_ctx.service = mock_service
    mock_ctx.auth = web_module._auth
    with patch("team_memory.web.app.bootstrap", return_value=mock_ctx), patch(
        "team_memory.web.app.start_background_tasks", new_callable=AsyncMock
    ), patch(
        "team_memory.web.app.stop_background_tasks", new_callable=AsyncMock
    ):
        yield mock_service

    web_module._auth = None
    web_module._settings = None
    web_module._service = None


@pytest.fixture
def client(setup_app):
    """Create a test client; use context manager so lifespan runs and routes are registered."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Auth headers for the test admin."""
    return {"Authorization": "Bearer test-key-123"}


# ============================================================
# P0-1: Draft Mode
# ============================================================

class TestDraftMode:
    """Test draft creation, listing, and publishing."""

    def test_create_experience_as_draft(self, client, auth_headers, setup_app):
        """Creating with status=draft should work."""
        setup_app.save.return_value = {
            "id": str(uuid.uuid4()),
            "title": "Draft Exp",
            "exp_status": "draft",
            "visibility": "project",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        resp = client.post("/api/v1/experiences", json={
            "title": "Draft Exp",
            "problem": "Test problem",
            "solution": "Test solution",
            "status": "draft",
        }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("exp_status") == "draft"
        setup_app.save.assert_called_once()
        call_kwargs = setup_app.save.call_args
        assert call_kwargs.kwargs.get("exp_status") == "draft"

    def test_create_experience_default_published(self, client, auth_headers, setup_app):
        """Creating without explicit status should default to published."""
        resp = client.post("/api/v1/experiences", json={
            "title": "Published Exp",
            "problem": "Test problem",
            "solution": "Test solution",
        }, headers=auth_headers)
        assert resp.status_code == 200
        call_kwargs = setup_app.save.call_args
        assert call_kwargs.kwargs.get("exp_status") == "published"

    def test_list_drafts(self, client, auth_headers, setup_app):
        """GET /api/v1/experiences/drafts should return drafts."""
        setup_app.get_drafts.return_value = [
            {"id": str(uuid.uuid4()), "title": "My Draft", "exp_status": "draft"},
        ]
        resp = client.get("/api/v1/experiences/drafts", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["experiences"]) == 1
        assert data["experiences"][0]["title"] == "My Draft"

    def test_publish_draft(self, client, auth_headers):
        """POST /api/v1/experiences/{id}/publish should publish a draft."""
        exp_id = str(uuid.uuid4())
        mock_repo = MagicMock()
        mock_exp = MagicMock()
        mock_exp.to_dict.return_value = {"id": exp_id, "exp_status": "published"}
        mock_repo.change_status = AsyncMock(return_value=mock_exp)
        mock_repo.get_by_id = AsyncMock(return_value=MagicMock())
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("team_memory.web.routes.experiences.app_module.get_session", return_value=mock_session_ctx), \
             patch("team_memory.web.routes.experiences.app_module.ExperienceRepository", return_value=mock_repo), \
             patch("team_memory.web.routes.experiences.write_audit_log", new_callable=AsyncMock):
            resp = client.post(f"/api/v1/experiences/{exp_id}/publish", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        mock_repo.change_status.assert_called()

    def test_publish_not_found(self, client, auth_headers):
        """Publishing non-existent experience should return 404."""
        mock_repo = MagicMock()
        mock_repo.change_status = AsyncMock(return_value=None)
        mock_repo.get_by_id = AsyncMock(return_value=MagicMock())
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("team_memory.web.routes.experiences.app_module.get_session", return_value=mock_session_ctx), \
             patch("team_memory.web.routes.experiences.app_module.ExperienceRepository", return_value=mock_repo), \
             patch("team_memory.web.routes.experiences.write_audit_log", new_callable=AsyncMock):
            resp = client.post(f"/api/v1/experiences/{uuid.uuid4()}/publish", headers=auth_headers)
        assert resp.status_code == 404

    def test_list_experiences_with_status_filter(self, client, auth_headers):
        """GET /api/v1/experiences?status=all should include all statuses."""
        with patch("team_memory.web.app.get_session") as mock_session:
            mock_ctx = MagicMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_session.return_value.__aexit__ = AsyncMock()

            mock_repo = MagicMock()
            mock_repo.list_recent = AsyncMock(return_value=[])
            mock_repo.count = AsyncMock(return_value=0)
            mock_repo.list_drafts = AsyncMock(return_value=[])
            mock_repo.count_drafts = AsyncMock(return_value=0)

            with patch("team_memory.web.app.ExperienceRepository", return_value=mock_repo):
                resp = client.get("/api/v1/experiences?status=draft", headers=auth_headers)
                assert resp.status_code == 200


# ============================================================
# P0-3: Dedup-on-Save
# ============================================================

class TestDedupOnSave:
    """Test duplicate detection during save."""

    def test_dedup_detected_returns_409(self, client, auth_headers, setup_app):
        """When save() returns duplicate_detected, API should return 409."""
        setup_app.save.return_value = {
            "status": "duplicate_detected",
            "candidates": [
                {"id": str(uuid.uuid4()), "title": "Existing", "similarity": 0.95, "tags": ["python"]},
            ],
            "message": "发现 1 条相似经验",
            "experience_id": None,
        }
        resp = client.post("/api/v1/experiences", json={
            "title": "Dup Exp",
            "problem": "Same problem",
            "solution": "Same solution",
        }, headers=auth_headers)
        assert resp.status_code == 409
        data = resp.json()
        assert data["status"] == "duplicate_detected"
        assert len(data["candidates"]) == 1

    def test_skip_dedup_creates_normally(self, client, auth_headers, setup_app):
        """When skip_dedup_check=true, should bypass dedup."""
        setup_app.save.return_value = {
            "id": str(uuid.uuid4()),
            "title": "Force Saved",
            "exp_status": "published",
            "visibility": "project",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        resp = client.post("/api/v1/experiences", json={
            "title": "Force Saved",
            "problem": "Force save",
            "solution": "Force solution",
            "skip_dedup_check": True,
        }, headers=auth_headers)
        assert resp.status_code == 200
        call_kwargs = setup_app.save.call_args.kwargs
        assert call_kwargs.get("skip_dedup") is True

    def test_lifecycle_config_includes_dedup_settings(self, client, auth_headers):
        """GET /api/v1/config/lifecycle should include dedup_on_save settings."""
        resp = client.get("/api/v1/config/lifecycle", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "dedup_on_save" in data
        assert "dedup_on_save_threshold" in data


# ============================================================
# P0-4: Summary / Memory Compaction
# ============================================================

class TestSummary:
    """Test summary generation endpoints."""

    def test_summarize_single(self, client, auth_headers, setup_app):
        """POST /api/v1/experiences/{id}/summarize should generate summary."""
        exp_id = str(uuid.uuid4())
        resp = client.post(f"/api/v1/experiences/{exp_id}/summarize", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "Summary generated"
        assert data["experience"]["summary"] == "这是一个测试摘要。"

    def test_summarize_not_found(self, client, auth_headers, setup_app):
        """Summarizing non-existent experience should return 404."""
        setup_app.generate_summary.return_value = None
        resp = client.post(f"/api/v1/experiences/{uuid.uuid4()}/summarize", headers=auth_headers)
        assert resp.status_code == 404

    def test_batch_summarize(self, client, auth_headers, setup_app):
        """POST /api/v1/experiences/batch-summarize should batch generate."""
        resp = client.post("/api/v1/experiences/batch-summarize?limit=5", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated"] == 3
        assert data["total_candidates"] == 5

    def test_memory_config_crud(self, client, auth_headers):
        """GET and PUT /api/v1/config/memory should work."""
        resp = client.get("/api/v1/config/memory", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "auto_summarize" in data
        assert "summary_threshold_tokens" in data

        resp = client.put("/api/v1/config/memory", json={
            "auto_summarize": False,
            "summary_threshold_tokens": 1000,
            "summary_model": "llama3",
            "batch_size": 20,
        }, headers=auth_headers)
        assert resp.status_code == 200


# ============================================================
# Config Integration
# ============================================================

class TestConfigClasses:
    """Test P0 config classes."""

    def test_memory_config_defaults(self):
        cfg = MemoryConfig()
        assert cfg.auto_summarize is True
        assert cfg.summary_threshold_tokens == 500
        assert cfg.summary_model == ""
        assert cfg.batch_size == 10

    def test_lifecycle_config_dedup_defaults(self):
        cfg = LifecycleConfig()
        assert cfg.dedup_on_save is True
        assert cfg.dedup_on_save_threshold == 0.90


# ============================================================
# Event Bus Integration
# ============================================================

class TestEventBusP0:
    """Test P0-related event constants."""

    def test_published_event_exists(self):
        from team_memory.services.event_bus import Events
        assert hasattr(Events, "EXPERIENCE_PUBLISHED")
        assert Events.EXPERIENCE_PUBLISHED == "experience.published"

    def test_reviewed_event_constant_exists(self):
        from team_memory.services.event_bus import Events
        assert hasattr(Events, "EXPERIENCE_REVIEWED")
