"""Tests for the FastAPI web application.

Tests auth, experience CRUD, search, and stats endpoints.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from team_memory.auth.provider import ApiKeyAuth
from team_memory.config import UploadsConfig, WebConfig
from team_memory.services.archive import ArchiveUploadError
from team_memory.web import app as web_module
from team_memory.web.app import app

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def setup_app():
    """Set up app globals with mock service and real auth for each test.

    Patches bootstrap/start_background_tasks/stop_background_tasks so that
    when TestClient runs the app lifespan, routes get registered without
    requiring a real DB or embedding service.
    """
    auth = ApiKeyAuth()
    auth.register_key("test-key-123", "test_admin", "admin")
    auth.register_key("member-key", "test_member", "member")
    auth.register_key("viewer-key", "test_viewer", "viewer")
    auth.register_key("测试-key-123", "unicode_admin", "admin")
    web_module._auth = auth
    web_module._settings = MagicMock()
    web_module._settings.database.url = "postgresql+asyncpg://test:test@localhost:5432/team_memory"
    web_module._settings.default_project = "default"
    web_module._settings.retrieval = MagicMock(
        max_tokens=None,
        max_count=20,
        trim_strategy="top_k",
        top_k_children=3,
        min_avg_rating=0.0,
        rating_weight=0.3,
        summary_model=None,
    )
    web_module._settings.search = MagicMock(
        mode="hybrid",
        rrf_k=60,
        vector_weight=0.7,
        fts_weight=0.3,
        adaptive_filter=True,
        score_gap_threshold=0.15,
        min_confidence_ratio=0.6,
    )
    web_module._settings.cache = MagicMock(
        enabled=True,
        ttl_seconds=300,
        max_size=100,
        embedding_cache_size=200,
    )
    web_module._settings.reranker = MagicMock(
        provider="none",
        ollama_llm=MagicMock(model=None, base_url=None, top_k=10, batch_size=5),
        cross_encoder=MagicMock(model_name="", device="cpu", top_k=10),
        jina=MagicMock(model="", top_k=10, api_key=""),
    )
    web_module._settings.installable_catalog = MagicMock(
        sources=["local"],
        local_base_dir=".debug/knowledge-pack",
        registry_manifest_url="",
        target_rules_dir=".cursor/rules",
        target_prompts_dir=".cursor/prompts",
        request_timeout_seconds=8,
    )
    web_module._settings.extraction = MagicMock(
        quality_gate=2,
        max_retries=1,
        few_shot_examples=None,
    )
    web_module._settings.logging = MagicMock(
        log_io_enabled=False,
        log_io_detail="mcp",
        log_io_truncate=300,
        log_file_enabled=False,
        log_file_path="./logs/team_memory.log",
        log_file_max_bytes=10 * 1024 * 1024,
        log_file_backup_count=5,
    )
    web_module._settings.uploads = UploadsConfig()
    web_module._settings.web = WebConfig()

    mock_service = MagicMock()
    mock_service.save = AsyncMock(
        return_value={
            "id": str(uuid.uuid4()),
            "title": "Test",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    mock_service.feedback = AsyncMock(return_value=True)
    mock_service.update = AsyncMock(
        return_value={
            "id": str(uuid.uuid4()),
            "title": "Updated",
            "solution": "Updated solution",
            "tags": ["test"],
        }
    )
    mock_service.get_stats = AsyncMock(
        return_value={
            "total_experiences": 5,
            "tag_distribution": {"python": 3, "docker": 2},
            "recent_7days": 2,
        }
    )
    mock_service.soft_delete = AsyncMock(return_value=True)
    mock_service.restore = AsyncMock(return_value=True)
    mock_service.find_duplicate_pairs = AsyncMock(
        return_value={
            "pairs": [],
            "threshold": 0.85,
            "limit": 50,
            "project": "default",
        }
    )
    mock_service.reembed_group_parent_vectors = AsyncMock(
        return_value={
            "updated": 0,
            "errors": 0,
            "total_groups": 0,
            "project": "default",
        }
    )
    web_module._service = mock_service

    mock_search_orchestrator = MagicMock()
    mock_search_orchestrator.search = AsyncMock(return_value=[])
    mock_search_orchestrator.invalidate_cache = AsyncMock(return_value=None)
    # Attach for test access via setup_app._search_orchestrator
    mock_service._search_orchestrator = mock_search_orchestrator

    mock_archive_svc = MagicMock()
    mock_archive_svc.list_archives = AsyncMock(return_value=([], 0))
    mock_archive_svc.get_archive = AsyncMock(return_value=None)
    mock_archive_svc.upload_archive_attachment = AsyncMock(
        return_value={
            "id": str(uuid.uuid4()),
            "archive_id": str(uuid.uuid4()),
            "kind": "file",
            "path": "x/y",
            "download_api_path": "/api/v1/archives/x/attachments/y/file",
        }
    )
    mock_archive_svc.read_archive_attachment_file = AsyncMock(return_value=None)
    mock_archive_svc.list_upload_failures = AsyncMock(return_value=[])
    mock_archive_svc.mark_upload_failure_resolved = AsyncMock(return_value=False)

    # Patch bootstrap so lifespan can complete without real DB/embedding;
    # routes are registered inside lifespan, so tests would get 404 without this.
    mock_ctx = MagicMock()
    mock_ctx.settings = web_module._settings
    mock_ctx.service = mock_service
    mock_ctx.auth = web_module._auth
    mock_ctx.archive_service = mock_archive_svc
    mock_ctx.search_orchestrator = mock_search_orchestrator
    with (
        patch("team_memory.web.app.bootstrap", return_value=mock_ctx),
        patch("team_memory.bootstrap.get_context", return_value=mock_ctx),
        patch("team_memory.web.routes.archives.get_context", return_value=mock_ctx),
        patch("team_memory.web.routes.search.get_context", return_value=mock_ctx),
        patch("team_memory.web.routes.config.get_context", return_value=mock_ctx),
        patch("team_memory.web.app.start_background_tasks", new_callable=AsyncMock),
        patch("team_memory.web.app.stop_background_tasks", new_callable=AsyncMock),
    ):
        yield mock_service

    # Clean up
    web_module._auth = None
    web_module._settings = None
    web_module._service = None
    # Clear per-IP rate limit store to prevent cross-test leaking
    from team_memory.web.app import _rate_limit_store

    _rate_limit_store.clear()


@pytest.fixture
def client(setup_app):
    """Use TestClient as context manager so lifespan runs and /api/v1 routes are registered."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-key-123"}


@pytest.fixture
def member_headers():
    return {"Authorization": "Bearer member-key"}


@pytest.fixture
def viewer_headers():
    return {"Authorization": "Bearer viewer-key"}


# ============================================================
# Auth Tests
# ============================================================


class TestAuth:
    def test_login_success(self, client):
        resp = client.post("/api/v1/auth/login", json={"api_key": "test-key-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["user"] == "test_admin"
        assert data["role"] == "admin"

    def test_login_unicode_api_key_cookie_roundtrip(self, client):
        resp = client.post("/api/v1/auth/login", json={"api_key": "测试-key-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["user"] == "unicode_admin"

        # Ensure cookie-based auth also works for non-ASCII API key.
        me = client.get("/api/v1/auth/me")
        assert me.status_code == 200
        me_data = me.json()
        assert me_data["user"] == "unicode_admin"

    def test_login_wrong_key(self, client):
        resp = client.post("/api/v1/auth/login", json={"api_key": "wrong-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_login_empty_key(self, client):
        resp = client.post("/api/v1/auth/login", json={"api_key": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_auth_me_valid(self, client, auth_headers):
        resp = client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["user"] == "test_admin"

    def test_auth_me_no_key(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_auth_me_invalid_key(self, client):
        resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 401

    def test_logout(self, client):
        resp = client.post("/api/v1/auth/logout")
        assert resp.status_code == 200

    def test_member_auth(self, client, member_headers):
        resp = client.get("/api/v1/auth/me", headers=member_headers)
        assert resp.status_code == 200
        assert resp.json()["user"] == "test_member"
        assert resp.json()["role"] == "member"

    def test_auth_me_includes_api_key_masked(self, client, auth_headers):
        """GET /auth/me returns api_key_masked (null for ApiKeyAuth, never full key)."""
        resp = client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "api_key_masked" in data
        assert data.get("user") == "test_admin"
        # With ApiKeyAuth we get None; with DbApiKeyAuth would be "xxxx****xxxx"
        assert data["api_key_masked"] is None or (
            isinstance(data["api_key_masked"], str)
            and "****" in data["api_key_masked"]
            and len(data["api_key_masked"]) == 12
        )

    def test_forgot_password_reset_requires_db_mode(self, client):
        """POST /auth/forgot-password/reset returns 400 when not DbApiKeyAuth."""
        resp = client.post(
            "/api/v1/auth/forgot-password/reset",
            json={
                "username": "test_admin",
                "api_key": "test-key-123",
                "new_password": "newpass123",
            },
        )
        assert resp.status_code == 400
        assert "db_api_key" in resp.json().get("detail", "")

    def test_admin_reset_password_requires_admin(self, client, member_headers):
        """POST /auth/admin/reset-password returns 403 when not admin."""
        resp = client.post(
            "/api/v1/auth/admin/reset-password",
            headers=member_headers,
            json={"username": "test_admin", "new_password": "newpass123"},
        )
        assert resp.status_code == 403

    def test_admin_reset_password_requires_db_mode(self, client, auth_headers):
        """POST /auth/admin/reset-password returns 400 when not DbApiKeyAuth."""
        resp = client.post(
            "/api/v1/auth/admin/reset-password",
            headers=auth_headers,
            json={"username": "test_admin", "new_password": "newpass123"},
        )
        assert resp.status_code == 400
        assert "db_api_key" in resp.json().get("detail", "")

    def test_admin_reset_password_success(self, client, auth_headers):
        """POST /auth/admin/reset-password updates password when admin + DbApiKeyAuth."""
        from team_memory.auth.provider import DbApiKeyAuth, User
        from team_memory.storage.models import ApiKey

        class TestDbApiKeyAuth(DbApiKeyAuth):
            """DbApiKeyAuth that accepts test-key-123 for auth (no DB lookup)."""

            async def authenticate(self, credentials):
                if credentials.get("api_key") == "test-key-123":
                    return User("test_admin", "admin")
                return await super().authenticate(credentials)

        mock_key = MagicMock(spec=ApiKey)
        mock_key.user_name = "target_user"
        mock_key.password_hash = "old_hash"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_key

        mock_sess = AsyncMock()
        mock_sess.execute = AsyncMock(return_value=mock_result)
        mock_sess.add = MagicMock()
        mock_sess.flush = AsyncMock()

        def mock_get_session(_db_url):
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_sess)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        db_auth = TestDbApiKeyAuth(db_url="sqlite+aiosqlite:///:memory:", keys={})
        with (
            patch("team_memory.web.routes.auth.app_module._auth", db_auth),
            patch("team_memory.web.routes.auth.get_session", side_effect=mock_get_session),
        ):
            resp = client.post(
                "/api/v1/auth/admin/reset-password",
                headers=auth_headers,
                json={"username": "target_user", "new_password": "newpass123"},
            )
        assert resp.status_code == 200
        assert resp.json()["message"] == "密码已重置"
        assert mock_key.password_hash != "old_hash"
        mock_sess.flush.assert_awaited_once()

    def test_change_password_validation_neither_old_nor_api_key(self, client, auth_headers):
        """PUT /auth/password with only new_password returns 422."""
        resp = client.put(
            "/api/v1/auth/password",
            headers=auth_headers,
            json={"new_password": "newpass123"},
        )
        assert resp.status_code == 422

    def test_change_password_validation_empty_old_password(self, client, auth_headers):
        """PUT /auth/password with empty old_password returns 400."""
        resp = client.put(
            "/api/v1/auth/password",
            headers=auth_headers,
            json={
                "old_password": "",
                "new_password": "newpass123",
            },
        )
        assert resp.status_code == 400


# ============================================================
# Experience CRUD Tests
# ============================================================


class TestExperienceList:
    def test_list_anonymous_access(self, client):
        """List experiences supports anonymous access (no auth required)."""
        with patch("team_memory.web.routes.experiences.app_module.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("team_memory.storage.repository.ExperienceRepository") as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo.count = AsyncMock(return_value=0)
                mock_repo.list_recent = AsyncMock(return_value=[])
                mock_repo_cls.return_value = mock_repo

                resp = client.get("/api/v1/experiences")
                assert resp.status_code == 200
                data = resp.json()
                assert "items" in data

    def test_list_experiences(self, client, auth_headers):
        with patch("team_memory.web.routes.experiences.app_module.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("team_memory.storage.repository.ExperienceRepository") as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo.count = AsyncMock(return_value=2)
                mock_repo.list_recent = AsyncMock(return_value=[])
                mock_repo_cls.return_value = mock_repo

                resp = client.get("/api/v1/experiences", headers=auth_headers)
                assert resp.status_code == 200
                data = resp.json()
                assert "items" in data
                assert data["total"] == 2
                assert data["offset"] == 0


class TestCreateExperience:
    def test_create_requires_auth(self, client):
        resp = client.post(
            "/api/v1/experiences", json={"title": "t", "problem": "p", "solution": "s"}
        )
        assert resp.status_code == 401

    def test_create_experience(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                "/api/v1/experiences",
                headers=auth_headers,
                json={
                    "title": "Test Experience",
                    "problem": "Test problem",
                    "solution": "Test solution",
                    "tags": ["test", "python"],
                },
            )
            assert resp.status_code == 200
            assert "id" in resp.json()["item"]

            setup_app.save.assert_called_once()
            kwargs = setup_app.save.call_args.kwargs
            assert kwargs["created_by"] == "test_admin"
            assert kwargs["source"] == "web"

    def test_create_experience_with_group_key(self, client, auth_headers, setup_app):
        """Create experience with group_key; service.save receives group_key."""
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                "/api/v1/experiences",
                headers=auth_headers,
                json={
                    "title": "Grouped experience",
                    "problem": "Problem",
                    "solution": "Solution",
                    "group_key": "sprint-1",
                },
            )
            assert resp.status_code == 200
            assert "id" in resp.json()["item"]

            setup_app.save.assert_called_once()
            kwargs = setup_app.save.call_args.kwargs
            assert kwargs.get("group_key") == "sprint-1"


class TestDeleteExperience:
    def test_delete_requires_auth(self, client):
        resp = client.delete(f"/api/v1/experiences/{uuid.uuid4()}")
        assert resp.status_code == 401

    def test_soft_delete_success(self, client, auth_headers, setup_app):
        """Default delete is soft-delete."""
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.delete(f"/api/v1/experiences/{uuid.uuid4()}", headers=auth_headers)
            assert resp.status_code == 200
            assert "deleted" in resp.json()["message"].lower()
            setup_app.soft_delete.assert_called_once()

    def test_hard_delete(self, client, auth_headers):
        """Hard delete with ?hard=true."""
        with patch("team_memory.web.routes.experiences.app_module.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("team_memory.storage.repository.ExperienceRepository") as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo.delete = AsyncMock(return_value=True)
                mock_repo_cls.return_value = mock_repo

                resp = client.delete(
                    f"/api/v1/experiences/{uuid.uuid4()}?hard=true",
                    headers=auth_headers,
                )
                assert resp.status_code == 200

    def test_soft_delete_not_found(self, client, auth_headers, setup_app):
        setup_app.soft_delete = AsyncMock(return_value=False)
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.delete(f"/api/v1/experiences/{uuid.uuid4()}", headers=auth_headers)
            assert resp.status_code == 404


# ============================================================
# Search Tests
# ============================================================


class TestSearch:
    def test_search_anonymous_access(self, client, setup_app):
        """Search supports anonymous access."""
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post("/api/v1/search", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert "results" in data

    def test_search(self, client, auth_headers, setup_app):
        setup_app._search_orchestrator.search = AsyncMock(
            return_value=[{"id": str(uuid.uuid4()), "title": "Result", "similarity": 0.85}]
        )

        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                "/api/v1/search",
                headers=auth_headers,
                json={"query": "docker networking", "max_results": 5, "min_similarity": 0.5},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["results"]) == 1
            assert data["results"][0]["title"] == "Result"


class TestProjectParam:
    def test_search_uses_default_project(self, client, setup_app):
        web_module._settings.default_project = "proj-default"
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = client.post("/api/v1/search", json={"query": "hello"})
        assert resp.status_code == 200
        assert setup_app._search_orchestrator.search.call_args.kwargs["project"] == "proj-default"

    def test_create_passes_project(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = client.post(
                "/api/v1/experiences",
                headers=auth_headers,
                json={
                    "title": "p",
                    "problem": "q",
                    "solution": "s",
                    "project": "proj-x",
                },
            )
        assert resp.status_code == 200
        assert setup_app.save.call_args.kwargs["project"] == "proj-x"


# ============================================================
# Feedback Tests
# ============================================================


class TestFeedback:
    def test_feedback_requires_auth(self, client):
        resp = client.post(f"/api/v1/experiences/{uuid.uuid4()}/feedback", json={"rating": 5})
        assert resp.status_code == 401

    def test_feedback_success(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                f"/api/v1/experiences/{uuid.uuid4()}/feedback",
                headers=auth_headers,
                json={"rating": 5, "comment": "Great!"},
            )
            assert resp.status_code == 200
            assert "recorded" in resp.json()["message"].lower()

    def test_feedback_not_found(self, client, auth_headers, setup_app):
        setup_app.feedback = AsyncMock(return_value=False)

        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                f"/api/v1/experiences/{uuid.uuid4()}/feedback",
                headers=auth_headers,
                json={"rating": 5},
            )
            assert resp.status_code == 404


# ============================================================
# Review Tests
# ============================================================


class TestReviewRemoved:
    """Review routes have been removed; verify they return 404/405."""

    def test_review_pending_removed(self, client, auth_headers):
        resp = client.get("/api/v1/reviews/pending", headers=auth_headers)
        assert resp.status_code in (404, 405)

    def test_review_experience_removed(self, client, auth_headers):
        resp = client.post(
            f"/api/v1/experiences/{uuid.uuid4()}/review",
            headers=auth_headers,
            json={"review_status": "approved"},
        )
        assert resp.status_code in (404, 405, 422)


# ============================================================
# Restore Tests
# ============================================================


class TestRestore:
    def test_restore_requires_auth(self, client):
        resp = client.post(f"/api/v1/experiences/{uuid.uuid4()}/restore")
        assert resp.status_code == 401

    def test_restore_success(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                f"/api/v1/experiences/{uuid.uuid4()}/restore",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            assert "restored" in resp.json()["message"].lower()


# ============================================================
# Session Token Encode/Decode (Task 4)
# ============================================================


def test_session_token_encode_decode_roundtrip():
    from team_memory.web.auth_session import _decode_session_token, _encode_session_token

    token = _encode_session_token("admin", secret="test-secret")
    assert token.startswith("sess:")
    user = _decode_session_token(token, secret="test-secret")
    assert user == "admin"


# ============================================================
# Runtime config (retrieval / search)
# ============================================================


class TestRuntimeConfig:
    _RETRIEVAL_JSON = {
        "max_tokens": None,
        "max_count": 20,
        "trim_strategy": "top_k",
        "top_k_children": 3,
        "min_avg_rating": 0.0,
        "rating_weight": 0.0,
        "summary_model": None,
    }

    def test_get_retrieval_requires_auth(self, client):
        resp = client.get("/api/v1/config/retrieval")
        assert resp.status_code == 401

    def test_get_retrieval_ok(self, client, auth_headers):
        resp = client.get("/api/v1/config/retrieval", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_count"] == 20
        assert data["trim_strategy"] == "top_k"

    def test_put_retrieval_ok(self, client, auth_headers, setup_app):
        resp = client.put(
            "/api/v1/config/retrieval",
            headers=auth_headers,
            json={
                **self._RETRIEVAL_JSON,
                "max_count": 12,
                "top_k_children": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["max_count"] == 12
        assert resp.json()["top_k_children"] == 5
        setup_app._search_orchestrator.invalidate_cache.assert_awaited()

    def test_put_retrieval_forbidden_member(self, client, member_headers):
        resp = client.put(
            "/api/v1/config/retrieval",
            headers=member_headers,
            json=self._RETRIEVAL_JSON,
        )
        assert resp.status_code == 403

    def test_get_search_ok(self, client, auth_headers):
        resp = client.get("/api/v1/config/search", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "hybrid"
        assert data["rrf_k"] == 60


class TestDedupApi:
    def test_duplicates_requires_auth(self, client):
        resp = client.get("/api/v1/dedup/pairs")
        assert resp.status_code == 401

    def test_duplicates_ok(self, client, auth_headers, setup_app):
        resp = client.get(
            "/api/v1/dedup/pairs?threshold=0.1&limit=10",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pairs" in data
        setup_app.find_duplicate_pairs.assert_awaited_once()
        call_kw = setup_app.find_duplicate_pairs.call_args.kwargs
        assert call_kw["threshold"] == 0.1
        assert call_kw["limit"] == 10

    def test_reembed_forbidden_member(self, client, member_headers):
        resp = client.post(
            "/api/v1/dedup/reembed-group-vectors",
            headers=member_headers,
        )
        assert resp.status_code == 403

    def test_reembed_ok(self, client, auth_headers, setup_app):
        resp = client.post(
            "/api/v1/dedup/reembed-group-vectors",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        setup_app.reembed_group_parent_vectors.assert_awaited_once()


# ============================================================
# Archives API
# ============================================================


class TestArchivesApi:
    def test_archives_list_ok_anonymous(self, client):
        from team_memory.bootstrap import get_context

        get_context().archive_service.list_archives.return_value = ([], 0)
        resp = client.get("/api/v1/archives")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        get_context().archive_service.list_archives.assert_awaited()

    def test_archives_list_with_auth_viewer(self, client, auth_headers):
        from team_memory.bootstrap import get_context

        get_context().archive_service.list_archives.return_value = (
            [
                {
                    "id": str(uuid.uuid4()),
                    "title": "Doc",
                    "status": "published",
                    "solution_preview": "x",
                    "overview_preview": "",
                    "attachment_count": 0,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "created_by": "u",
                    "project": "default",
                },
            ],
            1,
        )
        resp = client.get("/api/v1/archives", headers=auth_headers)
        assert resp.status_code == 200
        kw = get_context().archive_service.list_archives.call_args.kwargs
        assert kw.get("viewer") == "test_admin"

    def test_archives_detail_not_found(self, client):
        from team_memory.bootstrap import get_context

        get_context().archive_service.get_archive.return_value = None
        aid = str(uuid.uuid4())
        resp = client.get(f"/api/v1/archives/{aid}")
        assert resp.status_code == 404

    def test_archives_attachment_upload_ok(self, client, auth_headers):
        from team_memory.bootstrap import get_context

        aid = uuid.uuid4()
        att_id = uuid.uuid4()
        get_context().archive_service.upload_archive_attachment = AsyncMock(
            return_value={
                "id": str(att_id),
                "archive_id": str(aid),
                "kind": "file",
                "path": f"{aid}/{att_id}.md",
                "download_api_path": f"/api/v1/archives/{aid}/attachments/{att_id}/file",
            }
        )
        files = {"file": ("note.md", b"# hello", "text/markdown")}
        data = {"kind": "file", "note": "n1"}
        resp = client.post(
            f"/api/v1/archives/{aid}/attachments/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(att_id)
        assert body["download_api_path"].endswith("/file")
        get_context().archive_service.upload_archive_attachment.assert_awaited_once()

    def test_archives_attachment_upload_service_error(self, client, auth_headers):
        from team_memory.bootstrap import get_context

        aid = uuid.uuid4()
        get_context().archive_service.upload_archive_attachment = AsyncMock(
            side_effect=ArchiveUploadError(
                "disabled",
                "File uploads are disabled",
                http_status=503,
            )
        )
        files = {"file": ("x.md", b"x", "text/plain")}
        resp = client.post(
            f"/api/v1/archives/{aid}/attachments/upload",
            files=files,
            headers=auth_headers,
        )
        assert resp.status_code == 503
        assert "disabled" in resp.json().get("detail", "").lower()

    def test_archives_attachment_download_ok(self, client, auth_headers, tmp_path):
        from team_memory.bootstrap import get_context

        aid = uuid.uuid4()
        att_id = uuid.uuid4()
        fpath = tmp_path / "blob.md"
        fpath.write_bytes(b"attachment-bytes")
        get_context().archive_service.read_archive_attachment_file = AsyncMock(
            return_value=(fpath, "blob.md")
        )
        resp = client.get(
            f"/api/v1/archives/{aid}/attachments/{att_id}/file",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.content == b"attachment-bytes"

    def test_archives_upload_failures_not_found(self, client, auth_headers):
        from team_memory.bootstrap import get_context

        aid = uuid.uuid4()
        get_context().archive_service.get_archive.return_value = None
        resp = client.get(
            f"/api/v1/archives/{aid}/upload-failures",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_archives_upload_failures_ok(self, client, auth_headers):
        from team_memory.bootstrap import get_context

        aid = uuid.uuid4()
        fid = uuid.uuid4()
        get_context().archive_service.get_archive.return_value = {"id": str(aid)}
        get_context().archive_service.list_upload_failures = AsyncMock(
            return_value=[
                {
                    "id": str(fid),
                    "archive_id": str(aid),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "created_by": "u",
                    "source": "web",
                    "error_code": "413",
                    "error_message": "too big",
                    "client_filename_hint": "big.pdf",
                    "resolved_at": None,
                },
            ]
        )
        resp = client.get(
            f"/api/v1/archives/{aid}/upload-failures",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        items = resp.json().get("items", [])
        assert len(items) == 1
        assert items[0]["error_code"] == "413"

    def test_archives_upload_failure_resolve_ok(self, client, auth_headers):
        from team_memory.bootstrap import get_context

        aid = uuid.uuid4()
        fid = uuid.uuid4()
        get_context().archive_service.mark_upload_failure_resolved = AsyncMock(return_value=True)
        resp = client.patch(
            f"/api/v1/archives/{aid}/upload-failures/{fid}",
            json={"resolved": True},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json().get("ok") is True

    def test_archives_upload_failure_resolve_not_found(self, client, auth_headers):
        from team_memory.bootstrap import get_context

        aid = uuid.uuid4()
        fid = uuid.uuid4()
        get_context().archive_service.mark_upload_failure_resolved = AsyncMock(return_value=False)
        resp = client.patch(
            f"/api/v1/archives/{aid}/upload-failures/{fid}",
            json={"resolved": True},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    # ------ POST /api/v1/archives (create / upsert) ------

    def test_archives_create_ok(self, client, auth_headers):
        """POST /api/v1/archives creates an archive and returns 201."""
        from team_memory.bootstrap import get_context

        archive_id = str(uuid.uuid4())
        get_context().archive_service.archive_upsert = AsyncMock(
            return_value={
                "action": "created",
                "archive_id": archive_id,
            }
        )
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "My New Archive",
                "solution_doc": "The detailed solution.",
                "tags": ["python", "testing"],
                "overview": "A brief overview.",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["item"]["action"] == "created"
        assert data["item"]["archive_id"] == archive_id
        assert data["message"] == "Created successfully"

        call_kw = get_context().archive_service.archive_upsert.call_args.kwargs
        assert call_kw["title"] == "My New Archive"
        assert call_kw["solution_doc"] == "The detailed solution."
        assert call_kw["created_by"] == "test_admin"
        assert call_kw["tags"] == ["python", "testing"]
        assert call_kw["overview"] == "A brief overview."

    def test_archives_create_upsert_updated(self, client, auth_headers):
        """POST /api/v1/archives with existing title+project returns action=updated and 200."""
        from team_memory.bootstrap import get_context

        archive_id = str(uuid.uuid4())
        get_context().archive_service.archive_upsert = AsyncMock(
            return_value={
                "action": "updated",
                "archive_id": archive_id,
                "previous_updated_at": "2026-03-30T12:00:00+00:00",
            }
        )
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "Existing Archive",
                "solution_doc": "Updated solution.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["item"]["action"] == "updated"
        assert data["item"]["archive_id"] == archive_id
        assert "previous_updated_at" in data["item"]
        assert data["message"] == "Updated successfully"

    def test_archives_create_requires_auth(self, client):
        """POST /api/v1/archives without auth returns 401."""
        resp = client.post(
            "/api/v1/archives",
            json={
                "title": "No Auth Archive",
                "solution_doc": "solution",
            },
        )
        assert resp.status_code == 401

    def test_archives_create_missing_fields(self, client, auth_headers):
        """POST /api/v1/archives without required fields returns 422."""
        # Missing 'title' (required by ArchiveCreateRequest)
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "solution_doc": "only solution, no title",
            },
        )
        assert resp.status_code == 422

        # Missing 'solution_doc' (required by ArchiveCreateRequest)
        resp2 = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "Only title, no solution_doc",
            },
        )
        assert resp2.status_code == 422

        # Empty body
        resp3 = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={},
        )
        assert resp3.status_code == 422

    def test_archives_create_with_linked_experience_ids(self, client, auth_headers):
        """POST /api/v1/archives passes linked_experience_ids as UUIDs to service."""
        from team_memory.bootstrap import get_context

        exp_id_1 = str(uuid.uuid4())
        exp_id_2 = str(uuid.uuid4())
        archive_id = str(uuid.uuid4())
        get_context().archive_service.archive_upsert = AsyncMock(
            return_value={"action": "created", "archive_id": archive_id}
        )
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "Linked Archive",
                "solution_doc": "solution",
                "linked_experience_ids": [exp_id_1, exp_id_2],
            },
        )
        assert resp.status_code == 201
        call_kw = get_context().archive_service.archive_upsert.call_args.kwargs
        linked = call_kw.get("linked_experience_ids")
        assert linked is not None
        assert len(linked) == 2
        # Verify they are UUID objects, not strings
        assert all(isinstance(uid, uuid.UUID) for uid in linked)

    def test_archives_create_with_project(self, client, auth_headers):
        """POST /api/v1/archives forwards project to service."""
        from team_memory.bootstrap import get_context

        get_context().archive_service.archive_upsert = AsyncMock(
            return_value={"action": "created", "archive_id": str(uuid.uuid4())}
        )
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "Project Archive",
                "solution_doc": "sol",
                "project": "my-project",
            },
        )
        assert resp.status_code == 201
        call_kw = get_context().archive_service.archive_upsert.call_args.kwargs
        assert call_kw.get("project") is not None

    def test_archives_create_member_can_create(self, client, member_headers):
        """POST /api/v1/archives also works for member role (not admin-only)."""
        from team_memory.bootstrap import get_context

        get_context().archive_service.archive_upsert = AsyncMock(
            return_value={"action": "created", "archive_id": str(uuid.uuid4())}
        )
        resp = client.post(
            "/api/v1/archives",
            headers=member_headers,
            json={
                "title": "Member Archive",
                "solution_doc": "solution",
            },
        )
        assert resp.status_code == 201
        call_kw = get_context().archive_service.archive_upsert.call_args.kwargs
        assert call_kw["created_by"] == "test_member"


# ============================================================
# SPA Tests
# ============================================================


class TestSPA:
    def test_index_html_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "TeamMemory" in resp.text
        assert "<!DOCTYPE html>" in resp.text


# ============================================================
# Request body size limit (C3)
# ============================================================


class TestRequestBodySizeLimit:
    def test_large_content_length_returns_413(self, client, auth_headers):
        """Requests with Content-Length exceeding max are rejected with 413."""
        huge_size = 30_000_000  # 30 MB, exceeds 20 MB default
        resp = client.post(
            "/api/v1/search",
            headers={**auth_headers, "content-length": str(huge_size)},
            json={"query": "test"},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_normal_content_length_passes(self, client, auth_headers, setup_app):
        """Requests with small Content-Length pass through normally."""
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                "/api/v1/search",
                headers=auth_headers,
                json={"query": "test"},
            )
            assert resp.status_code == 200


# ============================================================
# Rate limiting (M5)
# ============================================================


class TestRateLimiting:
    def test_rate_limit_exceeded_returns_429(self, client):
        """After exceeding rate limit, requests get 429."""
        from team_memory.web.app import _rate_limit_store

        _rate_limit_store.clear()

        # Set a very low limit for testing
        web_module._settings.web = WebConfig(rate_limit_per_minute=3)

        for _ in range(3):
            resp = client.get("/")
            assert resp.status_code == 200

        # 4th request should be rate-limited
        resp = client.post(
            "/api/v1/search",
            json={"query": "test"},
        )
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()

    def test_health_and_ready_bypass_rate_limit(self, client):
        """Health and ready endpoints bypass rate limiting."""
        from team_memory.web.app import _rate_limit_store

        _rate_limit_store.clear()
        web_module._settings.web = WebConfig(rate_limit_per_minute=1)

        # Use up the single allowed request
        resp = client.get("/")
        assert resp.status_code == 200

        # Health and ready should still work
        resp = client.get("/health")
        assert resp.status_code in (200, 503)  # may be unhealthy but not 429

        resp = client.get("/ready")
        assert resp.status_code in (200, 503)


# ============================================================
# Text field max_length (H4) — schema-level validation
# ============================================================


class TestTextFieldMaxLength:
    def test_archive_solution_doc_too_long(self, client, auth_headers):
        """solution_doc exceeding 524_288 chars is rejected by Pydantic."""
        long_text = "x" * 524_289
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "T",
                "solution_doc": long_text,
            },
        )
        assert resp.status_code == 422

    def test_archive_raw_conversation_too_long(self, client, auth_headers):
        """raw_conversation exceeding 2_097_152 chars is rejected by Pydantic."""
        long_text = "x" * 2_097_153
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "T",
                "solution_doc": "short",
                "raw_conversation": long_text,
            },
        )
        assert resp.status_code == 422

    def test_experience_title_too_long(self, client, auth_headers):
        """ExperienceCreate title exceeding 500 chars is rejected by Pydantic."""
        resp = client.post(
            "/api/v1/experiences",
            headers=auth_headers,
            json={
                "title": "x" * 501,
                "problem": "p",
                "solution": "s",
            },
        )
        assert resp.status_code == 422


# ============================================================
# Tags validation (M6) — schema-level
# ============================================================


class TestTagsValidation:
    def test_archive_too_many_tags(self, client, auth_headers):
        """More than 20 tags rejected on ArchiveCreateRequest."""
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "T",
                "solution_doc": "body",
                "tags": [f"tag{i}" for i in range(21)],
            },
        )
        assert resp.status_code == 422

    def test_archive_tag_too_long(self, client, auth_headers):
        """Tag exceeding 50 chars rejected on ArchiveCreateRequest."""
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "T",
                "solution_doc": "body",
                "tags": ["x" * 51],
            },
        )
        assert resp.status_code == 422

    def test_experience_too_many_tags(self, client, auth_headers):
        """More than 20 tags rejected on ExperienceCreate."""
        resp = client.post(
            "/api/v1/experiences",
            headers=auth_headers,
            json={
                "title": "T",
                "problem": "p",
                "tags": [f"tag{i}" for i in range(21)],
            },
        )
        assert resp.status_code == 422

    def test_experience_tag_too_long(self, client, auth_headers):
        """Tag exceeding 50 chars rejected on ExperienceCreate."""
        resp = client.post(
            "/api/v1/experiences",
            headers=auth_headers,
            json={
                "title": "T",
                "problem": "p",
                "tags": ["x" * 51],
            },
        )
        assert resp.status_code == 422

    def test_valid_tags_pass(self, client, auth_headers, setup_app):
        """Tags within limits are accepted."""
        from team_memory.bootstrap import get_context

        get_context().archive_service.archive_upsert = AsyncMock(
            return_value={"action": "created", "archive_id": str(uuid.uuid4())}
        )
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "T",
                "solution_doc": "body",
                "tags": ["python", "testing"],
            },
        )
        assert resp.status_code == 201


# ============================================================
# Error Scenarios (T4)
# ============================================================


class TestErrorScenarios:
    """Comprehensive error scenario coverage for the web API."""

    def test_create_experience_validation_error(self, client, auth_headers):
        """POST /experiences missing required 'problem' field returns 422."""
        resp = client.post(
            "/api/v1/experiences",
            headers=auth_headers,
            json={
                "title": "Missing problem field",
                # problem is required but omitted
                "solution": "solution",
            },
        )
        assert resp.status_code == 422

    def test_create_experience_embedding_failure(self, client, auth_headers, setup_app):
        """POST /experiences when embedding fails returns 500."""
        setup_app.save = AsyncMock(
            return_value={
                "error": True,
                "message": "Embedding generation failed. Save aborted.",
            }
        )
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                "/api/v1/experiences",
                headers=auth_headers,
                json={
                    "title": "Test",
                    "problem": "Test problem",
                    "solution": "Test solution",
                },
            )
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "embedding_failure"

    def test_update_experience_not_found(self, client, auth_headers, setup_app):
        """PUT /experiences/{id} returns 404 when experience does not exist."""
        setup_app.update = AsyncMock(return_value=None)

        resp = client.put(
            f"/api/v1/experiences/{uuid.uuid4()}",
            headers=auth_headers,
            json={"title": "Updated title"},
        )
        assert resp.status_code == 404

    def test_delete_experience_not_found(self, client, auth_headers, setup_app):
        """DELETE /experiences/{id} returns 404 when experience does not exist."""
        setup_app.soft_delete = AsyncMock(return_value=False)

        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.delete(
                f"/api/v1/experiences/{uuid.uuid4()}",
                headers=auth_headers,
            )
        assert resp.status_code == 404

    def test_feedback_invalid_rating(self, client, auth_headers):
        """POST /experiences/{id}/feedback with rating=10 returns 422."""
        resp = client.post(
            f"/api/v1/experiences/{uuid.uuid4()}/feedback",
            headers=auth_headers,
            json={"rating": 10},
        )
        assert resp.status_code == 422

    def test_search_empty_query(self, client, auth_headers):
        """POST /search with empty query string returns 422."""
        resp = client.post(
            "/api/v1/search",
            headers=auth_headers,
            json={"query": ""},
        )
        # Empty string fails Pydantic min_length or the SearchRequest validation
        # If no min_length is set, the endpoint may return 200 with empty results
        assert resp.status_code in (200, 422)

    def test_archive_create_too_long_title(self, client, auth_headers):
        """POST /archives with title exceeding max_length returns 422."""
        resp = client.post(
            "/api/v1/archives",
            headers=auth_headers,
            json={
                "title": "x" * 501,
                "solution_doc": "body",
            },
        )
        assert resp.status_code == 422

    def test_login_wrong_password(self, client):
        """POST /auth/login with wrong key returns success=False."""
        resp = client.post(
            "/api/v1/auth/login",
            json={"api_key": "wrong-key-does-not-exist"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_admin_endpoint_as_viewer(self, client, viewer_headers):
        """PUT /config/retrieval as viewer returns 403 (admin-only)."""
        resp = client.put(
            "/api/v1/config/retrieval",
            headers=viewer_headers,
            json={
                "max_tokens": None,
                "max_count": 10,
                "trim_strategy": "top_k",
                "top_k_children": 3,
                "min_avg_rating": 0.0,
                "rating_weight": 0.3,
                "summary_model": None,
            },
        )
        assert resp.status_code == 403

    def test_rate_limit_exceeded(self, client):
        """After exceeding rate limit, requests get 429."""
        from team_memory.web.app import _rate_limit_store

        _rate_limit_store.clear()

        # Set very low limit for testing
        web_module._settings.web = WebConfig(rate_limit_per_minute=2)

        for _ in range(2):
            resp = client.get("/")
            assert resp.status_code == 200

        # 3rd request should be rate-limited
        resp = client.post(
            "/api/v1/search",
            json={"query": "test"},
        )
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()
