"""Tests for the FastAPI web application.

Tests auth, experience CRUD, search, and stats endpoints.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from team_memory.auth.provider import ApiKeyAuth
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
    auth.register_key("member-key", "test_member", "member")
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
    web_module._settings.pageindex_lite = MagicMock(
        enabled=True,
        only_long_docs=True,
        min_doc_chars=800,
        max_tree_depth=4,
        max_nodes_per_doc=40,
        max_node_chars=1200,
        tree_weight=0.15,
        min_node_score=0.01,
        include_matched_nodes=True,
    )
    web_module._settings.installable_catalog = MagicMock(
        sources=["local"],
        local_base_dir=".debug/knowledge-pack",
        registry_manifest_url="",
        target_rules_dir=".cursor/rules",
        target_prompts_dir=".cursor/prompts",
        request_timeout_seconds=8,
    )

    mock_service = MagicMock()
    mock_service.search = AsyncMock(return_value=[])
    mock_service.save = AsyncMock(return_value={
        "id": str(uuid.uuid4()),
        "title": "Test",
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    mock_service.feedback = AsyncMock(return_value=True)
    mock_service.update = AsyncMock(return_value={
        "id": str(uuid.uuid4()),
        "title": "Updated",
        "solution": "Updated solution",
        "tags": ["test"],
    })
    mock_service.get_recent = AsyncMock(return_value=[])
    mock_service.get_stats = AsyncMock(return_value={
        "total_experiences": 5,
        "tag_distribution": {"python": 3, "docker": 2},
        "recent_7days": 2,
    })
    mock_service.soft_delete = AsyncMock(return_value=True)
    mock_service.restore = AsyncMock(return_value=True)
    mock_service.review = AsyncMock(return_value={
        "id": str(uuid.uuid4()),
        "title": "Reviewed",
        "review_status": "approved",
        "publish_status": "published",
    })
    mock_service.get_pending_reviews = AsyncMock(return_value=[])
    mock_service.get_query_logs = AsyncMock(return_value=[])
    mock_service.get_query_stats = AsyncMock(return_value={
        "total_queries": 0,
        "recent_7days": 0,
        "search_type_distribution": {},
    })
    web_module._service = mock_service

    yield mock_service

    # Clean up
    web_module._auth = None
    web_module._settings = None
    web_module._service = None


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-key-123"}


@pytest.fixture
def member_headers():
    return {"Authorization": "Bearer member-key"}


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


# ============================================================
# Experience CRUD Tests
# ============================================================

class TestExperienceList:
    def test_list_anonymous_access(self, client):
        """List experiences supports anonymous access (no auth required)."""
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("team_memory.web.app.ExperienceRepository") as MockRepo:
                mock_repo = MagicMock()
                mock_repo.count = AsyncMock(return_value=0)
                mock_repo.list_recent = AsyncMock(return_value=[])
                MockRepo.return_value = mock_repo

                resp = client.get("/api/v1/experiences")
                assert resp.status_code == 200
                data = resp.json()
                assert "experiences" in data

    def test_list_experiences(self, client, auth_headers):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("team_memory.web.app.ExperienceRepository") as MockRepo:
                mock_repo = MagicMock()
                mock_repo.count = AsyncMock(return_value=2)
                mock_repo.list_recent = AsyncMock(return_value=[])
                MockRepo.return_value = mock_repo

                resp = client.get("/api/v1/experiences", headers=auth_headers)
                assert resp.status_code == 200
                data = resp.json()
                assert "experiences" in data
                assert data["total"] == 2
                assert data["page"] == 1


class TestCreateExperience:
    def test_create_requires_auth(self, client):
        resp = client.post("/api/v1/experiences", json={"title": "t", "problem": "p", "solution": "s"})
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
            assert "id" in resp.json()

            setup_app.save.assert_called_once()
            kwargs = setup_app.save.call_args.kwargs
            assert kwargs["created_by"] == "test_admin"
            assert kwargs["source"] == "web"


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
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("team_memory.web.app.ExperienceRepository") as MockRepo:
                mock_repo = MagicMock()
                mock_repo.delete = AsyncMock(return_value=True)
                MockRepo.return_value = mock_repo

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
        setup_app.search = AsyncMock(return_value=[
            {"id": str(uuid.uuid4()), "title": "Result", "similarity": 0.85}
        ])

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


class TestPageIndexConfig:
    def test_get_pageindex_config(self, client, auth_headers):
        resp = client.get("/api/v1/config/pageindex-lite", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "tree_weight" in data

    def test_update_pageindex_config_admin(self, client, auth_headers):
        resp = client.put(
            "/api/v1/config/pageindex-lite",
            headers=auth_headers,
            json={
                "enabled": True,
                "only_long_docs": True,
                "min_doc_chars": 1000,
                "max_tree_depth": 3,
                "max_nodes_per_doc": 30,
                "max_node_chars": 1000,
                "tree_weight": 0.2,
                "min_node_score": 0.02,
                "include_matched_nodes": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["tree_weight"] == 0.2


class TestInstallables:
    def test_list_installables(self, client, auth_headers):
        mock_catalog = MagicMock()
        item = MagicMock()
        item.model_dump.return_value = {
            "id": "rule.demo",
            "type": "rule",
            "name": "demo",
            "version": "1.0.0",
            "source": "local",
        }
        mock_catalog.list_items = AsyncMock(return_value=[item])
        with patch("team_memory.web.app._get_catalog_service", return_value=mock_catalog):
            resp = client.get("/api/v1/installables", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == "rule.demo"

    def test_preview_installable(self, client, auth_headers):
        mock_catalog = MagicMock()
        mock_catalog.preview = AsyncMock(
            return_value={"item": {"id": "rule.demo"}, "content": "abc", "truncated": False}
        )
        with patch("team_memory.web.app._get_catalog_service", return_value=mock_catalog):
            resp = client.get(
                "/api/v1/installables/preview?id=rule.demo&source=local",
                headers=auth_headers,
            )
        assert resp.status_code == 200
        assert resp.json()["content"] == "abc"

    def test_install_requires_admin(self, client, member_headers):
        resp = client.post(
            "/api/v1/installables/install",
            headers=member_headers,
            json={"id": "rule.demo", "source": "local"},
        )
        assert resp.status_code == 403

    def test_install_success(self, client, auth_headers):
        mock_catalog = MagicMock()
        mock_catalog.install = AsyncMock(
            return_value={
                "installed": True,
                "item": {"id": "rule.demo"},
                "target_path": ".cursor/rules/demo.mdc",
            }
        )
        with patch("team_memory.web.app._get_catalog_service", return_value=mock_catalog):
            resp = client.post(
                "/api/v1/installables/install",
                headers=auth_headers,
                json={"id": "rule.demo", "source": "local"},
            )
        assert resp.status_code == 200
        assert resp.json()["installed"] is True


class TestProjectParam:
    def test_search_uses_default_project(self, client, setup_app):
        web_module._settings.default_project = "proj-default"
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = client.post("/api/v1/search", json={"query": "hello"})
        assert resp.status_code == 200
        assert setup_app.search.call_args.kwargs["project"] == "proj-default"

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
# Stats Tests
# ============================================================

class TestStats:
    def test_stats_anonymous_access(self, client, setup_app):
        """Stats supports anonymous access."""
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.get("/api/v1/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_experiences"] == 5

    def test_stats_with_auth(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.get("/api/v1/stats", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_experiences"] == 5


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

class TestReview:
    def test_list_pending_requires_auth(self, client):
        resp = client.get("/api/v1/reviews/pending")
        assert resp.status_code == 401

    def test_list_pending(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.get("/api/v1/reviews/pending", headers=auth_headers)
            assert resp.status_code == 200
            assert "experiences" in resp.json()

    def test_review_experience(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.post(
                f"/api/v1/experiences/{uuid.uuid4()}/review",
                headers=auth_headers,
                json={"review_status": "approved", "review_note": "LGTM"},
            )
            assert resp.status_code == 200

    def test_review_invalid_status(self, client, auth_headers):
        resp = client.post(
            f"/api/v1/experiences/{uuid.uuid4()}/review",
            headers=auth_headers,
            json={"review_status": "invalid"},
        )
        assert resp.status_code == 400


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
# Query Logs Tests
# ============================================================

class TestQueryLogs:
    def test_query_logs_requires_auth(self, client):
        resp = client.get("/api/v1/query-logs")
        assert resp.status_code == 401

    def test_query_logs(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.get("/api/v1/query-logs", headers=auth_headers)
            assert resp.status_code == 200
            assert "logs" in resp.json()

    def test_query_stats_requires_auth(self, client):
        resp = client.get("/api/v1/query-stats")
        assert resp.status_code == 401

    def test_query_stats(self, client, auth_headers, setup_app):
        with patch("team_memory.web.app.get_session") as mock_gs:
            mock_sess = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = client.get("/api/v1/query-stats", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "total_queries" in data


# ============================================================
# Parse Document Tests
# ============================================================

class TestParseDocument:
    def test_parse_requires_auth(self, client):
        resp = client.post("/api/v1/experiences/parse-document", json={"content": "some doc"})
        assert resp.status_code == 401

    def test_parse_empty_content(self, client, auth_headers):
        resp = client.post(
            "/api/v1/experiences/parse-document",
            headers=auth_headers,
            json={"content": "   "},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_parse_success(self, client, auth_headers):
        """Test successful document parsing with mocked llm_parser."""
        parsed_result = {
            "title": "Docker端口冲突解决方案",
            "problem": "在使用Docker部署FastAPI时，端口8000被占用导致容器启动失败",
            "solution": "通过修改docker-compose.yml中的端口映射解决",
            "tags": ["docker", "fastapi", "devops"],
            "language": "python",
            "framework": "fastapi",
            "code_snippets": "ports:\n  - '8001:8000'"
        }

        with patch("team_memory.services.llm_parser.parse_content", new_callable=AsyncMock, return_value=parsed_result):
            resp = client.post(
                "/api/v1/experiences/parse-document",
                headers=auth_headers,
                json={"content": "# Docker端口冲突\n部署FastAPI时8000端口被占用..."},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["title"] == "Docker端口冲突解决方案"
            assert "docker" in data["tags"]
            assert data["language"] == "python"
            assert data["framework"] == "fastapi"

    def test_parse_handles_markdown_fenced_json(self, client, auth_headers):
        """Test that the endpoint handles LLM results correctly."""
        parsed_result = {
            "title": "Test",
            "problem": "Test problem",
            "solution": "Test solution",
            "tags": ["test"],
            "language": None,
            "framework": None,
            "code_snippets": None,
        }

        with patch("team_memory.services.llm_parser.parse_content", new_callable=AsyncMock, return_value=parsed_result):
            resp = client.post(
                "/api/v1/experiences/parse-document",
                headers=auth_headers,
                json={"content": "some document text"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["title"] == "Test"
            assert data["tags"] == ["test"]

    def test_parse_ollama_unavailable(self, client, auth_headers):
        """Test error when Ollama is not running."""
        from team_memory.services.llm_parser import LLMParseError

        with patch("team_memory.services.llm_parser.parse_content", new_callable=AsyncMock, side_effect=LLMParseError("Cannot connect to Ollama")):
            resp = client.post(
                "/api/v1/experiences/parse-document",
                headers=auth_headers,
                json={"content": "some doc"},
            )
            assert resp.status_code == 503


# ============================================================
# Parse URL Tests
# ============================================================

class TestParseURL:
    def test_parse_url_requires_auth(self, client):
        resp = client.post("/api/v1/experiences/parse-url", json={"url": "https://example.com"})
        assert resp.status_code == 401

    def test_parse_url_empty(self, client, auth_headers):
        resp = client.post(
            "/api/v1/experiences/parse-url",
            headers=auth_headers,
            json={"url": "  "},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_parse_url_invalid_scheme(self, client, auth_headers):
        resp = client.post(
            "/api/v1/experiences/parse-url",
            headers=auth_headers,
            json={"url": "ftp://example.com/file"},
        )
        assert resp.status_code == 400
        assert "http" in resp.json()["detail"].lower()

    def test_parse_url_success_html(self, client, auth_headers):
        """Test URL parsing with an HTML page."""
        html_content = """
        <html><head><title>Fix Docker Issue</title></head>
        <body>
        <h1>Docker Port Conflict</h1>
        <p>Port 8000 was already in use when deploying FastAPI.</p>
        <h2>Solution</h2>
        <p>Changed port mapping in docker-compose.yml to 8001:8000.</p>
        </body></html>
        """

        parsed_result = {
            "title": "Docker端口冲突",
            "problem": "部署FastAPI时8000端口被占用",
            "solution": "修改docker-compose端口映射",
            "tags": ["docker", "fastapi"],
            "language": None,
            "framework": "fastapi",
            "code_snippets": None,
        }

        # Mock URL fetch response
        mock_url_resp = MagicMock()
        mock_url_resp.status_code = 200
        mock_url_resp.raise_for_status = MagicMock()
        mock_url_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_url_resp.text = html_content

        with patch("team_memory.web.app.httpx") as mock_httpx, \
             patch("team_memory.services.llm_parser.parse_content", new_callable=AsyncMock, return_value=parsed_result):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_url_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.ConnectError = ConnectionError
            mock_httpx.HTTPStatusError = Exception
            mock_httpx.TimeoutException = TimeoutError
            mock_httpx.InvalidURL = ValueError

            resp = client.post(
                "/api/v1/experiences/parse-url",
                headers=auth_headers,
                json={"url": "https://example.com/blog/docker-fix"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["title"] == "Docker端口冲突"
            assert "docker" in data["tags"]

    def test_parse_url_success_markdown(self, client, auth_headers):
        """Test URL parsing with a markdown file."""
        md_content = "# Fix\n\n## Problem\nSomething broke.\n\n## Solution\nFixed it."

        parsed_result = {
            "title": "Fix",
            "problem": "Something broke.",
            "solution": "Fixed it.",
            "tags": ["fix"],
            "language": None,
            "framework": None,
            "code_snippets": None,
        }

        mock_url_resp = MagicMock()
        mock_url_resp.status_code = 200
        mock_url_resp.raise_for_status = MagicMock()
        mock_url_resp.headers = {"content-type": "text/markdown"}
        mock_url_resp.text = md_content

        with patch("team_memory.web.app.httpx") as mock_httpx, \
             patch("team_memory.services.llm_parser.parse_content", new_callable=AsyncMock, return_value=parsed_result):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_url_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.ConnectError = ConnectionError
            mock_httpx.HTTPStatusError = Exception
            mock_httpx.TimeoutException = TimeoutError
            mock_httpx.InvalidURL = ValueError

            resp = client.post(
                "/api/v1/experiences/parse-url",
                headers=auth_headers,
                json={"url": "https://example.com/doc.md"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["title"] == "Fix"

    def test_parse_url_connect_error(self, client, auth_headers):
        """Test error when URL cannot be reached."""
        import httpx as real_httpx

        with patch("team_memory.web.app.httpx") as mock_httpx:
            mock_httpx.ConnectError = real_httpx.ConnectError
            mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
            mock_httpx.TimeoutException = real_httpx.TimeoutException
            mock_httpx.InvalidURL = real_httpx.InvalidURL
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=real_httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            resp = client.post(
                "/api/v1/experiences/parse-url",
                headers=auth_headers,
                json={"url": "https://unreachable.example.com"},
            )
            assert resp.status_code == 502
            assert "connect" in resp.json()["detail"].lower()

    def test_parse_url_empty_content(self, client, auth_headers):
        """Test error when URL returns empty content."""
        mock_url_resp = MagicMock()
        mock_url_resp.status_code = 200
        mock_url_resp.raise_for_status = MagicMock()
        mock_url_resp.headers = {"content-type": "text/plain"}
        mock_url_resp.text = "   "

        with patch("team_memory.web.app.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_url_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.ConnectError = ConnectionError
            mock_httpx.HTTPStatusError = Exception
            mock_httpx.TimeoutException = TimeoutError
            mock_httpx.InvalidURL = ValueError

            resp = client.post(
                "/api/v1/experiences/parse-url",
                headers=auth_headers,
                json={"url": "https://example.com/empty"},
            )
            assert resp.status_code == 502
            assert "empty" in resp.json()["detail"].lower()


# ============================================================
# SPA Tests
# ============================================================

class TestSPA:
    def test_index_html_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "team_memory" in resp.text
        assert "<!DOCTYPE html>" in resp.text
