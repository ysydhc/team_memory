"""Tests for MCP HTTP compat routes (/api/v1/mcp/*) and tm-cli argument parsing."""

from __future__ import annotations

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from team_memory.auth.provider import ApiKeyAuth
from team_memory.config import UploadsConfig, WebConfig
from team_memory.services.search_orchestrator import OrchestratedSearchResult
from team_memory.web import app as web_module
from team_memory.web.app import app

from team_memory import cli as cli_module

# ============================================================
# Fixtures (aligned with tests/test_web.py — local copy so we do not
# pytest_plugins-import test_web, which would register autouse setup_app
# for the entire suite and break test_server mocks).
# ============================================================


@pytest.fixture(autouse=True)
def setup_app():
    """Set up app globals with mock service and real auth for each test."""
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
        local_base_dir=".tm_cursor/installables",
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
    mock_search_orchestrator.search = AsyncMock(return_value=OrchestratedSearchResult(results=[]))
    mock_search_orchestrator.invalidate_cache = AsyncMock(return_value=None)
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
        patch("team_memory.services.memory_operations.get_context", return_value=mock_ctx),
        patch("team_memory.web.app.start_background_tasks", new_callable=AsyncMock),
        patch("team_memory.web.app.stop_background_tasks", new_callable=AsyncMock),
    ):
        yield mock_service

    web_module._auth = None
    web_module._settings = None
    web_module._service = None
    from team_memory.web.app import _rate_limit_store

    _rate_limit_store.clear()


@pytest.fixture
def client(setup_app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-key-123"}


_MCP = "team_memory.web.routes.mcp_compat.memory_operations"


class TestMcpCompatRoutes:
    """POST/GET /api/v1/mcp/* — orchestration mocked at memory_operations."""

    def test_mcp_save_success(self, client, auth_headers):
        payload = {
            "message": "Knowledge saved successfully.",
            "data": {"id": "exp-1", "title": "T", "status": "published"},
        }
        with patch(f"{_MCP}.op_save", new_callable=AsyncMock, return_value=payload):
            resp = client.post(
                "/api/v1/mcp/save",
                headers={**auth_headers, "Content-Type": "application/json"},
                json={
                    "title": "Hello",
                    "problem": "P",
                    "solution": "S",
                },
            )
        assert resp.status_code == 200
        assert resp.json() == payload

    def test_mcp_recall_success(self, client, auth_headers):
        payload = {
            "message": "Found 1 result(s).",
            "results": [{"id": "x", "title": "Hit"}],
            "reranked": False,
            "profile": None,
        }
        with patch(f"{_MCP}.op_recall", new_callable=AsyncMock, return_value=payload):
            resp = client.post(
                "/api/v1/mcp/recall",
                headers={**auth_headers, "Content-Type": "application/json"},
                json={"query": "test query", "max_results": 5},
            )
        assert resp.status_code == 200
        assert resp.json() == payload

    def test_mcp_context_success(self, client, auth_headers):
        payload = {
            "user": "test_admin",
            "project": "default",
            "profile": {"static": ["a"], "dynamic": []},
            "relevant_experiences": [],
        }
        with patch(f"{_MCP}.op_context", new_callable=AsyncMock, return_value=payload):
            resp = client.post(
                "/api/v1/mcp/context",
                headers={**auth_headers, "Content-Type": "application/json"},
                json={"file_paths": ["src/foo.py"]},
            )
        assert resp.status_code == 200
        assert resp.json() == payload

    def test_mcp_get_archive_success(self, client, auth_headers):
        aid = str(uuid.uuid4())
        payload = {
            "id": aid,
            "title": "Arch",
            "solution_doc": "body",
            "attachments": [],
            "document_tree_nodes": [],
        }
        with patch(f"{_MCP}.op_get_archive", new_callable=AsyncMock, return_value=payload):
            resp = client.get(
                f"/api/v1/mcp/archive/{aid}",
                headers=auth_headers,
            )
        assert resp.status_code == 200
        assert resp.json() == payload

    def test_mcp_archive_upsert_success(self, client, auth_headers):
        aid = str(uuid.uuid4())
        payload = {
            "archive_id": aid,
            "action": "created",
            "message": "Created successfully",
            "item": {"archive_id": aid, "action": "created"},
        }
        with patch(f"{_MCP}.op_archive_upsert", new_callable=AsyncMock, return_value=payload):
            resp = client.post(
                "/api/v1/mcp/archive-upsert",
                headers={**auth_headers, "Content-Type": "application/json"},
                json={
                    "title": "T",
                    "solution_doc": "doc",
                },
            )
        assert resp.status_code == 200
        assert resp.json() == payload

    def test_mcp_feedback_success(self, client, auth_headers):
        payload = {"message": "Feedback recorded. Thank you!"}
        with patch(f"{_MCP}.op_feedback", new_callable=AsyncMock, return_value=payload):
            resp = client.post(
                "/api/v1/mcp/feedback",
                headers={**auth_headers, "Content-Type": "application/json"},
                json={"experience_id": str(uuid.uuid4()), "rating": 5},
            )
        assert resp.status_code == 200
        assert resp.json() == payload

    def test_mcp_save_no_auth(self, client):
        resp = client.post(
            "/api/v1/mcp/save",
            json={"title": "x", "problem": "y", "solution": "z"},
        )
        assert resp.status_code == 401

    def test_mcp_recall_empty_results(self, client, auth_headers):
        payload = {
            "message": "No matching experiences found.",
            "results": [],
            "reranked": False,
            "profile": None,
        }
        with patch(f"{_MCP}.op_recall", new_callable=AsyncMock, return_value=payload):
            resp = client.post(
                "/api/v1/mcp/recall",
                headers={**auth_headers, "Content-Type": "application/json"},
                json={"query": "zzzznomatchzzzz"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == []


class TestCliParsing:
    """tm-cli — no HTTP; argv validation only."""

    _SUBCOMMANDS = (
        "archive",
        "upload",
        "save",
        "recall",
        "context",
        "get-archive",
        "feedback",
    )

    def test_cli_help_shows_all_commands(self, capsys):
        with patch.object(sys, "argv", ["tm-cli", "--help"]):
            with pytest.raises(SystemExit) as exc:
                cli_module.main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        for name in self._SUBCOMMANDS:
            assert name in out

    def test_cli_save_requires_title_or_content(self):
        with patch.object(sys, "argv", ["tm-cli", "save"]):
            with pytest.raises(SystemExit) as exc:
                cli_module.main()
            assert exc.value.code == 1
