"""Phase 1 Multi-Tenant & PostgreSQL RLS Verification Suite (TC-01 ~ TC-03).

Tests:
  - TC-01: Backward compatibility (default tenant, legacy API keys, cache isolation)
  - TC-02: Hard PostgreSQL RLS isolation across Tenant A / Tenant B (raw SQL + ORM + WITH CHECK)
  - TC-03: RBAC permission boundaries (viewer / member / admin / forged X-Tenant-Id)
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from team_memory.auth.permissions import has_permission
from team_memory.auth.provider import ApiKeyAuth
from team_memory.services.search.cache import SearchCache
from team_memory.storage.database import close_db, get_session
from team_memory.storage.models import Entity, Experience, Tenant, TenantMember
from team_memory.storage.tenant_context import (
    DEFAULT_TENANT_ID,
    get_current_tenant_id,
    tenant_scope,
)
from team_memory.web import app as app_module

PG_TEST_URL = os.environ.get(
    "TEAM_MEMORY_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5433/team_memory",
)


# ============================================================
# Unit-level TC-01 & TC-03 (no external DB required)
# ============================================================


class TestTenantContextAndPermissions:
    """TC-01 & TC-03 unit checks for tenant_context, SearchCache, and RBAC."""

    def test_default_tenant_fallback(self) -> None:
        assert get_current_tenant_id() == DEFAULT_TENANT_ID

    def test_tenant_scope_nested_and_reset(self) -> None:
        t1 = uuid.uuid4()
        t2 = uuid.uuid4()
        with tenant_scope(t1):
            assert get_current_tenant_id() == t1
            with tenant_scope(t2):
                assert get_current_tenant_id() == t2
            assert get_current_tenant_id() == t1
        assert get_current_tenant_id() == DEFAULT_TENANT_ID

    def test_search_cache_key_isolates_tenants(self) -> None:
        t1 = uuid.uuid4()
        t2 = uuid.uuid4()
        with tenant_scope(t1):
            k1 = SearchCache._make_key("docker compose", project="default", current_user="alice")
        with tenant_scope(t2):
            k2 = SearchCache._make_key("docker compose", project="default", current_user="alice")
        assert k1 != k2

    def test_rbac_matrix(self) -> None:
        # Admin: full access
        assert has_permission("admin", "read")
        assert has_permission("admin", "create")
        assert has_permission("admin", "admin")
        # Member & legacy editor: read + write, no admin
        for role in ("member", "editor"):
            assert has_permission(role, "read")
            assert has_permission(role, "search")
            assert has_permission(role, "create")
            assert has_permission(role, "update")
            assert not has_permission(role, "admin")
        # Viewer: read/search/feedback only
        assert has_permission("viewer", "read")
        assert has_permission("viewer", "search")
        assert has_permission("viewer", "feedback")
        assert not has_permission("viewer", "create")
        assert not has_permission("viewer", "update")
        assert not has_permission("viewer", "delete")
        assert not has_permission("viewer", "admin")


class TestWebRBACAndTenantHeaders:
    """TC-03: HTTP-level RBAC and X-Tenant-Id rejection tests."""

    def _make_client_with_auth(self) -> tuple[TestClient, dict[str, str]]:
        from fastapi import APIRouter

        from team_memory.web.routes import mount_all

        auth = ApiKeyAuth()
        tid_a = str(uuid.uuid4())
        tid_b = str(uuid.uuid4())
        auth.register_key("key-admin-a", "alice", role="admin", tenant_id=tid_a)
        auth.register_key("key-member-a", "bob", role="member", tenant_id=tid_a)
        auth.register_key("key-viewer-a", "carol", role="viewer", tenant_id=tid_a)
        auth.register_key("key-admin-b", "dave", role="admin", tenant_id=tid_b)

        # Ensure v1 routes are mounted (lifespan may not have run in unit tests)
        existing_paths = {getattr(r, "path", "") for r in app_module.app.routes}
        if "/api/v1/tenants" not in existing_paths:
            v1_router = APIRouter(prefix="/api/v1")
            mount_all(v1_router)
            app_module.app.include_router(v1_router)

        mock_ctx = MagicMock()
        mock_ctx.auth = auth
        mock_ctx.db_url = "sqlite+aiosqlite:///:memory:"
        mock_ctx.settings = MagicMock()
        mock_ctx.settings.auth.session_secret = "test-secret"
        mock_ctx.settings.default_project = "default"
        mock_ctx.service = MagicMock()
        mock_ctx.janitor = None

        app_module._auth = auth
        return TestClient(app_module.app), {
            "tid_a": tid_a,
            "tid_b": tid_b,
            "mock_ctx": mock_ctx,
        }

    def test_viewer_cannot_create_experience(self) -> None:
        client, info = self._make_client_with_auth()
        with patch("team_memory.bootstrap.get_context", return_value=info["mock_ctx"]):
            resp = client.post(
                "/api/v1/experiences",
                headers={"Authorization": "Bearer key-viewer-a"},
                json={"title": "t", "problem": "p", "solution": "s"},
            )
        assert resp.status_code == 403

    def test_member_cannot_access_tenant_admin_or_ops(self) -> None:
        client, info = self._make_client_with_auth()
        with patch("team_memory.bootstrap.get_context", return_value=info["mock_ctx"]):
            r1 = client.get(
                "/api/v1/tenants/current/members",
                headers={"Authorization": "Bearer key-member-a"},
            )
            r2 = client.post(
                "/api/v1/tenants/current/members",
                headers={"Authorization": "Bearer key-member-a"},
                json={"user_id": "newbie", "role": "viewer"},
            )
            r3 = client.put(
                "/api/v1/tenants/current",
                headers={"Authorization": "Bearer key-member-a"},
                json={"name": "Hacked"},
            )
            r4 = client.post(
                "/api/v1/janitor/run",
                headers={"Authorization": "Bearer key-member-a"},
            )
        assert r1.status_code == 403
        assert r2.status_code == 403
        assert r3.status_code == 403
        assert r4.status_code == 403

    def test_api_key_rejects_mismatched_x_tenant_id(self) -> None:
        client, info = self._make_client_with_auth()
        with patch("team_memory.bootstrap.get_context", return_value=info["mock_ctx"]):
            resp = client.get(
                "/api/v1/auth/me",
                headers={
                    "Authorization": "Bearer key-admin-a",
                    "X-Tenant-Id": info["tid_b"],
                },
            )
        assert resp.status_code == 403

    def test_auth_me_returns_tenant_id(self) -> None:
        client, info = self._make_client_with_auth()
        with patch("team_memory.bootstrap.get_context", return_value=info["mock_ctx"]):
            resp = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer key-admin-a"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user"] == "alice"
        assert body["role"] == "admin"
        assert body["tenant_id"] == info["tid_a"]


# ============================================================
# Integration TC-01 & TC-02 against Real PostgreSQL (RLS)
# ============================================================


async def _pg_available() -> bool:
    try:
        async with get_session(PG_TEST_URL, bypass_rls=True) as s:
            await s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_pg_rls_hard_isolation_tc01_tc02() -> None:
    """TC-01 + TC-02 against real PostgreSQL with tm_app_user + FORCE RLS."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not reachable in this environment")

    slug_a = f"test-rls-a-{uuid.uuid4().hex[:8]}"
    slug_b = f"test-rls-b-{uuid.uuid4().hex[:8]}"
    tid_a = uuid.uuid4()
    tid_b = uuid.uuid4()
    exp_a_id = uuid.uuid4()
    ent_a_id = uuid.uuid4()

    try:
        # 0. TC-01: Verify default tenant exists
        async with get_session(PG_TEST_URL, bypass_rls=True) as session:
            default_t = await session.execute(
                select(Tenant).where(Tenant.id == DEFAULT_TENANT_ID)
            )
            assert default_t.scalar_one_or_none() is not None

            # Seed Tenant A and Tenant B
            session.add(Tenant(id=tid_a, slug=slug_a, name="Tenant A", settings_json={}))
            session.add(Tenant(id=tid_b, slug=slug_b, name="Tenant B", settings_json={}))
            await session.flush()
            session.add(TenantMember(tenant_id=tid_a, user_id="user_a", role="admin"))
            session.add(TenantMember(tenant_id=tid_b, user_id="user_b", role="admin"))

        # 1. Tenant A writes an Experience and an Entity under RLS
        with tenant_scope(tid_a):
            async with get_session(PG_TEST_URL) as session:
                # Verify session is actually running as non-superuser tm_app_user
                cur_user = (await session.execute(text("SELECT current_user"))).scalar_one()
                assert cur_user == "tm_app_user"

                session.add(
                    Experience(
                        id=exp_a_id,
                        tenant_id=tid_a,
                        title="Tenant A Secret Runbook",
                        description="Internal secret problem for tenant A",
                        solution="Secret fix A",
                        created_by="user_a",
                        project="rls-test",
                        exp_status="published",
                    )
                )
                session.add(
                    Entity(
                        id=ent_a_id,
                        tenant_id=tid_a,
                        name=f"SecretEntity-{slug_a}",
                        entity_type="concept",
                        project="rls-test",
                    )
                )

        # 2. TC-02: Tenant A can read its own records
        with tenant_scope(tid_a):
            async with get_session(PG_TEST_URL) as session:
                row = (
                    await session.execute(
                        text("SELECT id FROM experiences WHERE id = :id"),
                        {"id": exp_a_id},
                    )
                ).scalar_one_or_none()
                assert row == exp_a_id

                ent = (
                    await session.execute(
                        text("SELECT id FROM entities WHERE id = :id"),
                        {"id": ent_a_id},
                    )
                ).scalar_one_or_none()
                assert ent == ent_a_id

        # 3. TC-02: Tenant B CANNOT see Tenant A's records EVEN WITHOUT WHERE tenant_id
        with tenant_scope(tid_b):
            async with get_session(PG_TEST_URL) as session:
                row_b = (
                    await session.execute(
                        text("SELECT id FROM experiences WHERE id = :id"),
                        {"id": exp_a_id},
                    )
                ).scalar_one_or_none()
                assert row_b is None

                ent_b = (
                    await session.execute(
                        text("SELECT id FROM entities WHERE id = :id"),
                        {"id": ent_a_id},
                    )
                ).scalar_one_or_none()
                assert ent_b is None

                # Same entity name in Tenant B must succeed (tenant-scoped unique constraint)
                session.add(
                    Entity(
                        id=uuid.uuid4(),
                        tenant_id=tid_b,
                        name=f"SecretEntity-{slug_a}",
                        entity_type="concept",
                        project="rls-test",
                    )
                )

        # 4. TC-02: Tenant B attempting to INSERT with Tenant A's tenant_id is BLOCKED by WITH CHECK
        with tenant_scope(tid_b):
            with pytest.raises(DBAPIError):
                async with get_session(PG_TEST_URL) as session:
                    session.add(
                        Experience(
                            id=uuid.uuid4(),
                            tenant_id=tid_a,  # forged tenant_id!
                            title="Cross-tenant injection",
                            description="Should be rejected by PG RLS WITH CHECK",
                            created_by="user_b",
                            project="rls-test",
                        )
                    )
                    await session.flush()
    finally:
        # Clean up test tenants (CASCADE deletes all test rows)
        async with get_session(PG_TEST_URL, bypass_rls=True) as session:
            await session.execute(
                text("DELETE FROM tenants WHERE id IN (:a, :b)"),
                {"a": tid_a, "b": tid_b},
            )
        await close_db()
