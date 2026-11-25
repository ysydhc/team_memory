"""Tests for PersonalMemory governance: scope routing, budgeted injection, RLS isolation, and cleanup."""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

from team_memory.services.memory.ops.recall import op_recall
from team_memory.services.memory.personal_memory import PersonalMemoryService
from team_memory.storage.database import close_db, get_session
from team_memory.storage.models import PersonalMemory, Tenant
from team_memory.storage.personal_memory_repository import normalize_scope
from team_memory.storage.tenant_context import tenant_scope
from team_memory.utils.tokenizer import estimate_tokens

# Allow importing scripts/govern_personal_memories.py
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from govern_personal_memories import (  # noqa: E402
    GovernanceRecord,
    build_governance_plan,
    classify_scope_and_project,
)

PG_TEST_URL = os.environ.get(
    "TEAM_MEMORY_DB_URL",
    "postgresql+asyncpg://developer:devpass@localhost:5433/team_memory",
)


@dataclass
class _FakeRow:
    id: uuid.UUID
    user_id: str
    content: str
    scope: str = "global"
    project: str | None = None
    profile_kind: str = "static"
    persona: str = "general"
    priority: int = 70
    context_hint: str | None = None
    _similarity: float = 0.5


class TestScopeNormalization:
    """Verify canonical global/project normalization and legacy compatibility."""

    def test_legacy_generic_maps_to_global(self) -> None:
        assert normalize_scope("generic") == ("global", None, "static")

    def test_legacy_context_without_project_maps_to_global_dynamic(self) -> None:
        assert normalize_scope("context") == ("global", None, "dynamic")

    def test_project_scope_with_project_preserved(self) -> None:
        assert normalize_scope("project", project="adx-sdk-android") == (
            "project",
            "adx-sdk-android",
            "static",
        )

    def test_project_scope_without_project_falls_back_to_global(self) -> None:
        assert normalize_scope("project", project=None) == ("global", None, "static")


class TestScopeRoutingAndBudgetedInjection:
    """Verify scope/project routing, Top-K & token budget, and embedding reuse."""

    @pytest.mark.asyncio
    async def test_project_routing_isolates_other_projects_and_boosts_match(self) -> None:
        rows = [
            _FakeRow(
                id=uuid.uuid4(),
                user_id="alice",
                content="跨项目通用习惯：先调研再改代码",
                scope="global",
                project=None,
                priority=75,
                _similarity=0.60,
            ),
            _FakeRow(
                id=uuid.uuid4(),
                user_id="alice",
                content="Android 专属习惯：排查上报先看 Logcat Funnel 埋点",
                scope="project",
                project="adx-sdk-android",
                priority=75,
                _similarity=0.60,
            ),
            _FakeRow(
                id=uuid.uuid4(),
                user_id="alice",
                content="iOS 专属习惯：检查 PrivacyInfo.xcprivacy",
                scope="project",
                project="adx-sdk-ios",
                priority=95,
                _similarity=0.95,
            ),
        ]
        mock_emb = MagicMock()
        mock_emb.encode_single = AsyncMock(return_value=[0.1] * 8)
        svc = PersonalMemoryService(embedding_provider=mock_emb, db_url="test://")

        mock_repo = MagicMock()
        mock_repo.list_for_pull = AsyncMock(return_value=rows)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = MagicMock()
        mock_ctx.__aexit__.return_value = False

        with (
            patch.object(svc, "_session", return_value=mock_ctx),
            patch(
                "team_memory.services.memory.personal_memory.PersonalMemoryRepository",
                return_value=mock_repo,
            ),
        ):
            profile_android = await svc.build_profile_for_user(
                user_id="alice",
                current_context="排查埋点问题",
                project="adx-sdk-android",
                max_items=8,
                max_tokens=400,
            )
            profile_global_only = await svc.build_profile_for_user(
                user_id="alice",
                current_context="通用问题",
                project=None,
                max_items=8,
                max_tokens=400,
            )

        # Android context: gets global + android (android ranked first due to project bonus), never iOS
        assert len(profile_android["static"]) == 2
        assert "Logcat Funnel" in profile_android["static"][0]
        assert "先调研再改代码" in profile_android["static"][1]
        assert all("PrivacyInfo" not in s for s in profile_android["static"])

        # No project context: only global returned
        assert profile_global_only["static"] == ["跨项目通用习惯：先调研再改代码"]

    @pytest.mark.asyncio
    async def test_token_budget_and_item_cap_enforced(self) -> None:
        # 15 items, each ~40 Chinese chars (~40 tokens)
        rows = [
            _FakeRow(
                id=uuid.uuid4(),
                user_id="alice",
                content=f"这是一条编号{i:02d}的长期稳定个人工程偏好描述，用于测试预算截断控制是否生效。",
                scope="global",
                priority=90 - i,
                _similarity=0.8 - i * 0.02,
            )
            for i in range(15)
        ]
        mock_emb = MagicMock()
        mock_emb.encode_single = AsyncMock(return_value=[0.1] * 8)
        svc = PersonalMemoryService(embedding_provider=mock_emb, db_url="test://")

        mock_repo = MagicMock()
        mock_repo.list_for_pull = AsyncMock(return_value=rows)
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = MagicMock()
        mock_ctx.__aexit__.return_value = False

        with (
            patch.object(svc, "_session", return_value=mock_ctx),
            patch(
                "team_memory.services.memory.personal_memory.PersonalMemoryRepository",
                return_value=mock_repo,
            ),
        ):
            profile = await svc.build_profile_for_user(
                user_id="alice",
                current_context="代码规范",
                max_items=8,
                max_tokens=150,
            )

        total_tokens = sum(estimate_tokens(s) for s in profile["static"])
        assert 1 <= len(profile["static"]) <= 8
        assert total_tokens <= 150

    @pytest.mark.asyncio
    async def test_precomputed_context_embedding_skips_rpc(self) -> None:
        mock_emb = MagicMock()
        mock_emb.encode_single = AsyncMock(return_value=[0.9] * 8)
        svc = PersonalMemoryService(embedding_provider=mock_emb, db_url="test://")

        mock_repo = MagicMock()
        mock_repo.list_for_pull = AsyncMock(return_value=[])
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = MagicMock()
        mock_ctx.__aexit__.return_value = False

        with (
            patch.object(svc, "_session", return_value=mock_ctx),
            patch(
                "team_memory.services.memory.personal_memory.PersonalMemoryRepository",
                return_value=mock_repo,
            ),
        ):
            await svc.build_profile_for_user(
                user_id="alice",
                current_context="some query",
                context_embedding=[0.2] * 8,
            )

        mock_emb.encode_single.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_op_recall_defaults_include_user_profile_false(self) -> None:
        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(
            return_value=MagicMock(results=[], reranked=False, intent_type="general")
        )
        mock_settings = MagicMock()
        mock_settings.mcp.profile_max_strings_per_side = 8
        mock_settings.mcp.profile_max_items = 8
        mock_settings.mcp.profile_max_tokens = 400

        with patch("team_memory.services.memory.ops.recall._maybe_write_search_log", new=AsyncMock()):
            res = await op_recall(
                user="alice",
                search_orchestrator=mock_orchestrator,
                db_url="test://",
                settings=mock_settings,
                embedding_provider=MagicMock(),
                query="test query",
                project="team_doc",
                include_archives=False,
            )
        assert res["profile"] is None


class TestGovernanceScriptLogic:
    """Unit tests for scripts/govern_personal_memories.py classification & dedup."""

    def test_classify_scope_and_project(self) -> None:
        assert classify_scope_and_project("排查问题时坚持先定位根因") == ("global", None)
        assert classify_scope_and_project("习惯分析 Logcat 中的 Funnel 埋点") == (
            "project",
            "adx-sdk-android",
        )
        assert classify_scope_and_project("使用 tm-cli status 监控 tm-daemon 状态") == (
            "project",
            "team_doc",
        )

    def test_build_governance_plan_prunes_and_merges(self) -> None:
        now = datetime.now(timezone.utc)
        records = [
            # Expired dynamic
            GovernanceRecord(
                id=uuid.uuid4(),
                tenant_id="t1",
                user_id="alice",
                content="正在调试某个临时接口",
                scope="generic",
                project=None,
                profile_kind="dynamic",
                persona="debugging",
                priority=50,
                embedding=[1.0, 0.0, 0.0],
                created_at=now - timedelta(days=120),
                updated_at=now - timedelta(days=120),
            ),
            # Ephemeral noise
            GovernanceRecord(
                id=uuid.uuid4(),
                tenant_id="t1",
                user_id="alice",
                content="把间距设置为 24px 解决卡片互相遮挡",
                scope="generic",
                project=None,
                profile_kind="static",
                persona="general",
                priority=50,
                embedding=[0.0, 1.0, 0.0],
                created_at=now,
                updated_at=now,
            ),
            # Duplicate pair in global
            GovernanceRecord(
                id=uuid.uuid4(),
                tenant_id="t1",
                user_id="alice",
                content="重大方案决策前要求提供量化基准与打分",
                scope="generic",
                project=None,
                profile_kind="static",
                persona="research",
                priority=80,
                embedding=[0.9, 0.1, 0.0],
                created_at=now,
                updated_at=now,
            ),
            GovernanceRecord(
                id=uuid.uuid4(),
                tenant_id="t1",
                user_id="alice",
                content="评估项目优化效果时，倾向于提供量化基准和打分对比",
                scope="generic",
                project=None,
                profile_kind="static",
                persona="research",
                priority=70,
                embedding=[0.89, 0.12, 0.0],
                created_at=now,
                updated_at=now,
            ),
            # Project-scoped android habit
            GovernanceRecord(
                id=uuid.uuid4(),
                tenant_id="t1",
                user_id="alice",
                content="排查 Android 埋点时优先查看 Logcat 中的 Funnel 日志",
                scope="generic",
                project=None,
                profile_kind="static",
                persona="debugging",
                priority=75,
                embedding=[0.0, 0.0, 1.0],
                created_at=now,
                updated_at=now,
            ),
        ]
        plan = build_governance_plan(records, dedup_threshold=0.80, dynamic_ttl_days=90, now=now)
        assert plan.total_before == 5
        assert plan.total_after == 2
        assert len(plan.expired_dynamic_ids) == 1
        assert len(plan.ephemeral_noise_ids) == 1
        assert len(plan.merges) == 1
        assert plan.scope_after == {"global": 1, "project": 1}
        assert plan.project_after == {"adx-sdk-android": 1}


async def _pg_available() -> bool:
    try:
        async with get_session(PG_TEST_URL, bypass_rls=True) as s:
            await s.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_pg_rls_personal_memories_strict_isolation() -> None:
    """Verify strict single-tenant RLS on personal_memories for both global and project scopes."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not reachable in this environment")

    tid_a = uuid.uuid4()
    tid_b = uuid.uuid4()
    pm_global_a = uuid.uuid4()
    pm_proj_a = uuid.uuid4()

    try:
        async with get_session(PG_TEST_URL, bypass_rls=True) as session:
            session.add(
                Tenant(id=tid_a, slug=f"pm-rls-a-{tid_a.hex[:8]}", name="PM A", settings_json={})
            )
            session.add(
                Tenant(id=tid_b, slug=f"pm-rls-b-{tid_b.hex[:8]}", name="PM B", settings_json={})
            )

        with tenant_scope(tid_a):
            async with get_session(PG_TEST_URL) as session:
                session.add(
                    PersonalMemory(
                        id=pm_global_a,
                        tenant_id=tid_a,
                        user_id="shared_name",
                        content="Tenant A global preference",
                        scope="global",
                        project=None,
                        profile_kind="static",
                    )
                )
                session.add(
                    PersonalMemory(
                        id=pm_proj_a,
                        tenant_id=tid_a,
                        user_id="shared_name",
                        content="Tenant A project secret habit",
                        scope="project",
                        project="secret-proj",
                        profile_kind="static",
                    )
                )

        # Tenant B (even with the exact same user_id='shared_name') sees ZERO rows from Tenant A
        with tenant_scope(tid_b):
            async with get_session(PG_TEST_URL) as session:
                rows = (
                    await session.execute(
                        text("SELECT id FROM personal_memories WHERE id IN (:g, :p)"),
                        {"g": pm_global_a, "p": pm_proj_a},
                    )
                ).scalars().all()
                assert rows == []
    finally:
        async with get_session(PG_TEST_URL, bypass_rls=True) as session:
            await session.execute(
                text("DELETE FROM tenants WHERE id IN (:a, :b)"),
                {"a": tid_a, "b": tid_b},
            )
        await close_db()
