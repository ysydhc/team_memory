"""Unit tests for User Persona optimization:
1. De-contextualized extraction (_is_ephemeral_task_memory)
2. Frequency-based priority boost on semantic overwrite (upsert_by_semantic)
3. Hybrid vector + keyword + priority Top-K ranking (build_profile_for_user)
4. Trivial confirmation noise filtering ("可以,开始优化", "进入phase2")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.daemon.integration.personal_memory_extractor import (
    PersonalMemoryExtractor,
    _is_ephemeral_task_memory,
)
from team_memory.daemon.refinement.query_noise import NoiseReason, classify_noise_rule
from team_memory.daemon.runtime.config import DaemonConfig
from team_memory.services.memory.personal_memory import PersonalMemoryService
from team_memory.storage.personal_memory_repository import (
    PERSONAL_MEMORY_OVERWRITE_THRESHOLD,
    PersonalMemoryRepository,
)


class TestEphemeralTaskMemoryGuard:
    """Verify _is_ephemeral_task_memory rejects one-off feature/bug tweaks."""

    @pytest.mark.parametrize(
        "ephemeral_text",
        [
            "对落地页交互行为有明确控制要求，倾向于支持自动触发特定类型的跳转（如302跳转）",
            "在交互设计上，倾向于将复杂的展开卡片简化为紧凑的下拉菜单",
            "偏好在文档中同时展示多个头部客户案例（如高德与滴滴）以增强说服力",
            "关注UI细节，要求解决卡片互相遮挡及标题重合问题",
            "在UI布局调整中，关注间距的紧凑性，倾向于减小模块间的间距（如设置为 24px）",
        ],
    )
    def test_rejects_ephemeral_task_details(self, ephemeral_text: str) -> None:
        assert _is_ephemeral_task_memory(ephemeral_text) is True

    @pytest.mark.parametrize(
        "lasting_habit",
        [
            "在开发过程中倾向于使用 /spec-copilot 辅助完成技术方案与验收",
            "排查问题时，要求提供链路可视化（图表+文字）以符合直觉，并区分数据问题与功能问题",
            "偏好使用 CLI 工具进行交互，要求将常用诊断能力封装为命令行子命令",
            "习惯在决策前进行充分讨论，倾向于先梳理逻辑和交互流程再动手修改代码",
        ],
    )
    def test_accepts_lasting_engineering_habits(self, lasting_habit: str) -> None:
        assert _is_ephemeral_task_memory(lasting_habit) is False

    @pytest.mark.asyncio
    async def test_extract_and_save_filters_ephemeral_and_forwards_priority(self) -> None:
        cfg = DaemonConfig()
        extractor = PersonalMemoryExtractor(config=cfg)
        with (
            patch.object(
                extractor,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "category": "preference",
                        "content": "倾向于将复杂的展开卡片简化为紧凑的下拉菜单",
                        "persona": "general",
                        "priority": 60,
                    },
                    {
                        "category": "habit",
                        "content": "偏好使用 CLI 工具进行交互，将诊断能力封装为命令行工具",
                        "persona": "coding",
                        "priority": 80,
                    },
                ],
            ),
            patch.object(extractor, "_save_memory", new_callable=AsyncMock) as mock_save,
        ):
            saved = await extractor.extract_and_save(
                session_id="sess-12345678",
                user_messages=[
                    {"content": "请把诊断能力封装成 tm-cli 子命令，方便我们在终端直接调用检查运行状态"},
                    {"content": "另外支持 watch 模式持续观察四个平面的核心指标"},
                ],
            )

        assert saved == 1
        mock_save.assert_awaited_once_with(
            "habit",
            "偏好使用 CLI 工具进行交互，将诊断能力封装为命令行工具",
            persona="coding",
            priority=80,
        )


@dataclass
class _FakePersonalMemoryRow:
    id: uuid.UUID
    user_id: str
    content: str
    scope: str = "generic"
    context_hint: str | None = None
    profile_kind: str = "static"
    persona: str = "general"
    priority: int = 50
    _similarity: float = 0.0
    updated_at: Any = None
    created_at: Any = None


class TestPriorityBoostAndHybridRanking:
    """Verify semantic overwrite priority boost and hybrid Top-K profile ranking."""

    def test_overwrite_threshold_lowered_for_paraphrase_dedup(self) -> None:
        assert PERSONAL_MEMORY_OVERWRITE_THRESHOLD == 0.82

    @pytest.mark.asyncio
    async def test_upsert_by_semantic_boosts_priority_on_repeat_hit(self) -> None:
        existing_row = _FakePersonalMemoryRow(
            id=uuid.uuid4(),
            user_id="yeshouyou",
            content="在开发过程中倾向于使用 /spec-copilot",
            priority=60,
        )
        mock_session = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()

        repo = PersonalMemoryRepository(mock_session)
        with (
            patch.object(repo, "find_most_similar", new_callable=AsyncMock) as mock_find,
            patch.object(repo, "get_by_id", new_callable=AsyncMock) as mock_get,
        ):
            mock_find.return_value = existing_row
            mock_get.return_value = existing_row
            updated = await repo.upsert_by_semantic(
                user_id="yeshouyou",
                content="复杂需求开发倾向于使用 /spec-copilot 进行分阶段推进",
                embedding=[0.1] * 8,
                priority=65,
            )

        assert updated is existing_row
        # max(60, 65) + 10 = 75
        assert existing_row.priority == 75
        assert existing_row.content == "复杂需求开发倾向于使用 /spec-copilot 进行分阶段推进"

    @pytest.mark.asyncio
    async def test_build_profile_for_user_ranks_by_vector_priority_and_keywords(self) -> None:
        rows = [
            _FakePersonalMemoryRow(
                id=uuid.uuid4(),
                user_id="u1",
                content="最近一条低优无关记忆 A",
                priority=50,
                _similarity=0.10,
            ),
            _FakePersonalMemoryRow(
                id=uuid.uuid4(),
                user_id="u1",
                content="最近一条低优无关记忆 B",
                priority=50,
                _similarity=0.12,
            ),
            _FakePersonalMemoryRow(
                id=uuid.uuid4(),
                user_id="u1",
                content="在开发过程中倾向于使用 /spec-copilot 辅助方案设计与验收",
                priority=90,
                _similarity=0.78,
            ),
            _FakePersonalMemoryRow(
                id=uuid.uuid4(),
                user_id="u1",
                content="偏好使用 CLI 工具进行交互，通过命令行查看系统状态",
                priority=85,
                _similarity=0.65,
            ),
        ]

        mock_embedding = MagicMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.5] * 8)

        svc = PersonalMemoryService(
            embedding_provider=mock_embedding,
            db_url="postgresql+asyncpg://localhost/test",
        )

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
                user_id="u1",
                current_context="请使用 spec-copilot 帮我规划 CLI 方案",
                max_items=2,
            )

        assert len(profile["static"]) == 2
        assert "/spec-copilot" in profile["static"][0]
        assert "CLI 工具" in profile["static"][1]


class TestTrivialConfirmationNoise:
    """Verify short confirmation phrases are filtered as trivial noise."""

    @pytest.mark.parametrize(
        "query",
        [
            "可以,开始优化",
            "可以，开始执行",
            "好,继续修改",
            "进入phase2",
            "进入 phase 1",
        ],
    )
    def test_short_confirmation_directives_are_trivial(self, query: str) -> None:
        assert classify_noise_rule(query) == NoiseReason.RULE
