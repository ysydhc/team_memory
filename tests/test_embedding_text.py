"""Tests for Step 2: embedding text construction improvements.

Covers:
1. save() embed_text includes tags
2. save_group() parent embed_text includes children titles and tags
3. save_group() child embed_text includes tags
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.services.experience import ExperienceService


def _make_service(embedding_mock=None):
    """Create an ExperienceService with mock dependencies."""
    if embedding_mock is None:
        embedding_mock = AsyncMock()
        embedding_mock.encode_single = AsyncMock(return_value=[0.1] * 768)

    event_bus = MagicMock()
    event_bus.emit = AsyncMock()
    auth_provider = MagicMock()

    return ExperienceService(
        embedding_provider=embedding_mock,
        auth_provider=auth_provider,
        event_bus=event_bus,
    )


class TestSaveEmbedTextIncludesTags:
    """save() should include tags in the embedding text."""

    @pytest.mark.asyncio
    async def test_embed_text_contains_tags(self):
        """Tags should be appended to the embedding text."""
        embedding_mock = AsyncMock()
        embedding_mock.encode_single = AsyncMock(return_value=[0.1] * 768)
        service = _make_service(embedding_mock)

        mock_repo = MagicMock()
        mock_exp = MagicMock()
        mock_exp.id = "test-id"
        mock_exp.to_dict.return_value = {"id": "test-id", "title": "Test"}
        mock_exp.children = []
        mock_repo.create = AsyncMock(return_value=mock_exp)
        mock_repo.check_similar = AsyncMock(return_value=[])

        mock_session = AsyncMock()

        with patch(
            "team_memory.services.experience.ExperienceRepository",
            return_value=mock_repo,
        ):
            await service.save(
                title="Docker networking fix",
                problem="Container cannot connect",
                solution="Use bridge network",
                tags=["docker", "networking", "container"],
                created_by="test",
                session=mock_session,
            )

        embed_call = embedding_mock.encode_single.call_args[0][0]
        assert "docker" in embed_call
        assert "networking" in embed_call
        assert "container" in embed_call

    @pytest.mark.asyncio
    async def test_embed_text_without_tags(self):
        """When tags is None, embedding should still work."""
        embedding_mock = AsyncMock()
        embedding_mock.encode_single = AsyncMock(return_value=[0.1] * 768)
        service = _make_service(embedding_mock)

        mock_repo = MagicMock()
        mock_exp = MagicMock()
        mock_exp.id = "test-id"
        mock_exp.to_dict.return_value = {"id": "test-id", "title": "Test"}
        mock_exp.children = []
        mock_repo.create = AsyncMock(return_value=mock_exp)
        mock_repo.check_similar = AsyncMock(return_value=[])

        mock_session = AsyncMock()

        with patch(
            "team_memory.services.experience.ExperienceRepository",
            return_value=mock_repo,
        ):
            await service.save(
                title="Simple fix",
                problem="Something broke",
                tags=None,
                created_by="test",
                session=mock_session,
            )

        embedding_mock.encode_single.assert_called_once()


class TestSaveGroupEmbedText:
    """save_group() should include children titles in parent embedding."""

    @pytest.mark.asyncio
    async def test_parent_embed_includes_children_titles(self):
        """Parent embedding text should contain all children titles."""
        embedding_mock = AsyncMock()
        embedding_mock.encode_single = AsyncMock(return_value=[0.1] * 768)
        service = _make_service(embedding_mock)

        mock_repo = MagicMock()
        mock_parent = MagicMock()
        mock_parent.id = "parent-id"
        mock_parent.children = []
        mock_parent.to_dict.return_value = {"id": "parent-id", "title": "Parent"}
        mock_repo.create_group = AsyncMock(return_value=mock_parent)

        mock_session = AsyncMock()

        parent = {
            "title": "事故复盘",
            "problem": "生产数据被误删",
            "solution": "恢复备份",
            "tags": ["database", "incident"],
        }
        children = [
            {
                "title": "集成测试硬删除生产数据库全部经验",
                "problem": "测试代码误删生产数据",
                "solution": "添加环境检查",
                "tags": ["testing"],
            },
            {
                "title": "数据库备份恢复流程",
                "problem": "缺少自动备份",
                "solution": "配置定时备份",
            },
        ]

        with patch(
            "team_memory.services.experience.ExperienceRepository",
            return_value=mock_repo,
        ):
            await service.save_group(
                parent=parent,
                children=children,
                created_by="test",
                session=mock_session,
            )

        # First call is parent embedding
        parent_embed_call = embedding_mock.encode_single.call_args_list[0][0][0]
        assert "集成测试硬删除生产数据库全部经验" in parent_embed_call
        assert "数据库备份恢复流程" in parent_embed_call
        assert "database" in parent_embed_call
        assert "incident" in parent_embed_call

    @pytest.mark.asyncio
    async def test_child_embed_includes_tags(self):
        """Child embedding text should contain its own tags."""
        embedding_mock = AsyncMock()
        embedding_mock.encode_single = AsyncMock(return_value=[0.1] * 768)
        service = _make_service(embedding_mock)

        mock_repo = MagicMock()
        mock_parent = MagicMock()
        mock_parent.id = "parent-id"
        mock_child = MagicMock()
        mock_child.id = "child-id"
        mock_parent.children = [mock_child]
        mock_parent.to_dict.return_value = {"id": "parent-id"}
        mock_repo.create_group = AsyncMock(return_value=mock_parent)

        mock_session = AsyncMock()

        parent = {"title": "Parent", "problem": "P", "solution": "S"}
        children = [
            {
                "title": "Child fix",
                "problem": "Child problem",
                "solution": "Child solution",
                "tags": ["python", "bugfix"],
            }
        ]

        with patch(
            "team_memory.services.experience.ExperienceRepository",
            return_value=mock_repo,
        ):
            await service.save_group(
                parent=parent,
                children=children,
                created_by="test",
                session=mock_session,
            )

        # Second call is child embedding
        child_embed_call = embedding_mock.encode_single.call_args_list[1][0][0]
        assert "python" in child_embed_call
        assert "bugfix" in child_embed_call
