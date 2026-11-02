"""Unit tests for Task 8: Write-Path Non-Blocking Personal Memory Extraction,
Bi-Temporal Config & PersonalMemory Static Profile Cap Protection.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.config.lifecycle import LifecycleConfig
from team_memory.services.memory.experience import ExperienceService
from team_memory.services.memory.mutation_resolver import MutationDecision
from team_memory.services.memory.ops.save import _save_from_content
from team_memory.services.memory.personal_memory import PersonalMemoryService
from team_memory.storage.personal_memory_repository import PersonalMemoryRepository


class TestLifecycleAutoResolveMutations:
    """Test LifecycleConfig.auto_resolve_mutations and ExperienceService integration."""

    def test_auto_resolve_mutations_default_and_override(self):
        cfg_default = LifecycleConfig()
        assert cfg_default.auto_resolve_mutations is False

        cfg_enabled = LifecycleConfig(auto_resolve_mutations=True)
        assert cfg_enabled.auto_resolve_mutations is True

    @pytest.mark.asyncio
    async def test_experience_service_save_triggers_mutation_resolver_via_config(self):
        mock_embedding = MagicMock()
        mock_embedding.encode_single = AsyncMock(return_value=[0.1] * 8)
        mock_auth = MagicMock()
        cfg = LifecycleConfig(auto_resolve_mutations=True)

        svc = ExperienceService(
            embedding_provider=mock_embedding,
            auth_provider=mock_auth,
            lifecycle_config=cfg,
        )

        target_id = str(uuid.uuid4())
        mock_exp = MagicMock()
        mock_exp.to_dict.return_value = {"id": target_id, "title": "Existing"}

        mock_repo = MagicMock()
        mock_repo.check_similar = AsyncMock(
            return_value=[{"id": target_id, "title": "Existing", "similarity": 0.92}]
        )
        mock_repo.touch_verified = AsyncMock(return_value=mock_exp)

        mock_session = AsyncMock()
        with (
            patch(
                "team_memory.services.memory.experience.ExperienceRepository",
                return_value=mock_repo,
            ),
            patch(
                "team_memory.services.memory.mutation_resolver.MutationResolver.resolve",
                new_callable=AsyncMock,
                return_value=MutationDecision(
                    action="NOOP",
                    target_id=target_id,
                    reason="Already covered",
                ),
            ) as mock_resolve,
        ):
            out = await svc.save(
                title="New Title",
                problem="New Problem",
                solution="New Solution",
                session=mock_session,
            )

        mock_resolve.assert_awaited_once()
        mock_repo.touch_verified.assert_awaited_once()
        assert out["mutation_action"] == "NOOP"
        assert out["target_id"] == target_id


class TestNonBlockingPersonalMemoryExtraction:
    """Test non-blocking background personal memory extraction in _save_from_content."""

    @pytest.mark.asyncio
    async def test_save_from_content_does_not_block_on_slow_personal_memory(self):
        started = asyncio.Event()
        finished = asyncio.Event()

        async def _slow_extract(*args, **kwargs):
            started.set()
            await asyncio.sleep(0.5)
            finished.set()

        mock_service = MagicMock()
        mock_service.save = AsyncMock(
            return_value={
                "id": str(uuid.uuid4()),
                "title": "Parsed Title",
                "tags": ["python"],
                "exp_status": "draft",
                "mutation_action": "ADD",
            }
        )
        mock_settings = MagicMock()
        mock_settings.extraction.quality_gate = 2
        mock_settings.extraction.max_retries = 0

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "team_memory.services.llm.llm_parser.parse_content",
                new_callable=AsyncMock,
                return_value={
                    "title": "Parsed Title",
                    "problem": "Parsed Problem",
                    "solution": "Parsed Solution",
                    "tags": ["python"],
                },
            ),
            patch(
                "team_memory.services.memory.ops.save.get_session",
                return_value=mock_session_cm,
            ),
            patch(
                "team_memory.services.memory.ops.save._try_extract_and_save_personal_memory",
                side_effect=_slow_extract,
            ),
        ):
            t0 = time.monotonic()
            res = await _save_from_content(
                content="User conversation content",
                tags=["test"],
                user="alice",
                project="default",
                settings=mock_settings,
                service=mock_service,
                db_url="sqlite://",
                embedding_provider=MagicMock(),
            )
            elapsed = time.monotonic() - t0

            assert elapsed < 0.1, f"Expected non-blocking return (<0.1s), took {elapsed:.3f}s"
            assert started.is_set(), "Background task should have started immediately"
            assert not finished.is_set(), "Background task should still be running"
            assert res["data"]["mutation_action"] == "ADD"

            await asyncio.wait_for(finished.wait(), timeout=1.0)
            assert finished.is_set()

    @pytest.mark.asyncio
    async def test_save_from_content_compatible_with_memory_operations_facade_patch(self):
        mock_service = MagicMock()
        mock_service.save = AsyncMock(
            return_value={
                "id": str(uuid.uuid4()),
                "title": "T",
                "tags": [],
                "exp_status": "draft",
            }
        )
        mock_settings = MagicMock()
        mock_settings.extraction = None

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "team_memory.services.llm.llm_parser.parse_content",
                new_callable=AsyncMock,
                return_value={"title": "T", "problem": "P", "solution": "S", "tags": []},
            ),
            patch(
                "team_memory.services.memory.ops.save.get_session",
                return_value=mock_session_cm,
            ),
            patch(
                "team_memory.services.memory.memory_operations._try_extract_and_save_personal_memory",
                new_callable=AsyncMock,
            ) as mock_facade_extract,
        ):
            await _save_from_content(
                content="Some conversation",
                tags=None,
                user="bob",
                project="default",
                settings=mock_settings,
                service=mock_service,
                db_url="sqlite://",
                embedding_provider=MagicMock(),
            )
            mock_facade_extract.assert_awaited_once()


class TestPersonalMemoryCapProtection:
    """Test static profile cap protection in PersonalMemoryRepository and Service."""

    @pytest.mark.asyncio
    async def test_delete_oldest_by_persona_orders_dynamic_then_priority_then_created_at(self):
        captured_queries = []
        target_uuid = uuid.uuid4()

        async def _capture_execute(stmt):
            captured_queries.append(stmt)
            mock_res = MagicMock()
            mock_res.scalar_one_or_none.return_value = target_uuid
            return mock_res

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=_capture_execute)

        repo = PersonalMemoryRepository(mock_session)
        deleted = await repo.delete_oldest_by_persona("alice", "general")

        assert deleted is True
        assert len(captured_queries) == 2
        select_sql = str(
            captured_queries[0].compile(compile_kwargs={"literal_binds": True})
        )
        order_clause = select_sql.split("ORDER BY")[-1]
        assert "CASE WHEN (personal_memories.profile_kind = 'dynamic') THEN 0 ELSE 1 END ASC" in order_clause
        assert "personal_memories.priority ASC" in order_clause
        assert "personal_memories.created_at ASC" in order_clause

    @pytest.mark.asyncio
    async def test_enforce_persona_cap_helper(self):
        svc = PersonalMemoryService(embedding_provider=MagicMock(), db_url="sqlite://")
        mock_repo = MagicMock()
        mock_repo.count_by_persona = AsyncMock(return_value=100)
        mock_repo.delete_oldest_by_persona = AsyncMock(return_value=True)

        evicted = await svc._enforce_persona_cap(mock_repo, "alice", "general", cap=100)
        assert evicted is True
        mock_repo.delete_oldest_by_persona.assert_awaited_once_with("alice", "general")

        mock_repo.count_by_persona = AsyncMock(return_value=99)
        mock_repo.delete_oldest_by_persona.reset_mock()
        evicted_below = await svc._enforce_persona_cap(mock_repo, "alice", "general", cap=100)
        assert evicted_below is False
        mock_repo.delete_oldest_by_persona.assert_not_awaited()
