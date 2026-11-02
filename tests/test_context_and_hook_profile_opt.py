"""Unit tests for Task 9: Hook before_prompt Dynamic Profile Activation &
memory_context Concurrent Execution + L1 Summary.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import team_memory.daemon.refinement.pipeline as pipeline_mod
from team_memory.daemon.refinement.pipeline import (
    _get_personal_memory_service,
    process_before_prompt,
)
from team_memory.daemon.runtime.config import DaemonConfig, ProjectMapping
from team_memory.services.memory.ops.context import op_context


class TestHookBeforePromptDynamicProfile:
    """Test before_prompt dynamic personal memory activation and PMS caching."""

    @pytest.mark.asyncio
    async def test_before_prompt_passes_current_context_and_injects_dynamic_memories(
        self, monkeypatch
    ):
        monkeypatch.setattr("team_memory.project_resolver.read_tm_toml", lambda _: None)
        monkeypatch.setattr(
            "team_memory.project_resolver.infer_project_name", lambda _cwd="": "default"
        )

        config = DaemonConfig(
            projects=[ProjectMapping(name="team_doc", path_patterns=["team_doc"])]
        )
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value=[{"id": "exp-1", "title": "Fix DB pool"}])
        buf = AsyncMock()

        mock_pms = MagicMock()
        mock_pms.build_profile_for_user = AsyncMock(
            return_value={
                "static": ["Always use async SQLAlchemy"],
                "dynamic": ["Prefer connection pool size 10 for team_doc"],
            }
        )

        prompt_text = "How should we configure the database connection pool in this service?"
        with patch.object(
            pipeline_mod, "_get_personal_memory_service", return_value=mock_pms
        ):
            res = await process_before_prompt(
                {
                    "prompt": prompt_text,
                    "workspace_roots": ["/work/team_doc"],
                    "conversation_id": "conv-123",
                },
                config,
                sink,
                buf,
            )

        mock_pms.build_profile_for_user.assert_awaited_once()
        call_kwargs = mock_pms.build_profile_for_user.await_args.kwargs
        assert call_kwargs["current_context"] == prompt_text
        assert call_kwargs["user_id"] == "admin"
        assert "persona" in call_kwargs

        assert "Always use async SQLAlchemy" in res["persona_context"]
        assert "Prefer connection pool size 10 for team_doc" in res["persona_context"]
        assert "Prefer connection pool size 10 for team_doc" in res["context"]

    def test_get_personal_memory_service_caches_instance(self, monkeypatch):
        monkeypatch.setattr(pipeline_mod, "_cached_pms", None)

        fake_settings = SimpleNamespace(database=SimpleNamespace(url="postgresql+asyncpg://localhost/test"))
        fake_embedding = MagicMock()
        load_calls = 0
        embed_calls = 0

        def _fake_load_settings():
            nonlocal load_calls
            load_calls += 1
            return fake_settings

        def _fake_create_embedding(settings):
            nonlocal embed_calls
            embed_calls += 1
            return fake_embedding

        monkeypatch.setattr("team_memory.config.load_settings", _fake_load_settings)
        monkeypatch.setattr(
            "team_memory.embedding.factory.create_embedding_provider",
            _fake_create_embedding,
        )

        pms1 = _get_personal_memory_service()
        pms2 = _get_personal_memory_service()

        assert pms1 is pms2
        assert load_calls == 1
        assert embed_calls == 1
        assert pms1._db_url == "postgresql+asyncpg://localhost/test"
        assert pms1._embedding is fake_embedding


class TestOpContextConcurrencyAndL1Summary:
    """Test op_context concurrent execution via asyncio.gather and L1 summary field."""

    @pytest.mark.asyncio
    async def test_op_context_runs_profile_and_search_concurrently_with_l1_summary(self):
        async def _slow_profile(*args, **kwargs):
            await asyncio.sleep(0.15)
            return {
                "static": ["Use strict typing"],
                "dynamic": ["Use asyncio.gather for parallel I/O"],
            }

        long_solution = (
            "Use asyncio.gather to run profile retrieval and experience search in parallel "
            "so total latency equals the slower branch rather than the sum of both sequential calls. "
            "Extra details beyond 160 characters should be cleanly truncated."
        )

        async def _slow_search(*args, **kwargs):
            await asyncio.sleep(0.15)
            return SimpleNamespace(
                reranked=True,
                results=[
                    {
                        "group_id": str(uuid.uuid4()),
                        "parent": {
                            "id": str(uuid.uuid4()),
                            "title": "Parallel Context Fetch",
                            "tags": ["async", "performance"],
                            "solution": long_solution,
                        },
                        "confidence": "high",
                    },
                    {
                        "group_id": str(uuid.uuid4()),
                        "parent": {
                            "id": str(uuid.uuid4()),
                            "title": "Fallback Problem Summary",
                            "tags": ["summary"],
                            "problem": "Short   multi-space\nproblem description.",
                        },
                        "confidence": "medium",
                    },
                ],
            )

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(side_effect=_slow_search)
        mock_settings = SimpleNamespace(mcp=SimpleNamespace(profile_max_strings_per_side=20))

        with (
            patch(
                "team_memory.services.memory.ops.context._resolve_project",
                return_value="team_doc",
            ),
            patch(
                "team_memory.services.memory.personal_memory.PersonalMemoryService"
            ) as mock_pm_cls,
        ):
            mock_pm = MagicMock()
            mock_pm.build_profile_for_user = AsyncMock(side_effect=_slow_profile)
            mock_pm_cls.return_value = mock_pm

            t0 = time.monotonic()
            out = await op_context(
                "alice",
                search_orchestrator=mock_orchestrator,
                settings=mock_settings,
                embedding_provider=MagicMock(),
                db_url="sqlite://",
                file_paths=["src/team_memory/services/memory/ops/context.py"],
                task_description="Optimize memory_context latency",
                project="team_doc",
            )
            elapsed = time.monotonic() - t0

        assert elapsed < 0.24, f"Expected concurrent execution (<0.24s), took {elapsed:.3f}s"
        assert out["profile"] == {
            "static": ["Use strict typing"],
            "dynamic": ["Use asyncio.gather for parallel I/O"],
        }
        assert out["search_reranked"] is True
        assert len(out["relevant_experiences"]) == 2

        first_exp = out["relevant_experiences"][0]
        assert first_exp["title"] == "Parallel Context Fetch"
        assert first_exp["summary"].endswith("...")
        assert len(first_exp["summary"]) == 163

        second_exp = out["relevant_experiences"][1]
        assert second_exp["summary"] == "Short multi-space problem description."
