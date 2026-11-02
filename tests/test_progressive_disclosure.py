"""Unit tests for Phase 1B MCP memory_recall progressive disclosure & token telemetry."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import pytest

    _async_mark = pytest.mark.asyncio
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

    def _async_mark(fn: Any) -> Any:
        return fn

from team_memory.services.memory.ops.recall import op_recall
from team_memory.services.search.progressive_disclosure import (
    apply_progressive_disclosure,
    estimate_payload_tokens,
)


def _make_realistic_experiences(count: int = 5) -> list[dict[str, Any]]:
    """Build realistic grouped experience payloads (~600c desc, ~800c sol, 3 children)."""
    items: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        desc = (
            f"[Exp-{i}] 生产环境高并发检索故障分析与根因定位 (Production incident analysis): "
            + (
                "当多个异步 Worker 并发访问 pgvector HNSW 索引且连接池满载时，会出现检索尾延迟飙升与上下文 Token 溢出问题；"
                "When concurrent workers access pgvector HNSW indexes under high connection pool load, "
                * 5
            )
        )[:620]
        sol = (
            f"[Exp-{i}] 分步修复方案与参数调优指南 (Step-by-step remediation guide): "
            + (
                "配置 AsyncSession pool_size=20, max_overflow=10，启用 RRF 混合检索与 L0/L1/L2 渐进式披露机制，"
                "对 Top-2 保留完整 L2 详情，其余降级为 L1 摘要卡片以严格控制 MCP 返回 Token 预算。"
                "Configure AsyncSession pool_size=20, max_overflow=10, and apply progressive disclosure. "
                * 5
            )
        )[:820]
        children = [
            {
                "id": f"exp-{i}-child-{c}",
                "title": f"Sub-case {i}.{c}: tuning step {c}",
                "description": (
                    f"子问题 {i}.{c} 详细排查上下文: "
                    + (
                        "排查批量向量回填与混合检索期间的数据库连接争用、HNSW ef_search 参数调优及慢查询日志分析。"
                        "Investigating query latency spikes during batch embedding backfill operations. "
                        * 5
                    )
                )[:600],
                "solution": (
                    f"子方案 {i}.{c} 具体修复步骤: "
                    + (
                        "设置 statement_timeout 与批量大小 64，结合指数退避重试、游标分页与 SearchLog 阶段耗时监控。"
                        "Use statement_timeout and batch size 64 with exponential backoff and cursor pagination. "
                        * 6
                    )
                )[:800],
                "tags": ["postgres", "pgvector", f"step-{c}"],
            }
            for c in range(1, 4)
        ]
        items.append(
            {
                "id": f"exp-{i}",
                "group_id": f"grp-{i}",
                "title": f"Experience {i}: Optimizing pgvector hybrid recall latency",
                "description": desc,
                "solution": sol,
                "tags": ["postgres", "pgvector", "mcp"],
                "score": round(0.95 - i * 0.05, 2),
                "similarity": round(0.92 - i * 0.05, 2),
                "confidence": "high" if i <= 2 else "medium",
                "type": "experience",
                "status": "published",
                "children": children,
            }
        )
    return items


class TestProgressiveDisclosure:
    """Tests for apply_progressive_disclosure and estimate_payload_tokens."""

    def test_progressive_mode_reduces_tokens_over_50_pct(self) -> None:
        raw_results = _make_realistic_experiences(5)
        disclosed, telemetry = apply_progressive_disclosure(
            raw_results,
            detail_level="progressive",
            full_top_k=2,
        )

        assert len(disclosed) == 5
        assert telemetry["detail_level"] == "progressive"
        assert telemetry["full_tokens"] > telemetry["returned_tokens"] > 0
        assert telemetry["saved_tokens_pct"] >= 50.0

        # Top-1 and Top-2 retain full L2 payload
        for idx in (0, 1):
            assert disclosed[idx]["disclosure_level"] == "full"
            assert disclosed[idx]["solution"] == raw_results[idx]["solution"]
            assert disclosed[idx]["description"] == raw_results[idx]["description"]
            assert len(disclosed[idx]["children"]) == 3

        # Ranks 3..5 are compacted into L1 summary cards
        for idx in (2, 3, 4):
            card = disclosed[idx]
            assert card["disclosure_level"] == "summary"
            assert "children" not in card
            assert card["children_count"] == 3
            assert len(card["children_titles"]) == 3
            assert card["summary"].endswith("...")
            assert card["solution"].endswith("... [L1 summary]")
            assert len(card["solution"]) < len(raw_results[idx]["solution"])
            assert "memory_recall(" in card["expand_hint"]

    def test_full_mode_preserves_all_items(self) -> None:
        raw_results = _make_realistic_experiences(5)
        disclosed, telemetry = apply_progressive_disclosure(
            raw_results,
            detail_level="full",
            full_top_k=2,
        )

        assert len(disclosed) == 5
        assert telemetry["detail_level"] == "full"
        assert telemetry["saved_tokens_pct"] <= 0.0
        for idx, item in enumerate(disclosed):
            assert item["disclosure_level"] == "full"
            assert item["solution"] == raw_results[idx]["solution"]
            assert len(item["children"]) == 3

    def test_summary_mode_compacts_all_items(self) -> None:
        raw_results = _make_realistic_experiences(5)
        disclosed, telemetry = apply_progressive_disclosure(
            raw_results,
            detail_level="summary",
            full_top_k=2,
        )

        assert len(disclosed) == 5
        assert telemetry["detail_level"] == "summary"
        assert telemetry["saved_tokens_pct"] >= 75.0
        for item in disclosed:
            assert item["disclosure_level"] == "summary"
            assert "children" not in item
            assert item["children_count"] == 3
            assert "expand_hint" in item

    def test_empty_results(self) -> None:
        disclosed, telemetry = apply_progressive_disclosure([])
        assert disclosed == []
        assert telemetry == {
            "detail_level": "progressive",
            "full_tokens": 0,
            "returned_tokens": 0,
            "saved_tokens_pct": 0.0,
        }
        assert estimate_payload_tokens([]) == 0


@dataclass
class _DummyOrchestratedSearch:
    results: list[dict[str, Any]]
    reranked: bool = True
    intent_type: str = "troubleshoot"
    stage_metrics: dict[str, float] = field(
        default_factory=lambda: {"vector_ms": 12.4, "fts_ms": 4.1, "rrf_ms": 1.2, "total_ms": 19.8}
    )
    stage_counts: dict[str, int] = field(
        default_factory=lambda: {"vector_candidates": 18, "fts_candidates": 9, "rrf_fused": 5}
    )


class TestOpRecallProgressiveTelemetry:
    """Tests for op_recall integration with progressive disclosure and SearchLog telemetry."""

    @_async_mark
    async def test_op_recall_progressive_attaches_token_and_stage_telemetry(self) -> None:
        raw_results = _make_realistic_experiences(5)
        dummy_search = _DummyOrchestratedSearch(results=raw_results)

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(return_value=dummy_search)
        mock_settings = MagicMock()
        mock_settings.mcp.profile_max_strings_per_side = 10

        with (
            patch(
                "team_memory.services.memory.ops.recall._resolve_project",
                side_effect=lambda p: p or "team_doc",
            ),
            patch(
                "team_memory.services.memory.memory_operations.write_search_log",
                new_callable=AsyncMock,
                return_value="log-123",
            ) as mock_write_log,
        ):
            payload = await op_recall(
                user="tester",
                search_orchestrator=mock_orchestrator,
                db_url="postgresql+asyncpg://localhost/test",
                settings=mock_settings,
                embedding_provider=MagicMock(),
                query="pgvector hybrid recall latency",
                project="team_doc",
                include_user_profile=False,
                include_archives=False,
                detail_level="progressive",
            )

        assert payload["detail_level"] == "progressive"
        assert payload["returned_tokens"] > 0
        assert payload["saved_tokens_pct"] >= 50.0
        assert payload["search_log_id"] == "log-123"
        assert "_stage_metrics" not in payload
        assert "_stage_counts" not in payload

        results = payload["results"]
        assert len(results) == 5
        assert results[0]["disclosure_level"] == "full"
        assert results[1]["disclosure_level"] == "full"
        assert results[2]["disclosure_level"] == "summary"

        mock_write_log.assert_awaited_once()
        call_kwargs = mock_write_log.call_args.kwargs
        assert call_kwargs["returned_tokens"] == payload["returned_tokens"]
        assert call_kwargs["stage_metrics"] == dummy_search.stage_metrics
        assert call_kwargs["stage_counts"] == dummy_search.stage_counts

    @_async_mark
    async def test_op_recall_suggest_mode_preserves_legacy_write_search_log_kwargs(self) -> None:
        raw_results = _make_realistic_experiences(2)
        dummy_search = _DummyOrchestratedSearch(
            results=raw_results,
            stage_metrics=None,  # type: ignore[arg-type]
            stage_counts=None,  # type: ignore[arg-type]
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(return_value=dummy_search)
        mock_settings = MagicMock()
        mock_settings.mcp.profile_max_strings_per_side = 10

        with (
            patch(
                "team_memory.services.memory.ops.recall._resolve_project",
                side_effect=lambda p: p or "team_doc",
            ),
            patch(
                "team_memory.services.memory.memory_operations.write_search_log",
                new_callable=AsyncMock,
                return_value="log-456",
            ) as mock_write_log,
        ):
            payload = await op_recall(
                user="tester",
                search_orchestrator=mock_orchestrator,
                db_url="postgresql+asyncpg://localhost/test",
                settings=mock_settings,
                embedding_provider=MagicMock(),
                file_path="src/api/auth.py",
                language="python",
                include_user_profile=False,
                include_archives=False,
            )

        assert payload["search_log_id"] == "log-456"
        mock_write_log.assert_awaited_once()
        call_kwargs = mock_write_log.call_args.kwargs
        assert "stage_metrics" not in call_kwargs
        assert "stage_counts" not in call_kwargs
        assert "returned_tokens" not in call_kwargs


    @_async_mark
    async def test_op_recall_cross_project_inference_and_l1_project_preserved(self) -> None:
        raw_results = _make_realistic_experiences(4)
        for item in raw_results:
            item["project"] = "adx_sdk_ios"
        dummy_search = _DummyOrchestratedSearch(results=raw_results)

        mock_orchestrator = MagicMock()
        mock_orchestrator.search = AsyncMock(return_value=dummy_search)
        mock_settings = MagicMock()
        mock_settings.mcp.profile_max_strings_per_side = 10

        with (
            patch(
                "team_memory.services.memory.ops.recall._resolve_project",
                return_value="team_memory",
            ),
            patch(
                "team_memory.services.memory.memory_operations.write_search_log",
                new_callable=AsyncMock,
                return_value="log-ios-789",
            ) as mock_write_log,
        ):
            payload = await op_recall(
                user="tester",
                search_orchestrator=mock_orchestrator,
                db_url="postgresql+asyncpg://localhost/test",
                settings=mock_settings,
                embedding_provider=MagicMock(),
                query="ADMFunnel reportEvent ADMSlotTemplate render",
                project=None,
                include_user_profile=False,
                include_archives=False,
                detail_level="progressive",
            )

        # L1 summary cards (rank 3 & 4) preserve 'project'
        assert payload["results"][2]["disclosure_level"] == "summary"
        assert payload["results"][2]["project"] == "adx_sdk_ios"

        # Both search_orchestrator.search and write_search_log receive inferred 'adx_sdk_ios'
        assert mock_orchestrator.search.call_args.kwargs["project"] == "adx_sdk_ios"
        assert mock_write_log.call_args.kwargs["project"] == "adx_sdk_ios"


if __name__ == "__main__":
    import types

    try:
        import team_memory.services.memory.memory_operations  # noqa: F401
    except ModuleNotFoundError:
        import team_memory.services.memory as _mem_pkg

        _stub = types.ModuleType("team_memory.services.memory.memory_operations")
        _stub.write_search_log = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        sys.modules["team_memory.services.memory.memory_operations"] = _stub
        _mem_pkg.memory_operations = _stub  # type: ignore[attr-defined]

    suite = TestProgressiveDisclosure()
    suite.test_progressive_mode_reduces_tokens_over_50_pct()
    suite.test_full_mode_preserves_all_items()
    suite.test_summary_mode_compacts_all_items()
    suite.test_empty_results()
    asyncio.run(
        TestOpRecallProgressiveTelemetry().test_op_recall_progressive_attaches_token_and_stage_telemetry()
    )
    asyncio.run(
        TestOpRecallProgressiveTelemetry().test_op_recall_suggest_mode_preserves_legacy_write_search_log_kwargs()
    )

