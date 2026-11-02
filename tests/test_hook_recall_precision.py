"""TDD unit tests for Hook Recall Precision & Adoption Loop (P0-1, P0-2, P1-1, P1-2).

Covers all 8 boundary scenarios (B1–B8) defined in the Spec:
- B1: Multi-root explicit project/alias match in prompt (`tm`, `tm-cli`, `team_doc`, `phone_pilot`)
- B2: Multi-root session sticky project inheritance when follow-up prompt has no project name
- B3: Multi-root explicit project switch updates `ConversationBuffer` session sticky project
- B4: Short follow-up context enrichment borrows core entities from previous user message
- B5: Same-topic entity overlap (>=2 entities & >=80% overlap) short-circuits redundant recall
- B6: Persona working set deduplication within a session + switching to a new persona
- B7: Persona Top-5 relevance cap when user has > 5 personal memories
- B8: Rule-based adoption evaluation (URI / ID prefix / entity overlap) & `record_rule_adoption`
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer
from team_memory.daemon.refinement.pipeline import (
    _resolve_project_with_meta,
    process_before_prompt,
    process_session_start,
)
from team_memory.daemon.runtime.config import DaemonConfig, ProjectMapping, RetrievalSettings
from team_memory.daemon.runtime.routes.hooks import hook_after_response
from team_memory.project_resolver import resolve_multi_root_project
from team_memory.services.memory.personal_memory import PersonalMemoryService
from team_memory.services.search.progressive_disclosure import (
    get_session_working_set,
)
from team_memory.services.search.search_log_writer import (
    SearchLogWriter,
    evaluate_rule_based_adoption,
)


def _make_multi_root_config() -> DaemonConfig:
    return DaemonConfig(
        projects=[
            ProjectMapping(name="ad_learning", path_patterns=["ad_learning"]),
            ProjectMapping(name="team_doc", path_patterns=["team_doc"]),
            ProjectMapping(name="phone_pilot", path_patterns=["phone_pilot"]),
        ],
        retrieval=RetrievalSettings(session_start_top_k=3),
    )


MULTI_ROOTS = [
    "/Users/yeshouyou/Work/ad_learning",
    "/Users/yeshouyou/Work/agent/team_doc",
    "/Users/yeshouyou/Work/agent/phone_pilot",
]


# ---------------------------------------------------------------------------
# B1, B2, B3: Multi-Root Smart Project Resolution & Session Sticky
# ---------------------------------------------------------------------------


class TestMultiRootProjectResolution:
    """B1–B3: Multi-root project resolution and session sticky persistence."""

    def test_b1_explicit_project_and_alias_match_in_multi_root(self) -> None:
        """B1: Prompt mentioning `tm`, `tm-cli`, `team_doc`, or file path resolves `team_doc` over roots[0]."""
        config = _make_multi_root_config()

        proj1, explicit1 = _resolve_project_with_meta(
            MULTI_ROOTS,
            config,
            prompt="查看 tm-cli status 的召回链路是否正常",
        )
        assert proj1 == "team_doc"
        assert explicit1 is True

        proj2, explicit2 = _resolve_project_with_meta(
            MULTI_ROOTS,
            config,
            prompt="在 tm 中分析当前的检索命中率",
        )
        assert proj2 == "team_doc"
        assert explicit2 is True

        proj3, explicit3 = _resolve_project_with_meta(
            MULTI_ROOTS,
            config,
            target_files=["/Users/yeshouyou/Work/agent/phone_pilot/src/driver.py"],
        )
        assert proj3 == "phone_pilot"
        assert explicit3 is True

        # Direct call to resolve_multi_root_project
        proj4, explicit4 = resolve_multi_root_project(
            MULTI_ROOTS,
            prompt="优化 team_doc 的 hook 召回策略",
        )
        assert proj4 in ("team_doc", "team_memory")
        assert explicit4 is True

    @pytest.mark.asyncio
    async def test_b2_multi_root_session_sticky_inheritance(self, tmp_path) -> None:
        """B2: Follow-up prompt without project name inherits sticky project from ConversationBuffer."""
        config = _make_multi_root_config()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value=[{"id": "exp-11112222", "title": "TM Recall Doc"}])

        async with ConversationBuffer(db_path=str(tmp_path / "buf.db")) as buf:
            # Session starts with roots[0] = ad_learning
            await process_session_start(
                {"conversation_id": "sess-b2", "workspace_roots": MULTI_ROOTS},
                config,
                sink,
                buf=buf,
            )
            sess_initial = await buf.get_session("sess-b2")
            assert sess_initial is not None
            assert sess_initial["project"] == "ad_learning"

            # Turn 1: User explicitly mentions tm-cli -> resolves to team_doc and updates sticky project
            res1 = await process_before_prompt(
                {
                    "conversation_id": "sess-b2",
                    "workspace_roots": MULTI_ROOTS,
                    "prompt": "帮我检查 tm-cli status 的召回链路统计",
                },
                config,
                sink,
                buf=buf,
            )
            assert res1["project"] == "team_doc"
            sess_after_t1 = await buf.get_session("sess-b2")
            assert sess_after_t1 is not None
            assert sess_after_t1["project"] == "team_doc"

            # Turn 2: Ambiguous follow-up without project name -> inherits sticky `team_doc`, NOT `ad_learning`
            res2 = await process_before_prompt(
                {
                    "conversation_id": "sess-b2",
                    "workspace_roots": MULTI_ROOTS,
                    "prompt": "为什么采纳率统计显示是零？",
                },
                config,
                sink,
                buf=buf,
            )
            assert res2["project"] == "team_doc"
            assert sink.recall.call_args.kwargs["project"] == "team_doc"

    @pytest.mark.asyncio
    async def test_b3_multi_root_explicit_project_switch_updates_sticky(self, tmp_path) -> None:
        """B3: Explicitly mentioning another project (`ad_learning`) updates sticky project."""
        config = _make_multi_root_config()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value=[])

        async with ConversationBuffer(db_path=str(tmp_path / "buf_b3.db")) as buf:
            await buf.ensure_session("sess-b3", "team_doc")
            res = await process_before_prompt(
                {
                    "conversation_id": "sess-b3",
                    "workspace_roots": MULTI_ROOTS,
                    "prompt": "切换到 ad_learning 查看 frida 抓包脚本配置",
                },
                config,
                sink,
                buf=buf,
            )
            assert res["project"] == "ad_learning"
            sess = await buf.get_session("sess-b3")
            assert sess is not None
            assert sess["project"] == "ad_learning"


# ---------------------------------------------------------------------------
# B4, B5: Short Follow-up Context Enrichment & Same-Topic Short-Circuit Gate
# ---------------------------------------------------------------------------


class TestShortFollowUpAndTopicShortCircuit:
    """B4–B5: Short follow-up entity borrowing and same-topic short-circuiting."""

    @pytest.mark.asyncio
    async def test_b4_short_followup_borrows_entities_from_previous_user_turn(self, tmp_path) -> None:
        """B4: Short follow-up (<25 chars / 0 entities) borrows entities from prior user turn."""
        config = _make_multi_root_config()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value=[{"id": "exp-aaaa1111", "title": "SearchLogWriter guide"}])

        ws = get_session_working_set()
        ws.clear_session("sess-b4")

        try:
            async with ConversationBuffer(db_path=str(tmp_path / "buf_b4.db")) as buf:
                # Turn 1: Rich technical query with entities `SearchLogWriter` and `hook_after_response`
                await process_before_prompt(
                    {
                        "conversation_id": "sess-b4",
                        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                        "prompt": "排查 SearchLogWriter 与 hook_after_response 的采纳率回写逻辑",
                    },
                    config,
                    sink,
                    buf=buf,
                )

                # Turn 2: Short follow-up without explicit technical entities
                res2 = await process_before_prompt(
                    {
                        "conversation_id": "sess-b4",
                        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                        "prompt": "为什么都是 0？",
                    },
                    config,
                    sink,
                    buf=buf,
                )

                enriched_query = res2["search_query"]
                assert "为什么都是 0？" in enriched_query
                assert "SearchLogWriter" in enriched_query or "hook_after_response" in enriched_query
                assert sink.recall.call_args.kwargs["query"] == enriched_query
        finally:
            ws.clear_session("sess-b4")

    @pytest.mark.asyncio
    async def test_b5_same_topic_entity_overlap_short_circuits_recall(self, tmp_path) -> None:
        """B5: Consecutive turn with >=2 entities and >=80% overlap short-circuits recall."""
        config = _make_multi_root_config()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value=[{"id": "exp-bbbb2222", "title": "Pipeline doc"}])

        ws = get_session_working_set()
        ws.clear_session("sess-b5")

        try:
            async with ConversationBuffer(db_path=str(tmp_path / "buf_b5.db")) as buf:
                # Turn 1: Searches entities `SearchLogWriter` and `ConversationBuffer`
                res1 = await process_before_prompt(
                    {
                        "conversation_id": "sess-b5",
                        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                        "prompt": "检查 SearchLogWriter 和 ConversationBuffer 的异步调用",
                    },
                    config,
                    sink,
                    buf=buf,
                )
                assert res1["short_circuited"] is False
                assert sink.recall.await_count == 1

                # Turn 2: Same two entities (>=80% overlap) -> short-circuits!
                res2 = await process_before_prompt(
                    {
                        "conversation_id": "sess-b5",
                        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                        "prompt": "继续分析 SearchLogWriter 和 ConversationBuffer 的返回值",
                    },
                    config,
                    sink,
                    buf=buf,
                )
                assert res2["short_circuited"] is True
                assert res2["results"] == []
                # sink.recall was NOT called a second time
                assert sink.recall.await_count == 1
        finally:
            ws.clear_session("sess-b5")


# ---------------------------------------------------------------------------
# B6, B7: Persona Working Set Deduplication & Top-5 Relevance Cap
# ---------------------------------------------------------------------------


class TestPersonaWorkingSetAndCap:
    """B6–B7: Session persona deduplication and Top-5 relevance cap."""

    @pytest.mark.asyncio
    async def test_b6_persona_deduplication_and_switch_in_session(self) -> None:
        """B6: Same persona is injected only once per session; switching persona injects the new one."""
        config = _make_multi_root_config()
        sink = AsyncMock()
        sink.recall = AsyncMock(return_value=[])
        ws = get_session_working_set()
        ws.clear_session("sess-b6")

        mock_pms = AsyncMock()
        mock_pms.build_profile_for_user = AsyncMock(
            return_value={
                "static": ["Prefer concise code", "Use async/await for DB"],
                "dynamic": ["Working on TeamMemory"],
            }
        )

        try:
            with patch(
                "team_memory.daemon.refinement.pipeline._get_personal_memory_service",
                return_value=mock_pms,
            ):
                # Turn 1: coding persona -> injected
                res1 = await process_before_prompt(
                    {
                        "conversation_id": "sess-b6",
                        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                        "prompt": "帮我重构这个函数和类的实现",
                    },
                    config,
                    sink,
                )
                assert res1["persona"] == "coding"
                assert "[PERSONA: coding]" in res1["persona_context"]

                # Turn 2: coding persona again in same session -> skipped (0 duplicate tokens)
                res2 = await process_before_prompt(
                    {
                        "conversation_id": "sess-b6",
                        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                        "prompt": "再写一个函数处理 API 参数",
                    },
                    config,
                    sink,
                )
                assert res2["persona"] == "coding"
                assert res2["persona_context"] == ""

                # Turn 3: switches to debugging persona -> injected!
                res3 = await process_before_prompt(
                    {
                        "conversation_id": "sess-b6",
                        "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"],
                        "prompt": "这里遇到了一个报错 exception 需要排查修复",
                    },
                    config,
                    sink,
                )
                assert res3["persona"] == "debugging"
                assert "[PERSONA: debugging]" in res3["persona_context"]
        finally:
            ws.clear_session("sess-b6")

    @pytest.mark.asyncio
    async def test_b7_persona_top_5_relevance_cap(self) -> None:
        """B7: When user has 20 personal memories, build_profile_for_user caps at Top-5 most relevant."""
        svc = PersonalMemoryService(embedding_provider=MagicMock(), db_url="postgresql+asyncpg://localhost/test")
        twenty_memories = [
            {
                "content": f"Generic preference #{i}: general coding style note",
                "scope": "generic",
                "profile_kind": "static",
                "persona": "debugging",
            }
            for i in range(15)
        ] + [
            {
                "content": "When debugging SearchLogWriter and hook_after_response, verify PostgreSQL used_count",
                "scope": "context",
                "profile_kind": "dynamic",
                "persona": "debugging",
                "similarity": 0.92,
            },
            {
                "content": "In team_doc pipeline.py, preserve backward-compatible function signatures",
                "scope": "generic",
                "profile_kind": "static",
                "persona": "debugging",
            },
            {
                "content": "Use async/await for all SQLAlchemy session operations in SearchLogWriter",
                "scope": "generic",
                "profile_kind": "static",
                "persona": "debugging",
            },
            {
                "content": "Extra dynamic item A",
                "scope": "context",
                "profile_kind": "dynamic",
                "persona": "debugging",
                "similarity": 0.45,
            },
            {
                "content": "Extra dynamic item B",
                "scope": "context",
                "profile_kind": "dynamic",
                "persona": "debugging",
                "similarity": 0.40,
            },
        ]

        with patch.object(svc, "pull", new_callable=AsyncMock, return_value=twenty_memories):
            profile = await svc.build_profile_for_user(
                user_id="tester",
                current_context="debugging SearchLogWriter in pipeline.py",
                persona="debugging",
                max_items=5,
                query_hint="SearchLogWriter hook_after_response pipeline.py",
            )

        total_items = profile["static"] + profile["dynamic"]
        assert len(total_items) <= 5
        assert any("SearchLogWriter" in item for item in total_items)


# ---------------------------------------------------------------------------
# B8: Lightweight Rule-Based Adoption & Utility Rate Feedback Loop
# ---------------------------------------------------------------------------


class TestRuleBasedAdoptionLoop:
    """B8: Rule-based adoption evaluation and SearchLogWriter / hook_after_response integration."""

    def test_b8_evaluate_rule_based_adoption_signals(self) -> None:
        """B8: Matches explicit tm://exp URI, 8-char ID prefix, or >=2 technical entities."""
        results = [
            {
                "id": "818d12d6-aaaa-bbbb-cccc-111122223333",
                "title": "Fix SearchLogWriter and ConversationBuffer sticky session bug",
            },
            {
                "id": "929e34f8-dddd-eeee-ffff-444455556666",
                "title": "Unrelated Android Gradle build tip",
            },
        ]

        # Signal 1: Explicit URI reference
        adopted1, ids1 = evaluate_rule_based_adoption(
            results,
            "参考 [tm://exp/818d12d6] 的方案完成了修复。",
        )
        assert adopted1 is True
        assert ids1 == ["818d12d6-aaaa-bbbb-cccc-111122223333"]

        # Signal 2: >=2 technical entities from title appear in response + tool_observations
        adopted2, ids2 = evaluate_rule_based_adoption(
            results,
            "我们已经更新了 SearchLogWriter 的采纳逻辑。",
            tool_observations=[
                {
                    "tool_name": "edit_file",
                    "target_files": ["src/team_memory/daemon/buffers/conversation_buffer.py"],
                    "command_summary": "Update ConversationBuffer",
                }
            ],
        )
        assert adopted2 is True
        assert "818d12d6-aaaa-bbbb-cccc-111122223333" in ids2

        # Negative case: unrelated response
        adopted3, ids3 = evaluate_rule_based_adoption(
            results,
            "今天天气不错，没有任何相关代码改动。",
        )
        assert adopted3 is False
        assert ids3 == []

    @pytest.mark.asyncio
    async def test_b8_hook_after_response_records_rule_adoption(self) -> None:
        """B8: hook_after_response triggers record_rule_adoption and increments used_count."""
        config = _make_multi_root_config()
        buf = AsyncMock()
        buf.get_session = AsyncMock(return_value={"id": "sess-b8", "status": "active", "project": "team_doc"})
        buf.add_message = AsyncMock(return_value=1)
        buf.get_tool_observations = AsyncMock(return_value=[])
        buf.count_messages = AsyncMock(return_value=1)

        detector = MagicMock()
        detector.detect_convergence = MagicMock(return_value=False)
        refiner = AsyncMock()
        sink = AsyncMock()

        search_log = AsyncMock(spec=SearchLogWriter)
        search_log.record_rule_adoption = AsyncMock(return_value=True)

        exp_full_id = "818d12d6-1234-5678-9abc-def012345678"
        recent_searches = {
            "sess-b8": {
                "log_id": "log-uuid-9999",
                "query": "SearchLogWriter sticky session",
                "result_ids": [
                    {
                        "id": exp_full_id,
                        "score": 0.91,
                        "source": "mem",
                        "title": "Fix SearchLogWriter and ConversationBuffer sticky session",
                    }
                ],
                "project": "team_doc",
            }
        }

        app_state = SimpleNamespace(
            config=config,
            sink=sink,
            buf=buf,
            detector=detector,
            refiner=refiner,
            metrics=None,
            search_log=search_log,
            recent_searches=recent_searches,
            response_buffer=None,
            payload_capture=None,
        )
        fake_request = MagicMock()
        fake_request.app = SimpleNamespace(state=app_state)
        fake_request.json = AsyncMock(
            return_value={
                "conversation_id": "sess-b8",
                "workspace_roots": MULTI_ROOTS,
                "response_text": "已按 [tm://exp/818d12d6] 修改 SearchLogWriter 与 ConversationBuffer。",
            }
        )

        result = await hook_after_response(fake_request)
        assert result["project"] == "team_doc"
        assert result.get("rule_adopted") is True
        assert result.get("adopted_experience_ids") == [exp_full_id]
        search_log.record_rule_adoption.assert_awaited_once_with(
            "log-uuid-9999",
            [exp_full_id],
        )
