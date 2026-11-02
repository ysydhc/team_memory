"""Unit tests for Phase 1–4 upgrades:
- Phase 1: L0/L1 Progressive Disclosure + Hook B-1/B-2 bugfixes
- Phase 2: Antigravity (AGY) native hooks + transcript.jsonl parser & watcher
- Phase 3: Tool Execution Observation + Error-to-Pass Convergence
- Phase 4: Module-Level Living Wiki synthesis
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer
from team_memory.daemon.observe.agy_transcript_watcher import (
    discover_agy_transcripts,
    evaluate_agy_trajectory,
    import_agy_sessions,
    sync_agy_mcp_recalls,
)
from team_memory.daemon.quality.convergence_detector import ConvergenceDetector
from team_memory.daemon.refinement.pipeline import (
    _build_l0_manifest,
    _build_l1_context,
    _format_l1_item,
)
from team_memory.daemon.wiki.wiki_compiler import WikiCompiler
from team_memory.hooks.core.transcript import (
    clean_antigravity_user_input,
    parse_antigravity_transcript,
    read_last_user_from_antigravity_transcript,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Phase 1: L0 / L1 Progressive Disclosure & Hook Bugfixes (B-1, B-2)
# ---------------------------------------------------------------------------


class TestProgressiveDisclosure:
    def test_build_l0_manifest_compact(self) -> None:
        manifest = _build_l0_manifest(
            project="team_doc",
            experiences=[
                {"id": "11111111-2222-3333-4444-555555555555", "title": "Fix pgvector index", "tags": ["db", "pg"]},
                {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "title": "Hook timeout tuning", "tags": ["hooks"]},
            ],
        )
        assert "[TM L0 Manifest | project=team_doc]" in manifest
        assert "tm://exp/11111111" in manifest
        assert "Fix pgvector index" in manifest
        assert len(manifest) <= 600

    def test_format_l1_item_and_context(self) -> None:
        item = {
            "id": "abcdef12-3456-7890-abcd-ef1234567890",
            "title": "Alembic 迁移冲突解决",
            "tags": ["alembic", "postgres"],
            "description": "多分支同时生成 migration 导致 multiple heads 报错",
            "solution": "使用 alembic merge heads 生成合并迁移文件，然后执行 alembic upgrade head。",
        }
        line = _format_l1_item(item)
        assert "[tm://exp/abcdef12]" in line
        assert "Alembic 迁移冲突解决" in line
        assert "alembic merge heads" in line

        ctx = _build_l1_context([item])
        assert "[TM L1 Context" in ctx
        assert "memory_get" in ctx

    def test_claude_session_start_handles_string_and_dict_context(self) -> None:
        from team_memory.hooks.platforms.claude.session_start import main

        mock_resp = httpx.Response(
            200,
            text=json.dumps(
                {
                    "l0_manifest": "[TM L0 Manifest | project=test]\n  * [tm://exp/12345678] Test title",
                    "additional_context": {
                        "relevant_experiences": [{"id": "12345678", "title": "Test title"}],
                    },
                }
            ),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/session_start"),
        )
        captured = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(json.dumps({"session_id": "s1", "cwd": "/tmp/p"}))),
            patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=mock_resp),
            patch("sys.stdout", captured),
        ):
            main()

        out = json.loads(captured.getvalue().strip())
        assert "TM L0 Manifest" in out["additionalContext"]

    def test_claude_user_prompt_submit_renders_description_solution_fallback(self) -> None:
        from team_memory.hooks.platforms.claude.user_prompt_submit import main

        mock_resp = httpx.Response(
            200,
            text=json.dumps(
                {
                    "results": [
                        {
                            "id": "deadbeef-0000-1111-2222-333333333333",
                            "title": "Redis 连接池耗尽排查",
                            "description": "高并发下连接未释放",
                            "solution": "使用 async with 管理连接生命周期",
                            "tags": ["redis"],
                        }
                    ]
                }
            ),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )
        captured = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(json.dumps({"session_id": "s1", "prompt": "redis 连接池", "cwd": "/tmp"}))),
            patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=mock_resp),
            patch("sys.stdout", captured),
        ):
            main()

        out = json.loads(captured.getvalue().strip())
        ctx = out["additionalContext"]
        assert "Redis 连接池耗尽排查" in ctx
        assert "使用 async with 管理连接生命周期" in ctx


# ---------------------------------------------------------------------------
# Phase 2: Antigravity Transcript Parser, Hooks & Watcher
# ---------------------------------------------------------------------------


class TestAntigravitySupport:
    def test_clean_antigravity_user_input(self) -> None:
        raw = (
            "<MEMORY[user_def]>\nsome memory\n</MEMORY[user_def]>\n"
            "<USER_REQUEST>\n请帮我修复 hooks 的序列化 bug\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\nThe current local time is: 2026-10-22\n</ADDITIONAL_METADATA>"
        )
        assert clean_antigravity_user_input(raw) == "请帮我修复 hooks 的序列化 bug"

    def test_parse_antigravity_transcript_with_truncated_fields(self, tmp_path: Path) -> None:
        compact_path = tmp_path / "transcript.jsonl"
        full_path = tmp_path / "transcript_full.jsonl"

        compact_steps = [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "content": "<USER_REQUEST>\n如何优化 pgvector 查询？\n</USER_REQUEST>",
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "content": "truncated...",
                "truncated_fields": ["content"],
                "tool_calls": [{"name": "run_command", "args": {"CommandLine": "pytest -q"}}],
            },
        ]
        full_steps = [
            compact_steps[0],
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "content": "可以通过建立 HNSW 索引并调整 ef_search 参数来优化 pgvector 查询。",
                "tool_calls": [{"name": "run_command", "args": {"CommandLine": "pytest -q"}}],
            },
        ]

        compact_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in compact_steps) + "\n")
        full_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in full_steps) + "\n")

        parsed = parse_antigravity_transcript(str(compact_path))
        turns = parsed["turns"]
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "如何优化 pgvector 查询？"
        assert turns[1]["role"] == "assistant"
        assert "HNSW 索引" in turns[1]["content"]
        assert turns[1]["tool_calls"][0]["tool"] == "run_command"

        last_user = read_last_user_from_antigravity_transcript(str(compact_path))
        assert last_user == "如何优化 pgvector 查询？"

    def test_antigravity_hooks_contracts(self) -> None:
        from team_memory.hooks.platforms.antigravity.post_tool_use import main as post_tool_main
        from team_memory.hooks.platforms.antigravity.pre_invocation import main as pre_inv_main
        from team_memory.hooks.platforms.antigravity.session_start import main as sess_start_main
        from team_memory.hooks.platforms.antigravity.stop import main as stop_main

        # 1. SessionStart
        mock_start = httpx.Response(
            200,
            text=json.dumps({"l0_manifest": "[TM L0 Manifest | project=agy]"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/session_start"),
        )
        buf = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(json.dumps({"conversation_id": "agy-1", "cwd": "/tmp/agy"}))),
            patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=mock_start),
            patch("sys.stdout", buf),
        ):
            sess_start_main()
        out = json.loads(buf.getvalue().strip())
        assert "TM L0 Manifest" in out["injectSteps"][0]["systemMessage"]["systemMessage"]

        # 2. PreInvocation
        mock_pre = httpx.Response(
            200,
            text=json.dumps({"context": "[TM L1 Context]\n- [tm://exp/11223344] Test"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/before_prompt"),
        )
        buf = io.StringIO()
        with (
            patch(
                "sys.stdin",
                io.StringIO(json.dumps({"conversation_id": "agy-1", "prompt": "修复数据库连接池泄漏", "cwd": "/tmp"})),
            ),
            patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=mock_pre),
            patch("sys.stdout", buf),
        ):
            pre_inv_main()
        out = json.loads(buf.getvalue().strip())
        assert "TM L1 Context" in out["injectSteps"][0]["ephemeralMessage"]

        # 3. PostToolUse (run_command observation)
        buf = io.StringIO()
        with (
            patch(
                "sys.stdin",
                io.StringIO(
                    json.dumps(
                        {
                            "conversation_id": "agy-1",
                            "tool_name": "run_command",
                            "tool_input": {"CommandLine": "pytest tests/test_server.py"},
                            "tool_response": {"exit_code": 0, "output": "1 passed"},
                        }
                    )
                ),
            ),
            patch("team_memory.hooks.platforms.antigravity.post_tool_use.post_hook_fire_and_forget") as mock_ff,
            patch("sys.stdout", buf),
        ):
            post_tool_main()
        mock_ff.assert_called_once()
        assert mock_ff.call_args[0][0] == "tool_observation"

        # 4. Stop
        mock_stop = httpx.Response(
            200,
            text=json.dumps({"status": "ok"}),
            request=httpx.Request("POST", "http://127.0.0.1:3901/hooks/after_response"),
        )
        buf = io.StringIO()
        with (
            patch(
                "sys.stdin",
                io.StringIO(
                    json.dumps(
                        {
                            "conversation_id": "agy-1",
                            "last_assistant_message": "已修复问题并验证通过。",
                            "cwd": "/tmp/agy",
                        }
                    )
                ),
            ),
            patch("team_memory.hooks.core.daemon_http.httpx.post", return_value=mock_stop),
            patch("sys.stdout", buf),
        ):
            stop_main()
        out = json.loads(buf.getvalue().strip())
        assert out["decision"] == "allow"

    @pytest.mark.asyncio
    async def test_antigravity_transcript_watcher_and_import(self, tmp_path: Path) -> None:
        brain_dir = tmp_path / "brain"
        conv_dir = brain_dir / "conv-uuid-1234" / ".system_generated" / "logs"
        conv_dir.mkdir(parents=True)
        t_file = conv_dir / "transcript.jsonl"
        steps = [
            {"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>报错 ImportError</USER_REQUEST>"},
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "content": "原因在于循环依赖，已将类型导入放入 TYPE_CHECKING 块解决。" * 8,
            },
        ]
        t_file.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in steps) + "\n")

        discovered = discover_agy_transcripts(brain_dir=brain_dir)
        assert len(discovered) == 1
        assert discovered[0][0] == "conv-uuid-1234"

        parsed = parse_antigravity_transcript(str(t_file))
        valuable, reason = evaluate_agy_trajectory(parsed)
        assert valuable is True

        async with ConversationBuffer(db_path=str(tmp_path / "drafts.db")) as buf:
            res1 = await import_agy_sessions(buf=buf, brain_dir=brain_dir, project="team_doc")
            assert res1["discovered"] == 1
            assert res1["imported"] == 1

            # Second import skips already queued session
            res2 = await import_agy_sessions(buf=buf, brain_dir=brain_dir, project="team_doc")
            assert res2["imported"] == 0
            assert res2["skipped"] == 1

    @pytest.mark.asyncio
    async def test_antigravity_mcp_recall_extraction_and_sync(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock, MagicMock

        brain_dir = tmp_path / "brain"
        conv_dir = brain_dir / "conv-recall-999" / ".system_generated" / "logs"
        conv_dir.mkdir(parents=True)
        t_file = conv_dir / "transcript.jsonl"
        f_file = conv_dir / "transcript_full.jsonl"

        log_uuid = "12345678-1234-5678-9abc-123456789abc"
        exp_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        recall_json = json.dumps(
            {
                "message": "Found 1 result(s).",
                "results": [{"id": exp_uuid, "group_id": exp_uuid, "score": 0.91}],
                "search_log_id": log_uuid,
            }
        )
        compact_steps = [
            {"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>查询 ADMFunnel 漏斗规范</USER_REQUEST>"},
            {
                "step_index": 2,
                "type": "GENERIC",
                "content": "truncated...",
                "truncated_fields": ["content"],
            },
            {
                "step_index": 3,
                "type": "PLANNER_RESPONSE",
                "content": "已按 ADMFunnel 漏斗打点规范完成埋点配置。",
            },
        ]
        full_steps = [
            compact_steps[0],
            {"step_index": 2, "type": "GENERIC", "content": recall_json},
            compact_steps[2],
        ]
        t_file.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in compact_steps) + "\n")
        f_file.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in full_steps) + "\n")

        parsed = parse_antigravity_transcript(str(t_file))
        recalls = parsed["mcp_recalls"]
        assert len(recalls) == 1
        assert recalls[0]["search_log_id"] == log_uuid
        assert recalls[0]["result_ids"] == [{"id": exp_uuid, "score": 0.91}]
        assert "ADMFunnel" in recalls[0]["assistant_response"]

        mock_rb = MagicMock()
        mock_rb.has_search_log = AsyncMock(side_effect=[False, True])
        mock_rb.add = AsyncMock(return_value=42)

        sync1 = await sync_agy_mcp_recalls(response_buffer=mock_rb, brain_dir=brain_dir)
        assert sync1["recalls_found"] == 1
        assert sync1["buffered"] == 1
        mock_rb.add.assert_awaited_once()
        assert mock_rb.add.call_args.kwargs["search_log_id"] == log_uuid

        sync2 = await sync_agy_mcp_recalls(response_buffer=mock_rb, brain_dir=brain_dir)
        assert sync2["buffered"] == 0
        assert sync2["already_buffered"] == 1


# ---------------------------------------------------------------------------
# Phase 3: Tool Observations & Error-to-Pass Convergence
# ---------------------------------------------------------------------------


class TestToolObservationAndConvergence:
    @pytest.mark.asyncio
    async def test_conversation_buffer_tool_observations(self, tmp_path: Path) -> None:
        async with ConversationBuffer(db_path=str(tmp_path / "drafts.db")) as buf:
            obs_id = await buf.add_tool_observation(
                "sess-tool-1",
                "run_command",
                command_summary="pytest tests/test_server.py",
                exit_code=1,
                error_excerpt="AssertionError: expected 6 tools",
                target_files=["src/team_memory/server.py"],
                succeeded=False,
            )
            assert obs_id > 0

            observations = await buf.get_tool_observations("sess-tool-1")
            assert len(observations) == 1
            assert observations[0]["tool_name"] == "run_command"
            assert observations[0]["exit_code"] == 1
            assert observations[0]["succeeded"] is False
            assert observations[0]["target_files"] == ["src/team_memory/server.py"]

    def test_error_to_pass_convergence_detection(self) -> None:
        detector = ConvergenceDetector()
        commands = [
            {"command": "pytest tests/test_auth.py", "exit_code": 1, "succeeded": False},
            {"command": "pytest tests/test_auth.py", "exit_code": 0, "succeeded": True},
        ]
        assert detector.check_error_to_pass_resolution(commands) is True

        conv = detector.detect_convergence(
            "Tests are now passing.",
            recent_tools=commands,
        )
        assert conv is True


# ---------------------------------------------------------------------------
# Phase 4: Module-Level Living Wiki Synthesis
# ---------------------------------------------------------------------------


class TestLivingWikiModuleSynthesis:
    @pytest.mark.asyncio
    async def test_compile_module_pages_clusters_by_project_and_primary_tag(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        experiences = [
            {
                "id": "exp-1",
                "title": "PostgreSQL 连接池超时配置",
                "description": "高并发下 QueuePool 耗尽报错 TimeoutError",
                "solution": "调大 pool_size=20, max_overflow=10 并开启 pool_pre_ping",
                "tags": ["database", "postgresql"],
                "project": "team_memory",
            },
            {
                "id": "exp-2",
                "title": "pgvector HNSW 索引构建参数优化",
                "description": "向量检索在百万级数据下延迟超过 200ms",
                "solution": "使用 HNSW 索引设置 m=16, ef_construction=64",
                "tags": ["database", "pgvector"],
                "project": "team_memory",
            },
            {
                "id": "exp-3",
                "title": "单条独立前端样式修复",
                "description": "移动端按钮溢出",
                "solution": "使用 flex-wrap",
                "tags": ["frontend"],
                "project": "team_memory",
            },
        ]

        async with WikiCompiler(wiki_root=str(wiki_root)) as compiler:
            res = await compiler.compile_incremental(experiences)
            assert res.created == 3
            stats = await compiler.get_stats()
            assert stats["total"] == 3

            module_file = wiki_root / "modules" / "team_memory" / "database.md"
            assert module_file.exists()
            content = module_file.read_text(encoding="utf-8")
            assert "type: module_wiki" in content
            assert "experience_count: 2" in content
            assert "PostgreSQL 连接池超时配置" in content
            assert "pgvector HNSW 索引构建参数优化" in content
            assert "## 核心方案与架构约定" in content
            assert "## 常见问题与避坑指南" in content

            # Single-experience tag ("frontend") should not generate a module page
            assert not (wiki_root / "modules" / "team_memory" / "frontend.md").exists()
