"""Unit tests for OpenAI Codex CLI transcript parser, hooks, watcher, and CLI setup."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team_memory.cli.commands.setup_check_uninstall import _check_codex, _uninstall_codex
from team_memory.cli.commands.setup_platforms import _setup_codex
from team_memory.daemon.buffers.conversation_buffer import ConversationBuffer
from team_memory.daemon.observe.codex_transcript_watcher import (
    discover_codex_transcripts,
    evaluate_codex_trajectory,
    import_codex_sessions,
)
from team_memory.hooks.core.transcript import (
    parse_codex_transcript,
    read_last_assistant_from_transcript,
    resolve_codex_transcript_path,
)
from team_memory.hooks.platforms.codex import (
    post_tool_use as codex_post_tool,
)
from team_memory.hooks.platforms.codex import (
    session_start as codex_session_start,
)
from team_memory.hooks.platforms.codex import (
    stop as codex_stop,
)
from team_memory.hooks.platforms.codex import (
    user_prompt_submit as codex_prompt,
)


def _write_sample_rollout(path: Path, session_id: str = "01a13d9b-94b1-7f21-bd65-dbdb47138f79") -> Path:
    events = [
        {
            "timestamp": "2026-10-15T03:32:11.397Z",
            "ordinal": 0,
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "cwd": "/Users/dev/work/demo-service",
                "runtime_workspace_roots": ["/Users/dev/work/demo-service"],
            },
        },
        {
            "timestamp": "2026-10-15T03:32:12.000Z",
            "ordinal": 1,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "帮我排查 pytest 报错 ConnectionRefusedError 并修复"}],
            },
        },
        {
            "timestamp": "2026-10-15T03:32:13.000Z",
            "ordinal": 2,
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_1",
                "arguments": json.dumps({"cmd": "pytest tests/test_api.py", "workdir": "/Users/dev/work/demo-service"}),
            },
        },
        {
            "timestamp": "2026-10-15T03:32:14.000Z",
            "ordinal": 3,
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "1 passed in 0.42s\nProcess exited with code 0",
            },
        },
        {
            "timestamp": "2026-10-15T03:32:15.000Z",
            "ordinal": 4,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "已定位根因：Redis 端口未初始化，已补齐 mock fixture 并通过 pytest 全量验证。",
                    }
                ],
            },
        },
        {
            "timestamp": "2026-10-15T03:32:16.000Z",
            "ordinal": 5,
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "last_agent_message": "已定位根因：Redis 端口未初始化，已补齐 mock fixture 并通过 pytest 全量验证。",
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
    return path


class TestCodexTranscriptParser:
    def test_parse_codex_rollout_jsonl(self, tmp_path: Path) -> None:
        rollout = _write_sample_rollout(
            tmp_path / "sessions" / "2026" / "10" / "15" / "rollout-2026-10-15T11-29-28-01a13d9b-94b1-7f21-bd65-dbdb47138f79.jsonl"
        )
        parsed = parse_codex_transcript(str(rollout))

        assert parsed["session_id"] == "01a13d9b-94b1-7f21-bd65-dbdb47138f79"
        assert parsed["workspace_roots"] == ["/Users/dev/work/demo-service"]
        assert "ConnectionRefusedError" in parsed["last_user_input"]
        assert "Redis 端口未初始化" in parsed["last_assistant_response"]
        assert len(parsed["turns"]) == 2
        assert len(parsed["recent_tools"]) == 1
        assert parsed["recent_tools"][0]["tool"] == "exec_command"
        assert parsed["recent_tools"][0]["exit_code"] == 0
        assert parsed["max_ordinal"] == 5

        # Generic helper also recognizes Codex rollout format
        assert "Redis 端口未初始化" in read_last_assistant_from_transcript(str(rollout))

        # Resolve by session_id
        resolved = resolve_codex_transcript_path(
            session_id="01a13d9b-94b1-7f21-bd65-dbdb47138f79",
            codex_home=tmp_path,
        )
        assert resolved == str(rollout)


class TestCodexHooks:
    def test_session_start_hook(self, capsys) -> None:
        stdin_payload = json.dumps({"session_id": "codex-s1", "cwd": "/tmp/myproj"})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"l0_manifest": "[TM L0 Manifest | project=myproj]"}

        with (
            patch("sys.stdin", io.StringIO(stdin_payload)),
            patch("team_memory.hooks.platforms.codex.session_start.post_hook", return_value=mock_resp) as m_post,
        ):
            codex_session_start.main()

        m_post.assert_called_once()
        hook_name, body = m_post.call_args[0]
        assert hook_name == "session_start"
        assert body["source_platform"] == "codex"
        assert body["workspace_roots"] == ["/tmp/myproj"]
        out = json.loads(capsys.readouterr().out)
        assert "additionalContext" in out

    def test_user_prompt_submit_hook(self, capsys) -> None:
        stdin_payload = json.dumps({"session_id": "codex-s1", "cwd": "/tmp/myproj", "prompt": "如何配置缓存"})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"l1_context": "[TM L1 Context]\n- [tm://exp/1234] Redis 缓存策略"}

        with (
            patch("sys.stdin", io.StringIO(stdin_payload)),
            patch("team_memory.hooks.platforms.codex.user_prompt_submit.post_hook", return_value=mock_resp) as m_post,
        ):
            codex_prompt.main()

        hook_name, body = m_post.call_args[0]
        assert hook_name == "before_prompt"
        assert body["source_platform"] == "codex"
        out = json.loads(capsys.readouterr().out)
        assert "Redis 缓存策略" in out["additionalContext"]

    def test_post_tool_use_coding_tool(self, capsys) -> None:
        stdin_payload = json.dumps(
            {
                "session_id": "codex-s1",
                "tool_name": "exec_command",
                "tool_input": {"cmd": "pytest -q"},
                "tool_response": "5 passed",
            }
        )
        with (
            patch("sys.stdin", io.StringIO(stdin_payload)),
            patch("team_memory.hooks.platforms.codex.post_tool_use.post_hook_fire_and_forget") as m_post,
        ):
            codex_post_tool.main()

        hook_name, body = m_post.call_args[0]
        assert hook_name == "tool_observation"
        assert body["source_platform"] == "codex"
        assert body["succeeded"] is True
        assert json.loads(capsys.readouterr().out) == {}

    def test_stop_hook_reads_rollout_transcript(self, tmp_path: Path, capsys) -> None:
        rollout = _write_sample_rollout(tmp_path / "rollout-test.jsonl")
        stdin_payload = json.dumps(
            {
                "session_id": "codex-s1",
                "transcript_path": str(rollout),
            }
        )
        with (
            patch("sys.stdin", io.StringIO(stdin_payload)),
            patch("team_memory.hooks.platforms.codex.stop.post_hook_fire_and_forget") as m_post,
        ):
            codex_stop.main()

        hook_name, body = m_post.call_args[0]
        assert hook_name == "after_response"
        assert body["source_platform"] == "codex"
        assert "Redis 端口未初始化" in body["response_text"]
        assert body["workspace_roots"] == ["/Users/dev/work/demo-service"]
        assert len(body["recent_tools"]) == 1
        assert json.loads(capsys.readouterr().out) == {}


class TestCodexWatcherAndCliSetup:
    @pytest.mark.asyncio
    async def test_import_codex_sessions_idempotent(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions" / "2026" / "10" / "15"
        _write_sample_rollout(sessions_dir / "rollout-2026-10-15T11-29-28-01a13d9b-94b1-7f21-bd65-dbdb47138f79.jsonl")

        discovered = discover_codex_transcripts(sessions_dir=tmp_path / "sessions")
        assert len(discovered) == 1

        parsed = parse_codex_transcript(str(discovered[0][1]))
        valuable, reason = evaluate_codex_trajectory(parsed)
        assert valuable is True
        assert reason == "has_tool_trajectory"

        db_path = str(tmp_path / "drafts.db")
        async with ConversationBuffer(db_path=db_path) as buf:
            res1 = await import_codex_sessions(buf=buf, sessions_dir=tmp_path / "sessions", dry_run=False)
            assert res1["imported"] == 1
            assert res1["skipped"] == 0
            assert res1["sessions"][0]["project"] == "demo_service"

            # Second import is idempotent and skips already-queued session
            res2 = await import_codex_sessions(buf=buf, sessions_dir=tmp_path / "sessions", dry_run=False)
            assert res2["imported"] == 0
            assert res2["skipped"] == 1

    def test_setup_check_uninstall_codex_preserves_third_party_hooks(self, tmp_path: Path, capsys) -> None:
        hooks_file = tmp_path / ".codex" / "hooks.json"
        hooks_file.parent.mkdir(parents=True, exist_ok=True)
        hooks_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "UserPromptSubmit": [
                            {"hooks": [{"type": "command", "command": "/usr/local/bin/audit-hook"}]}
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        _setup_codex(cfg=None, venv_python="/usr/bin/python3", scope="project", project_dir=tmp_path)
        data = json.loads(hooks_file.read_text(encoding="utf-8"))
        for ev in ("SessionStart", "UserPromptSubmit", "PostToolUse", "Stop"):
            assert ev in data["hooks"]
            assert "team_memory.hooks.platforms.codex" in json.dumps(data["hooks"][ev])
        # Third-party hook is preserved
        assert "/usr/local/bin/audit-hook" in json.dumps(data["hooks"]["UserPromptSubmit"])

        _check_codex(scope="project", project_dir=tmp_path)
        check_out = capsys.readouterr().out
        assert "[codex] ✅ SessionStart" in check_out
        assert "[codex] ✅ Stop" in check_out

        _uninstall_codex(scope="project", project_dir=tmp_path)
        data_after = json.loads(hooks_file.read_text(encoding="utf-8"))
        assert "team_memory" not in json.dumps(data_after)
        assert "/usr/local/bin/audit-hook" in json.dumps(data_after["hooks"]["UserPromptSubmit"])
