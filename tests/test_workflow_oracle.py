"""Unit tests for workflow_oracle (parse_workflow_steps_from_messages, _last_step_from_messages)."""

from datetime import datetime, timezone

from team_memory.workflow_oracle import (
    _last_step_from_messages,
    parse_workflow_steps_from_messages,
)


def test_parse_workflow_steps_empty():
    assert parse_workflow_steps_from_messages([], "task-execution-workflow") == []


def test_parse_workflow_steps_single_step():
    msgs = [
        {
            "content": "[workflow] task-execution-workflow step-claim: 已完成认领",
            "created_at": datetime(2025, 3, 2, 10, 0, 0, tzinfo=timezone.utc),
        },
    ]
    steps = parse_workflow_steps_from_messages(msgs, "task-execution-workflow")
    assert len(steps) == 1
    assert steps[0]["step_id"] == "step-claim"
    assert steps[0]["last_summary"] == "已完成认领"
    assert steps[0]["status"] == "current"


def test_parse_workflow_steps_aggregated():
    msgs = [
        {"content": "[workflow] task-execution-workflow step-coldstart: 冷启动完成", "created_at": "2025-03-02T10:00:00Z"},  # noqa: E501
        {"content": "[workflow] task-execution-workflow step-claim: 认领", "created_at": "2025-03-02T10:01:00Z"},  # noqa: E501
        {"content": "[workflow] task-execution-workflow step-execute: 执行中", "created_at": "2025-03-02T10:02:00Z"},  # noqa: E501
    ]
    steps = parse_workflow_steps_from_messages(msgs, "task-execution-workflow")
    assert len(steps) == 3
    assert steps[0]["step_id"] == "step-coldstart"
    assert steps[0]["status"] == "done"
    assert steps[1]["step_id"] == "step-claim"
    assert steps[1]["status"] == "done"
    assert steps[2]["step_id"] == "step-execute"
    assert steps[2]["status"] == "current"
    assert steps[2]["last_summary"] == "执行中"


def test_last_step_from_messages():
    msgs = [
        {"content": "[workflow] task-execution-workflow step-claim: x", "created_at": None},
        {"content": "[workflow] task-execution-workflow step-execute: y", "created_at": None},
    ]
    assert _last_step_from_messages(msgs, "task-execution-workflow") == "step-execute"


def test_last_step_from_messages_empty():
    assert _last_step_from_messages([], "task-execution-workflow") is None


def test_parse_workflow_steps_first_line_only_empty_content():
    """Empty content: no match, message does not produce a step."""
    msgs = [{"content": "", "created_at": None}]
    steps = parse_workflow_steps_from_messages(msgs, "task-execution-workflow")
    assert steps == []


def test_parse_workflow_steps_first_line_only_newline_content():
    """Content is only newline: first line is empty, no match."""
    msgs = [{"content": "\n", "created_at": None}]
    steps = parse_workflow_steps_from_messages(msgs, "task-execution-workflow")
    assert steps == []


def test_parse_workflow_steps_first_line_non_audit_second_line_audit():
    """First line non-audit, second line is audit: only first line is matched, so no match."""
    msgs = [
        {
            "content": "some other text\n[workflow] task-execution-workflow step-claim: 认领",
            "created_at": "2025-03-02T10:00:00Z",
        },
    ]
    steps = parse_workflow_steps_from_messages(msgs, "task-execution-workflow")
    assert steps == []
