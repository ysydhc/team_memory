"""Unit tests for workflow_oracle: parse_workflow_steps, _last_step, _load_workflow."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from team_memory.workflow_oracle import (
    _last_step_from_messages,
    _load_workflow,
    get_next_step,
    parse_workflow_steps_from_messages,
)


def test_load_workflow_resolves_step_ref(tmp_path):
    """$ref in steps is resolved; referenced step has id, name, action."""
    (tmp_path / "main.yaml").write_text("""
meta:
  id: test-workflow
steps:
  - $ref: steps/step-a.yaml
""")
    (tmp_path / "steps").mkdir()
    (tmp_path / "steps" / "step-a.yaml").write_text("""
id: step-a
name: Step A
action: Do A
acceptance_criteria: A done
allowed_next: [step-b]
""")
    data = _load_workflow(tmp_path / "main.yaml")
    assert len(data["steps"]) == 1
    assert data["steps"][0]["id"] == "step-a"
    assert data["steps"][0]["action"] == "Do A"


def test_load_workflow_inline_steps_unchanged(tmp_path):
    """Inline steps (no $ref) work as before."""
    (tmp_path / "w.yaml").write_text("""
meta:
  id: inline-workflow
steps:
  - id: step-1
    name: One
    action: Do one
    acceptance_criteria: done
    allowed_next: []
""")
    data = _load_workflow(tmp_path / "w.yaml")
    assert len(data["steps"]) == 1
    assert data["steps"][0]["id"] == "step-1"
    assert data["steps"][0]["action"] == "Do one"


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


def test_get_next_step_resolves_task_execution_workflow():
    """Integration: real task-execution-workflow with $ref returns full action."""
    root = Path(__file__).resolve().parent.parent
    workflow_path = root / ".cursor" / "plans" / "workflows" / "task-execution-workflow.yaml"
    if not workflow_path.exists():
        pytest.skip(".cursor/workflows not present")
    r = get_next_step("task-execution-workflow", workspace_root=root, current_step_id=None)
    assert r.get("next_step_id") == "step-coldstart"
    action = r.get("action") or ""
    assert len(action) > 50
    assert "tm_task" in action or "冷启动" in action


def test_load_workflow_step_with_optional_metadata(tmp_path):
    """Step with timeout_hint, retry_hint, idempotent is loaded."""
    (tmp_path / "w.yaml").write_text("""
meta:
  id: test
steps:
  - id: step-x
    name: X
    action: Do X
    timeout_hint: 5min
    retry_hint: "retry 2"
    idempotent: true
    allowed_next: []
""")
    data = _load_workflow(tmp_path / "w.yaml")
    s = data["steps"][0]
    assert s.get("timeout_hint") == "5min"
    assert s.get("retry_hint") == "retry 2"
    assert s.get("idempotent") is True


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
