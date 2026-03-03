"""Workflow step oracle: load YAML and return next step from current step or messages.

Used by tm_workflow_next_step MCP tool. Reads .cursor/plans/workflows/{workflow_id}.yaml.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def _workflow_path(workflow_id: str, workspace_root: Path) -> Path:
    """Path to workflow YAML; workflow_id is e.g. task-execution-workflow."""
    return workspace_root / ".cursor" / "plans" / "workflows" / f"{workflow_id}.yaml"


def _resolve_step_ref(step: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve step $ref to full step dict; return step unchanged if no $ref."""
    if set(step.keys()) == {"$ref"}:
        ref_path = base_dir / step["$ref"]
        if not ref_path.exists():
            raise FileNotFoundError(f"Step ref not found: {ref_path}")
        ref_raw = ref_path.read_text(encoding="utf-8")
        resolved = yaml.safe_load(ref_raw)
        if not resolved or not isinstance(resolved, dict):
            raise ValueError(f"Invalid step file: {ref_path}")
        return resolved
    return step


def _load_workflow(path: Path) -> dict[str, Any]:
    """Load workflow YAML; returns dict with meta and steps. Resolves $ref in steps."""
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not data or "steps" not in data:
        raise ValueError(f"Invalid workflow: no steps in {path}")
    base_dir = path.parent
    resolved_steps: list[dict[str, Any]] = []
    for step in data["steps"]:
        if isinstance(step, dict):
            resolved_steps.append(_resolve_step_ref(step, base_dir))
        else:
            resolved_steps.append(step)
    data["steps"] = resolved_steps
    return data


def _step_by_id(steps: list[dict], step_id: str) -> dict | None:
    """Return first step with id == step_id."""
    for s in steps:
        if s.get("id") == step_id:
            return s
    return None


def _next_step_id(step: dict, group_completed: bool = False) -> str | None:
    """From a step with allowed_next or when, return the next step id."""
    if "when" in step:
        for branch in step["when"]:
            cond = branch.get("condition", "")
            if cond == "group_completed" and group_completed:
                return branch.get("next")
            if cond == "else" and not group_completed:
                return branch.get("next")
        for branch in step["when"]:
            if branch.get("condition") == "else":
                return branch.get("next")
        return None
    allowed = step.get("allowed_next") or []
    return allowed[0] if allowed else None


def _last_step_from_messages(
    messages: list[dict],
    workflow_id: str,
) -> str | None:
    """Parse messages for last [workflow] <workflow_id> step-<id>, return step id."""
    steps = parse_workflow_steps_from_messages(messages, workflow_id)
    return steps[-1]["step_id"] if steps else None


def parse_workflow_steps_from_messages(
    messages: list[dict],
    workflow_id: str,
) -> list[dict[str, Any]]:
    """Parse messages for [workflow] <workflow_id> step-<id>: <summary>, return aggregated steps.

    Returns list of dicts: step_id, last_summary, last_at, status ("done" | "current").
    Order is by first appearance of each step; the last step in list is "current", others "done".
    """
    # Match: [workflow] <workflow_id> step-<id> or step-<id>: <optional summary>
    # Also allow legacy step-\d for backwards compatibility.
    pattern = re.compile(
        r"\[workflow\]\s+"
        + re.escape(workflow_id)
        + r"\s+(step-(?:[a-z][a-z0-9-]*|\d+(?:\.\d+)?))(?:\s*:\s*(.*))?",
        re.IGNORECASE | re.DOTALL,
    )
    # step_id -> { step_id, last_summary, last_at }
    # Only first line of each message is used for matching (phase 5: avoid summary pollution).
    by_step: dict[str, dict[str, Any]] = {}
    for m in messages:
        content = (m.get("content") or "") if isinstance(m, dict) else getattr(m, "content", "")
        first_line = content.split("\n")[0] if content else ""
        created = m.get("created_at") if isinstance(m, dict) else getattr(m, "created_at", None)
        match = pattern.search(first_line)
        if match:
            step_id = match.group(1)
            summary = (match.group(2) or "").strip() if match.lastindex >= 2 else ""
            by_step[step_id] = {
                "step_id": step_id,
                "last_summary": summary[:200] if summary else "",
                "last_at": (
                    created.isoformat()
                    if hasattr(created, "isoformat")
                    else str(created) if created else None
                ),
            }
    # Preserve order of first appearance (same first-line rule)
    seen_order: list[str] = []
    for m in messages:
        content = (m.get("content") or "") if isinstance(m, dict) else getattr(m, "content", "")
        first_line = content.split("\n")[0] if content else ""
        match = pattern.search(first_line)
        if match:
            sid = match.group(1)
            if sid not in seen_order:
                seen_order.append(sid)
    result = []
    for i, step_id in enumerate(seen_order):
        row = dict(by_step[step_id])
        row["status"] = "current" if i == len(seen_order) - 1 else "done"
        result.append(row)
    return result


def _step_optional_metadata(step: dict[str, Any]) -> dict[str, Any]:
    """Extract optional metadata (timeout_hint, retry_hint, idempotent) from step."""
    extra: dict[str, Any] = {}
    for k in ("timeout_hint", "retry_hint", "idempotent"):
        if k in step:
            extra[k] = step[k]
    return extra


def get_next_step(
    workflow_id: str,
    workspace_root: Path | None = None,
    current_step_id: str | None = None,
    group_completed: bool = False,
) -> dict[str, Any]:
    """Return next step after current_step_id for the given workflow.

    If current_step_id is None, returns the first step. Uses allowed_next or when
    (with group_completed) to determine next.

    Returns:
        dict with keys: next_step_id, name, action, acceptance_criteria, workflow_id,
        plus optional timeout_hint, retry_hint, idempotent when present in step.
        If workflow not found or no next step, raises or returns error dict.
    """
    root = workspace_root or Path.cwd()
    path = _workflow_path(workflow_id, root)
    data = _load_workflow(path)
    steps = data["steps"]
    meta = data.get("meta") or {}
    resolved_id = meta.get("id") or workflow_id

    if not steps:
        return {
            "next_step_id": None,
            "name": None,
            "action": None,
            "acceptance_criteria": None,
            "workflow_id": resolved_id,
            "error": "Workflow has no steps",
        }

    if current_step_id is None:
        first = steps[0]
        return {
            "next_step_id": first.get("id"),
            "name": first.get("name"),
            "action": first.get("action"),
            "acceptance_criteria": first.get("acceptance_criteria"),
            "workflow_id": resolved_id,
            **_step_optional_metadata(first),
        }

    current = _step_by_id(steps, current_step_id)
    if not current:
        return {
            "next_step_id": None,
            "name": None,
            "action": None,
            "acceptance_criteria": None,
            "workflow_id": resolved_id,
            "error": f"Step not found: {current_step_id}",
        }

    next_id = _next_step_id(current, group_completed=group_completed)
    if not next_id:
        return {
            "next_step_id": None,
            "name": None,
            "action": None,
            "acceptance_criteria": None,
            "workflow_id": resolved_id,
            "error": f"No allowed next step from {current_step_id}",
        }

    next_step = _step_by_id(steps, next_id)
    if not next_step:
        return {
            "next_step_id": next_id,
            "name": None,
            "action": None,
            "acceptance_criteria": None,
            "workflow_id": resolved_id,
            "error": f"Next step definition not found: {next_id}",
        }

    return {
        "next_step_id": next_step.get("id"),
        "name": next_step.get("name"),
        "action": next_step.get("action"),
        "acceptance_criteria": next_step.get("acceptance_criteria"),
        "workflow_id": resolved_id,
        **_step_optional_metadata(next_step),
    }


def get_next_step_for_task(
    workflow_id: str,
    task_id: str,
    messages: list[dict],
    workspace_root: Path | None = None,
    group_completed: bool = False,
) -> dict[str, Any]:
    """Infer current step from task messages, then return next step."""
    current = _last_step_from_messages(messages, workflow_id)
    return get_next_step(
        workflow_id=workflow_id,
        workspace_root=workspace_root,
        current_step_id=current,
        group_completed=group_completed,
    )
