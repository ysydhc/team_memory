"""Git utilities for task completion binding."""

from __future__ import annotations

import subprocess


def get_changed_files(project_paths: dict[str, str], project: str) -> tuple[list[str], str | None]:
    """Get list of changed file paths from git diff in the project directory.

    Args:
        project_paths: Mapping of project name to absolute path.
        project: Project identifier (must exist in project_paths).

    Returns:
        Tuple of (list of changed file paths, error message or None).
        On success: (paths, None).
        On failure: ([], error_message).
    """
    if project not in project_paths:
        return ([], f"项目 '{project}' 未在 project_paths 中配置")

    project_dir = project_paths[project]
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=30,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "git command failed").strip()
            return ([], err)
        out = (result.stdout or "").strip()
        paths = [p for p in out.splitlines() if p.strip()]
        return (paths, None)
    except Exception as e:
        return ([], str(e))
