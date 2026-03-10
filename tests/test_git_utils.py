"""Tests for git_utils."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from team_memory.utils.git_utils import get_changed_files


class TestGetChangedFiles:
    """Tests for get_changed_files."""

    def test_get_changed_files_returns_paths_from_git(self):
        """When git diff succeeds, return list of changed file paths."""
        project_paths = {"team_doc": "/path/to/team_doc"}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "src/foo.py\nsrc/bar.py\n"
        mock_result.stderr = ""

        with patch("team_memory.utils.git_utils.subprocess.run", return_value=mock_result):
            paths, err = get_changed_files(project_paths, "team_doc")

        assert paths == ["src/foo.py", "src/bar.py"]
        assert err is None

    def test_get_changed_files_project_not_configured(self):
        """When project not in project_paths, return empty list and error message."""
        project_paths = {"other": "/path/to/other"}
        paths, err = get_changed_files(project_paths, "team_doc")

        assert paths == []
        assert err == "项目 'team_doc' 未在 project_paths 中配置"

    def test_get_changed_files_git_timeout_or_failure(self):
        """When git raises TimeoutError, return empty list and error message."""
        project_paths = {"team_doc": "/path/to/team_doc"}

        with patch(
            "team_memory.utils.git_utils.subprocess.run",
            side_effect=TimeoutError("Command timed out"),
        ):
            paths, err = get_changed_files(project_paths, "team_doc")

        assert paths == []
        assert err == "Command timed out"
