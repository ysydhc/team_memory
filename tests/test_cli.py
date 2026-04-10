"""Tests for tm-cli command-line tool.

Covers archive and upload subcommands with mocked httpx calls.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    *,
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    resp.text = ""
    return resp


# ============================================================
# Archive subcommand
# ============================================================


class TestCliArchive:
    """Test tm-cli archive subcommand."""

    @patch("team_memory.cli.httpx")
    def test_archive_create_success(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        monkeypatch.setenv("TM_BASE_URL", "http://test:9111")

        mock_httpx.post.return_value = _make_mock_response(
            status_code=201,
            json_data={"action": "created", "archive_id": "abc-123"},
        )
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                ["tm-cli", "archive", "--title", "Test Archive", "--solution-doc", "Test solution"],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        mock_httpx.post.assert_called_once()
        call_args = mock_httpx.post.call_args
        # First positional arg is URL
        assert "archives" in call_args[0][0]
        body = call_args[1].get("json", {})
        assert body["title"] == "Test Archive"
        assert body["solution_doc"] == "Test solution"
        assert "created" in captured.getvalue()

    @patch("team_memory.cli.httpx")
    def test_archive_update_response(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        monkeypatch.setenv("TM_BASE_URL", "http://test:9111")

        mock_httpx.post.return_value = _make_mock_response(
            json_data={
                "action": "updated",
                "archive_id": "abc-123",
                "previous_updated_at": "2026-03-30T12:00:00",
            },
        )
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "archive", "--title", "T", "--solution-doc", "S"]),
            patch("sys.stdout", captured),
        ):
            main()

        output = captured.getvalue()
        assert "updated" in output
        assert "previous version" in output

    @patch("team_memory.cli.httpx")
    def test_archive_with_optional_fields(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify optional CLI flags are passed through to the request body."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        monkeypatch.setenv("TM_BASE_URL", "http://test:9111")

        mock_httpx.post.return_value = _make_mock_response(
            status_code=201,
            json_data={"action": "created", "archive_id": "x"},
        )
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "archive",
                    "--title",
                    "T",
                    "--solution-doc",
                    "S",
                    "--content-type",
                    "design_doc",
                    "--value-summary",
                    "summary",
                    "--tags",
                    "a,b,c",
                    "--overview",
                    "overview text",
                    "--project",
                    "proj-x",
                    "--scope",
                    "team",
                    "--scope-ref",
                    "ref-1",
                    "--linked-experience-ids",
                    "id1,id2",
                    "--summary",
                    "conv summary",
                ],
            ),
            patch("sys.stdout", StringIO()),
        ):
            main()

        body = mock_httpx.post.call_args[1]["json"]
        assert body["content_type"] == "design_doc"
        assert body["value_summary"] == "summary"
        assert body["tags"] == ["a", "b", "c"]
        assert body["overview"] == "overview text"
        assert body["project"] == "proj-x"
        assert body["scope"] == "team"
        assert body["scope_ref"] == "ref-1"
        assert body["linked_experience_ids"] == ["id1", "id2"]
        assert body["conversation_summary"] == "conv summary"

    @patch("team_memory.cli.httpx")
    def test_archive_overview_file(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--overview-file reads content from file."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        overview_file = tmp_path / "overview.md"
        overview_file.write_text("# Overview content", encoding="utf-8")

        mock_httpx.post.return_value = _make_mock_response(
            status_code=201,
            json_data={"action": "created", "archive_id": "x"},
        )
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "archive",
                    "--title",
                    "T",
                    "--solution-doc",
                    "S",
                    "--overview-file",
                    str(overview_file),
                ],
            ),
            patch("sys.stdout", StringIO()),
        ):
            main()

        body = mock_httpx.post.call_args[1]["json"]
        assert body["overview"] == "# Overview content"

    @patch("team_memory.cli.httpx")
    def test_archive_solution_file(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--solution-file overrides --solution-doc with file content."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        solution_file = tmp_path / "solution.md"
        solution_file.write_text("File-based solution", encoding="utf-8")

        mock_httpx.post.return_value = _make_mock_response(
            status_code=201,
            json_data={"action": "created", "archive_id": "x"},
        )
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "archive",
                    "--title",
                    "T",
                    "--solution-doc",
                    "original",
                    "--solution-file",
                    str(solution_file),
                ],
            ),
            patch("sys.stdout", StringIO()),
        ):
            main()

        body = mock_httpx.post.call_args[1]["json"]
        assert body["solution_doc"] == "File-based solution"

    @patch("team_memory.cli.httpx")
    def test_archive_http_error(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HTTP error causes sys.exit(1) and prints error to stderr."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 422
        error_resp.text = "Validation error"
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "422", request=MagicMock(), response=error_resp
        )
        mock_httpx.post.return_value = error_resp
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        captured_err = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "archive", "--title", "T", "--solution-doc", "S"]),
            patch("sys.stderr", captured_err),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1
        assert "422" in captured_err.getvalue() or "Error" in captured_err.getvalue()

    @patch("team_memory.cli.httpx")
    def test_archive_connect_error(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Connection error causes sys.exit(1) with helpful message."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")

        mock_httpx.post.side_effect = httpx.ConnectError("refused")
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        captured_err = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "archive", "--title", "T", "--solution-doc", "S"]),
            patch("sys.stderr", captured_err),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1
        assert "Cannot connect" in captured_err.getvalue()


# ============================================================
# API key
# ============================================================


class TestCliApiKey:
    """Test API key resolution."""

    def test_no_api_key_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEAM_MEMORY_API_KEY", raising=False)
        from team_memory.cli import _get_api_key

        with pytest.raises(SystemExit) as exc_info:
            _get_api_key()
        assert exc_info.value.code == 1

    def test_get_api_key_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "my-key")
        from team_memory.cli import _get_api_key

        assert _get_api_key() == "my-key"

    def test_get_base_url_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TM_BASE_URL", raising=False)
        from team_memory.cli import _get_base_url

        assert _get_base_url() == "http://localhost:9111"

    def test_get_base_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TM_BASE_URL", "http://custom:8080")
        from team_memory.cli import _get_base_url

        assert _get_base_url() == "http://custom:8080"


# ============================================================
# Upload subcommand
# ============================================================


class TestCliUpload:
    """Test tm-cli upload subcommand."""

    @patch("team_memory.cli.httpx")
    def test_upload_success(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        monkeypatch.setenv("TM_BASE_URL", "http://test:9111")

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test plan")

        mock_httpx.post.return_value = _make_mock_response(
            json_data={
                "id": "att-456",
                "download_api_path": "/api/v1/archives/abc/attachments/att-456/file",
            },
        )
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        captured = StringIO()
        with (
            patch(
                "sys.argv",
                ["tm-cli", "upload", "--archive-id", "abc-123", "--file", str(test_file)],
            ),
            patch("sys.stdout", captured),
        ):
            main()

        mock_httpx.post.assert_called_once()
        call_args = mock_httpx.post.call_args
        # URL contains the archive-id and "attachments/upload"
        assert "abc-123" in call_args[0][0]
        assert "attachments/upload" in call_args[0][0]
        output = captured.getvalue()
        assert "att-456" in output
        assert "Download" in output

    @patch("team_memory.cli.httpx")
    def test_upload_with_snippet(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--snippet flag is forwarded as form data 'note'."""
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")

        test_file = tmp_path / "note.md"
        test_file.write_text("content")

        mock_httpx.post.return_value = _make_mock_response(
            json_data={"id": "a1", "download_api_path": "/x"},
        )
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "upload",
                    "--archive-id",
                    "x",
                    "--file",
                    str(test_file),
                    "--snippet",
                    "my note",
                ],
            ),
            patch("sys.stdout", StringIO()),
        ):
            main()

        call_data = mock_httpx.post.call_args[1].get("data", {})
        assert call_data.get("note") == "my note"

    @patch("team_memory.cli.httpx")
    def test_upload_with_project(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        test_file = tmp_path / "a.md"
        test_file.write_text("x")
        mock_httpx.post.return_value = _make_mock_response(json_data={"id": "a1"})
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        with (
            patch(
                "sys.argv",
                [
                    "tm-cli",
                    "upload",
                    "--archive-id",
                    "aid",
                    "--file",
                    str(test_file),
                    "--project",
                    "team_memory",
                ],
            ),
            patch("sys.stdout", StringIO()),
        ):
            main()

        call_kw = mock_httpx.post.call_args[1]
        assert call_kw.get("params") == {"project": "team_memory"}

    def test_upload_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")
        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "upload", "--archive-id", "x", "--file", "/nonexistent"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    @patch("team_memory.cli.httpx")
    def test_upload_http_error(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")

        test_file = tmp_path / "big.bin"
        test_file.write_bytes(b"\x00" * 100)

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 413
        error_resp.text = "Payload Too Large"
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "413", request=MagicMock(), response=error_resp
        )
        mock_httpx.post.return_value = error_resp
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli", "upload", "--archive-id", "x", "--file", str(test_file)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    @patch("team_memory.cli.httpx")
    def test_upload_connect_error(
        self, mock_httpx: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TEAM_MEMORY_API_KEY", "test-key")

        test_file = tmp_path / "f.md"
        test_file.write_text("x")

        mock_httpx.post.side_effect = httpx.ConnectError("refused")
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.ConnectError = httpx.ConnectError

        from team_memory.cli import main

        captured_err = StringIO()
        with (
            patch("sys.argv", ["tm-cli", "upload", "--archive-id", "x", "--file", str(test_file)]),
            patch("sys.stderr", captured_err),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1
        assert "Cannot connect" in captured_err.getvalue()


# ============================================================
# No subcommand
# ============================================================


class TestCliNoCommand:
    def test_no_subcommand_exits(self) -> None:
        """Running tm-cli without a subcommand should exit with error."""
        from team_memory.cli import main

        with (
            patch("sys.argv", ["tm-cli"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 2  # argparse exits with 2 for usage errors
