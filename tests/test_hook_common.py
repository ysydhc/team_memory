"""Tests for scripts/hooks/common.py — shared utility functions."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/hooks/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "hooks"))

import common


# ---------------------------------------------------------------------------
# parse_hook_input
# ---------------------------------------------------------------------------

class TestParseHookInput:
    """parse_hook_input reads JSON from stdin and returns a dict."""

    def test_basic_payload(self):
        payload = {
            "conversation_id": "xxx",
            "prompt": "hello",
            "workspace_roots": ["/path/to/project"],
        }
        with patch("sys.stdin", new_callable=lambda: io.StringIO(json.dumps(payload))):
            # Re-import to pick up patched stdin; or call directly
            result = common.parse_hook_input()
        assert result == payload
        assert result["conversation_id"] == "xxx"
        assert result["prompt"] == "hello"

    def test_empty_json(self):
        with patch("sys.stdin", new_callable=lambda: io.StringIO("{}")):
            result = common.parse_hook_input()
        assert result == {}

    def test_invalid_json_returns_empty(self):
        with patch("sys.stdin", new_callable=lambda: io.StringIO("not json")):
            result = common.parse_hook_input()
        assert result == {}


# ---------------------------------------------------------------------------
# get_project_from_path
# ---------------------------------------------------------------------------

class TestGetProjectFromPath:
    """get_project_from_path maps a workspace path to a TM project name."""

    def test_ad_learning(self):
        config = common.load_config()
        result = common.get_project_from_path("/Users/yeshouyou/Work/ad_learning", config)
        assert result == "ad_learning"

    def test_team_doc(self):
        config = common.load_config()
        result = common.get_project_from_path("/Users/yeshouyou/Work/agent/team_doc", config)
        assert result == "team_doc"

    def test_ai_learning(self):
        config = common.load_config()
        result = common.get_project_from_path("/home/user/projects/ai_learning", config)
        assert result == "ai_learning"

    def test_no_match_returns_none(self):
        config = common.load_config()
        result = common.get_project_from_path("/some/random/path", config)
        assert result is None


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """load_config reads config.yaml from the same directory as common.py."""

    def test_returns_dict(self):
        config = common.load_config()
        assert isinstance(config, dict)

    def test_has_tm_section(self):
        config = common.load_config()
        assert "tm" in config
        assert "base_url" in config["tm"]
        assert config["tm"]["base_url"] == "http://localhost:3900"

    def test_has_projects_section(self):
        config = common.load_config()
        assert "projects" in config
        # projects is now a list of dicts with name + path_patterns
        if isinstance(config["projects"], list):
            names = [p["name"] for p in config["projects"]]
            assert "team_doc" in names
            assert "ad_learning" in names
            assert "ai_learning" in names
        else:
            assert "team_doc" in config["projects"]
            assert "ad_learning" in config["projects"]
            assert "ai_learning" in config["projects"]

    def test_has_retrieval_section(self):
        config = common.load_config()
        assert "retrieval" in config
        assert config["retrieval"]["session_start_top_k"] == 3

    def test_has_draft_section(self):
        config = common.load_config()
        assert "draft" in config
        assert config["draft"]["max_age_minutes"] == 30


# ---------------------------------------------------------------------------
# call_mcp_tool
# ---------------------------------------------------------------------------

class TestCallMcpTool:
    """call_mcp_tool POSTs to TM's MCP-over-HTTP endpoint."""

    @staticmethod
    def _mock_response(data: dict):
        """Build a lightweight mock httpx.Response."""
        return type("Resp", (), {
            "status_code": 200,
            "json": lambda self: data,
            "raise_for_status": lambda self: None,
        })()

    def test_posts_correct_payload(self):
        with patch("httpx.post", return_value=self._mock_response({"ok": True})) as mock_post:
            result = common.call_mcp_tool("memory_search", {"query": "test"})
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "memory_search" in call_kwargs[0][0] or "memory_search" in str(call_kwargs)
        assert result == {"ok": True}

    def test_uses_base_url_from_config(self):
        with patch("httpx.post", return_value=self._mock_response({})) as mock_post:
            common.call_mcp_tool("memory_search", {"query": "test"})
        url = mock_post.call_args[0][0]
        assert url.startswith("http://localhost:3900")
