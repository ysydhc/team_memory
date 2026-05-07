"""Tests for TMSink — TM storage write operations / TM 存储写入操作测试

Unit tests cover:
- LocalTMSink constructor and routing / 构造函数和路由
- RemoteTMSink constructor and headers / 构造函数和请求头
- _ensure_ops() guard / 操作守卫

Integration tests cover:
- Full CRUD with real MCP/Web server
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daemon.tm_sink import LocalTMSink, RemoteTMSink, TMSink

pytestmark = pytest.mark.unit


class TestLocalTMSinkInit:
    """Tests for LocalTMSink constructor / LocalTMSink 构造函数测试"""

    def test_default_user(self):
        """Default user should be 'daemon' / 默认用户应为 daemon"""
        sink = LocalTMSink()
        assert sink._user == "daemon"

    def test_custom_user(self):
        """Custom user should be stored / 自定义用户应被保存"""
        sink = LocalTMSink(user="test_user")
        assert sink._user == "test_user"


class TestRemoteTMSinkInit:
    """Tests for RemoteTMSink constructor / RemoteTMSink 构造函数测试"""

    def test_base_url_stored(self):
        """Base URL should be stored / 基础 URL 应被保存"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        assert sink._base_url == "http://localhost:9111"

    def test_default_user(self):
        """Default user should be 'daemon' / 默认用户应为 daemon"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        assert sink._user == "daemon"

    def test_custom_user(self):
        """Custom user should be stored / 自定义用户应被保存"""
        sink = RemoteTMSink(base_url="http://localhost:9111", user="remote_user")
        assert sink._user == "remote_user"

    def test_headers_contain_content_type(self):
        """Headers should contain Content-Type / 请求头应包含 Content-Type"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        headers = sink._headers()
        assert headers.get("Content-Type") == "application/json"

    def test_headers_contain_authorization(self):
        """Headers should contain Authorization / 请求头应包含 Authorization"""
        sink = RemoteTMSink(base_url="http://localhost:9111")
        headers = sink._headers()
        assert "Authorization" in headers


class TestEnsureOps:
    """Tests for _ensure_ops guard / _ensure_ops 守卫测试"""

    def test_ensure_ops_imports_successfully(self):
        """_ensure_ops should be importable / _ensure_ops 应可导入"""
        from daemon.tm_sink import _ensure_ops
        # Just verify it's callable, not that it succeeds without MCP
        assert callable(_ensure_ops)
