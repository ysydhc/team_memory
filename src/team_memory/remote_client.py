"""Remote HTTP client for team_memory MCP server — RemoteTMClient.

当 MCP server 运行在 remote 模式时，所有 op_* 调用通过 HTTP 转发到
远程 team_memory_service，无需本地 PostgreSQL 或 AppContext。

When running in remote mode, all op_* calls are forwarded via HTTP to a
remote team_memory_service instance — no local DB or AppContext needed.

模式检测逻辑 / Mode detection logic:
  1. TM_SERVER_URL 环境变量 → 使用该 URL / use that URL
  2. ClientConfig.server.url (config.toml) → 使用配置 URL / use config URL
  3. URL 不是 localhost/127.0.0.1 → remote 模式 / remote mode
  4. URL 是 localhost → local 模式（当前行为）/ local mode (current behavior)
  5. TM_REMOTE_MODE=1 → 强制 remote 模式（即使 localhost）/ force remote mode

Usage (automatic — set env var before starting MCP server):
    export TM_SERVER_URL=http://your-server:9111
    export TEAM_MEMORY_API_KEY=***
    python -m team_memory.server   # stdio mode, no local DB needed
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("team_memory.remote_client")


# ============================================================
# RemoteTMClient — 带连接池的 HTTP 远程客户端
# ============================================================


class RemoteTMClient:
    """HTTP client that mirrors the memory_operations API via /api/v1/mcp/* endpoints.

    使用 httpx.AsyncClient 连接池，避免每次请求都创建新连接。
    Uses httpx.AsyncClient with connection pooling for efficiency.

    注意: user 参数不发送到 HTTP API（服务端通过 Bearer token 解析用户身份）。
    Note: the `user` parameter is NOT sent to the HTTP API (resolved server-side).
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        # 连接池: 复用 TCP 连接 / connection pool: reuse TCP connections
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(timeout),
        )

    def _headers(self) -> dict[str, str]:
        """构建请求头，包含 Bearer token / Build headers with Bearer token."""
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def close(self) -> None:
        """关闭连接池 / Close the connection pool."""
        await self._client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> dict:
        """发送 POST 请求到远程 MCP 端点 / POST to remote MCP endpoint."""
        resp = await self._client.post(path, json=body)
        resp.raise_for_status()
        return resp.json()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """发送 GET 请求到远程 MCP 端点 / GET to remote MCP endpoint."""
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # op_* wrappers — 与 memory_operations 签名一致
    # Same signatures as memory_operations; `user` param accepted but not sent
    # ------------------------------------------------------------------

    async def op_save(
        self,
        user: str,
        *,
        service: object | None = None,
        settings: object | None = None,
        db_url: str | None = None,
        embedding_provider: object | None = None,
        title: str | None = None,
        problem: str | None = None,
        solution: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        scope: str = "project",
        experience_type: str | None = None,
        project: str | None = None,
        group_key: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"scope": scope}
        if title is not None:
            body["title"] = title
        if problem is not None:
            body["problem"] = problem
        if solution is not None:
            body["solution"] = solution
        if content is not None:
            body["content"] = content
        if tags is not None:
            body["tags"] = tags
        if experience_type is not None:
            body["experience_type"] = experience_type
        if project is not None:
            body["project"] = project
        if group_key is not None:
            body["group_key"] = group_key
        return await self._post("/api/v1/mcp/save", body)

    async def op_recall(
        self,
        user: str,
        *,
        search_orchestrator: object | None = None,
        db_url: str | None = None,
        settings: object | None = None,
        embedding_provider: object | None = None,
        query: str | None = None,
        problem: str | None = None,
        file_path: str | None = None,
        language: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        max_results: int = 10,
        project: str | None = None,
        include_archives: bool | None = None,
        include_user_profile: bool = False,
    ) -> dict:
        body: dict[str, Any] = {
            "max_results": max_results,
            "include_user_profile": include_user_profile,
        }
        if query is not None:
            body["query"] = query
        if problem is not None:
            body["problem"] = problem
        if file_path is not None:
            body["file_path"] = file_path
        if language is not None:
            body["language"] = language
        if framework is not None:
            body["framework"] = framework
        if tags is not None:
            body["tags"] = tags
        if project is not None:
            body["project"] = project
        if include_archives is not None:
            body["include_archives"] = include_archives
        return await self._post("/api/v1/mcp/recall", body)

    async def op_context(
        self,
        user: str,
        *,
        search_orchestrator: object | None = None,
        settings: object | None = None,
        embedding_provider: object | None = None,
        db_url: str | None = None,
        file_paths: list[str] | None = None,
        task_description: str | None = None,
        project: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if file_paths is not None:
            body["file_paths"] = file_paths
        if task_description is not None:
            body["task_description"] = task_description
        if project is not None:
            body["project"] = project
        return await self._post("/api/v1/mcp/context", body)

    async def op_get_archive(
        self,
        user: str,
        *,
        archive_id: str,
        archive_service: object | None = None,
        project: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {}
        if project is not None:
            params["project"] = project
        return await self._get(f"/api/v1/mcp/archive/{archive_id}", params=params or None)

    async def op_archive_upsert(
        self,
        user: str,
        *,
        title: str,
        solution_doc: str,
        content_type: str = "session_archive",
        value_summary: str | None = None,
        tags: list[str] | None = None,
        overview: str | None = None,
        conversation_summary: str | None = None,
        raw_conversation: str | None = None,
        linked_experience_ids: list[str] | None = None,
        project: str | None = None,
        scope: str = "session",
        scope_ref: str | None = None,
        archive_service: object | None = None,
        settings: object | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "title": title,
            "solution_doc": solution_doc,
            "content_type": content_type,
            "scope": scope,
        }
        if value_summary is not None:
            body["value_summary"] = value_summary
        if tags is not None:
            body["tags"] = tags
        if overview is not None:
            body["overview"] = overview
        if conversation_summary is not None:
            body["conversation_summary"] = conversation_summary
        if raw_conversation is not None:
            body["raw_conversation"] = raw_conversation
        if linked_experience_ids is not None:
            body["linked_experience_ids"] = linked_experience_ids
        if project is not None:
            body["project"] = project
        if scope_ref is not None:
            body["scope_ref"] = scope_ref
        return await self._post("/api/v1/mcp/archive-upsert", body)

    async def op_feedback(
        self,
        user: str,
        *,
        experience_id: str,
        rating: int,
        comment: str | None = None,
        project: str | None = None,
        service: object | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "experience_id": experience_id,
            "rating": rating,
        }
        if comment is not None:
            body["comment"] = comment
        if project is not None:
            body["project"] = project
        return await self._post("/api/v1/mcp/feedback", body)

    async def op_draft_save(
        self,
        user: str,
        *,
        title: str,
        content: str,
        tags: list[str] | None = None,
        project: str | None = None,
        group_key: str | None = None,
        conversation_id: str | None = None,
        service: object | None = None,
        db_url: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"title": title, "content": content}
        if tags is not None:
            body["tags"] = tags
        if project is not None:
            body["project"] = project
        if group_key is not None:
            body["group_key"] = group_key
        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        return await self._post("/api/v1/mcp/draft-save", body)

    async def op_draft_publish(
        self,
        user: str,
        *,
        draft_id: str,
        refined_content: str | None = None,
        project: str | None = None,
        service: object | None = None,
    ) -> dict:
        body: dict[str, Any] = {"draft_id": draft_id}
        if refined_content is not None:
            body["refined_content"] = refined_content
        if project is not None:
            body["project"] = project
        return await self._post("/api/v1/mcp/draft-publish", body)

    async def op_submit_response(
        self,
        user: str,
        *,
        query: str,
        response: str,
        result_ids: list[str] | None = None,
        project: str | None = None,
        db_url: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "response": response}
        if result_ids is not None:
            body["result_ids"] = result_ids
        if project is not None:
            body["project"] = project
        return await self._post("/api/v1/mcp/submit-response", body)


# ============================================================
# Mode detection helpers / 模式检测辅助函数
# ============================================================


def _is_localhost(url: str) -> bool:
    """判断 URL 是否指向本地 / Check if URL points to localhost.

    >>> _is_localhost("http://localhost:9111")
    True
    >>> _is_localhost("http://127.0.0.1:9111")
    True
    >>> _is_localhost("http://0.0.0.0:9111")
    True
    >>> _is_localhost("http://my-server:9111")
    False
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    except Exception:
        return False


def detect_remote_mode() -> tuple[bool, str, str]:
    """检测是否应该使用远程模式 / Detect whether remote mode should be used.

    Returns:
        (is_remote, base_url, api_key):
          - is_remote: True → remote 模式, False → local 模式
          - base_url: 远程服务器 URL (仅远程模式有意义)
          - api_key: Bearer token (仅远程模式有意义)

    优先级 / Priority:
      1. TM_REMOTE_MODE=1 强制远程模式 / force remote mode
      2. TM_SERVER_URL 环境变量 / environment variable
      3. ClientConfig.server.url (config.toml)
      4. URL 不是 localhost → remote 模式
    """
    import os

    # 获取 API key / get API key
    api_key = os.environ.get("TEAM_MEMORY_API_KEY", "")

    # Step 1: TM_SERVER_URL 环境变量 / environment variable
    server_url = os.environ.get("TM_SERVER_URL", "").strip()

    # Step 2: 如果没有环境变量，从 config.toml 读取 / read from config.toml
    if not server_url:
        try:
            from team_memory.client_config import ClientConfig

            config = ClientConfig.load()
            server_url = config.server.url
            if not api_key:
                api_key = config.server.api_key
        except Exception:
            pass

    if not server_url:
        # 没有配置任何 URL → local 模式 / no URL configured → local mode
        return False, "", api_key

    # Step 3: 判断是否远程 / determine if remote
    force_remote = os.environ.get("TM_REMOTE_MODE", "").strip() == "1"

    if force_remote:
        logger.info(
            "TM_REMOTE_MODE=1 — 强制远程模式 / forced remote mode → %s",
            server_url,
        )
        return True, server_url, api_key

    if _is_localhost(server_url):
        logger.info(
            "URL 指向 localhost (%s) → local 模式 / localhost URL → local mode",
            server_url,
        )
        return False, server_url, api_key

    # 非本地 URL → remote 模式 / non-localhost URL → remote mode
    logger.info(
        "非本地 URL (%s) → remote 模式 / non-localhost URL → remote mode",
        server_url,
    )
    return True, server_url, api_key


# ============================================================
# Legacy backward compatibility / 旧版向后兼容
# ============================================================


class RemoteMCPClient(RemoteTMClient):
    """Deprecated: use ``RemoteTMClient`` directly. Alias kept for import stability."""


def setup_remote_ops(base_url: str, api_key: str) -> RemoteTMClient:
    """Monkey-patch memory_operations with remote HTTP implementations.

    旧版接口，保留向后兼容。server.py main() 中 TEAM_MEMORY_REMOTE_URL 仍可使用。
    Legacy interface, kept for backward compatibility.
    """
    from team_memory.services.memory import memory_operations

    client = RemoteTMClient(base_url=base_url, api_key=api_key)

    # Patch each op_* used in server.py
    memory_operations.op_save = client.op_save  # type: ignore[method-assign]
    memory_operations.op_recall = client.op_recall  # type: ignore[method-assign]
    memory_operations.op_context = client.op_context  # type: ignore[method-assign]
    memory_operations.op_get_archive = client.op_get_archive  # type: ignore[method-assign]
    memory_operations.op_archive_upsert = client.op_archive_upsert  # type: ignore[method-assign]
    memory_operations.op_feedback = client.op_feedback  # type: ignore[method-assign]
    memory_operations.op_draft_save = client.op_draft_save  # type: ignore[method-assign]
    memory_operations.op_draft_publish = client.op_draft_publish  # type: ignore[method-assign]
    memory_operations.op_submit_response = client.op_submit_response  # type: ignore[method-assign]

    logger.info(
        "MCP server running in REMOTE mode (legacy) — delegating all ops to %s",
        base_url,
    )
    return client
