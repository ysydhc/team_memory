"""TMSink 抽象层：统一本地/远程 TeamMemory 访问接口。

- TMSink: 抽象基类，定义 draft_save / draft_publish / save / recall / context
- LocalTMSink: 直接 Python import 调用 team_memory.services.memory_operations
- RemoteTMSink: HTTP 客户端（httpx）调用远程 TM 服务
- create_sink(config): 工厂函数，根据配置创建对应实现
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger("daemon.tm_sink")

# 模块级延迟导入引用 — 方便测试 patch，也避免无 TM 环境下 import 失败
_op_draft_save = None
_op_draft_publish = None
_op_save = None
_op_recall = None
_op_context = None


def _ensure_ops() -> None:
    """首次调用时从 team_memory 导入 op_* 函数。"""
    global _op_draft_save, _op_draft_publish, _op_save, _op_recall, _op_context
    if _op_draft_save is not None:
        return
    from team_memory.services.memory_operations import (
        op_context,
        op_draft_publish,
        op_draft_save,
        op_recall,
        op_save,
    )

    _op_draft_save = op_draft_save
    _op_draft_publish = op_draft_publish
    _op_save = op_save
    _op_recall = op_recall
    _op_context = op_context


# ---------------------------------------------------------------------------
# TMSink 抽象基类
# ---------------------------------------------------------------------------


class TMSink(ABC):
    """TM 操作的抽象接口。"""

    @abstractmethod
    async def draft_save(
        self,
        *,
        title: str,
        content: str,
        tags: list[str] | None = None,
        project: str | None = None,
        group_key: str | None = None,
        conversation_id: str | None = None,
        skip_dedup: bool = False,
    ) -> dict:
        """保存草稿。"""

    @abstractmethod
    async def draft_publish(
        self,
        *,
        draft_id: str,
        refined_content: str | None = None,
    ) -> dict:
        """发布草稿。"""

    @abstractmethod
    async def save(
        self,
        *,
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
        """保存经验（直接保存或 LLM 提取）。"""

    @abstractmethod
    async def recall(
        self,
        *,
        query: str | None = None,
        problem: str | None = None,
        file_path: str | None = None,
        language: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        max_results: int = 5,
        project: str | None = None,
    ) -> list[dict]:
        """搜索经验，返回结果列表。"""

    @abstractmethod
    async def update_experience(
        self,
        *,
        experience_id: str,
        title: str | None = None,
        problem: str | None = None,
        solution: str | None = None,
        tags: list[str] | None = None,
        experience_type: str | None = None,
        exp_status: str | None = None,
    ) -> dict:
        """Update an existing experience in-place."""

    @abstractmethod
    async def context(
        self,
        *,
        file_paths: list[str] | None = None,
        task_description: str | None = None,
        project: str | None = None,
    ) -> dict:
        """获取上下文（用户画像 + 相关经验）。"""


# ---------------------------------------------------------------------------
# LocalTMSink — 直接 Python 调用
# ---------------------------------------------------------------------------


class LocalTMSink(TMSink):
    """通过直接 import team_memory.services.memory_operations 调用。"""

    def __init__(self, user: str = "daemon") -> None:
        self._user = user
        _ensure_ops()

    async def draft_save(
        self,
        *,
        title: str,
        content: str,
        tags: list[str] | None = None,
        project: str | None = None,
        group_key: str | None = None,
        conversation_id: str | None = None,
        skip_dedup: bool = False,
    ) -> dict:
        return await _op_draft_save(
            self._user,
            title=title,
            content=content,
            tags=tags,
            project=project,
            group_key=group_key,
            conversation_id=conversation_id,
            skip_dedup=skip_dedup,
        )

    async def draft_publish(
        self,
        *,
        draft_id: str,
        refined_content: str | None = None,
    ) -> dict:
        return await _op_draft_publish(
            self._user,
            draft_id=draft_id,
            refined_content=refined_content,
        )

    async def save(
        self,
        *,
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
        return await _op_save(
            self._user,
            title=title,
            problem=problem,
            solution=solution,
            content=content,
            tags=tags,
            scope=scope,
            experience_type=experience_type,
            project=project,
            group_key=group_key,
        )

    async def recall(
        self,
        *,
        query: str | None = None,
        problem: str | None = None,
        file_path: str | None = None,
        language: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        max_results: int = 5,
        project: str | None = None,
    ) -> list[dict]:
        raw = await _op_recall(
            self._user,
            query=query,
            problem=problem,
            file_path=file_path,
            language=language,
            framework=framework,
            tags=tags,
            max_results=max_results,
            project=project,
        )
        # op_recall 可能返回 JSON 字符串或 dict，统一处理
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            return raw.get("results", [])
        return []

    async def update_experience(
        self,
        *,
        experience_id: str,
        title: str | None = None,
        problem: str | None = None,
        solution: str | None = None,
        tags: list[str] | None = None,
        experience_type: str | None = None,
        exp_status: str | None = None,
    ) -> dict:
        _ensure_ops()
        from team_memory.services.memory_operations import op_experience_update
        return await op_experience_update(
            self._user,
            experience_id=experience_id,
            title=title,
            problem=problem,
            solution=solution,
            tags=tags,
            experience_type=experience_type,
            exp_status=exp_status,
        )

    async def context(
        self,
        *,
        file_paths: list[str] | None = None,
        task_description: str | None = None,
        project: str | None = None,
    ) -> dict:
        return await _op_context(
            self._user,
            file_paths=file_paths,
            task_description=task_description,
            project=project,
        )


# ---------------------------------------------------------------------------
# RemoteTMSink — HTTP 客户端
# ---------------------------------------------------------------------------


class RemoteTMSink(TMSink):
    """通过 HTTP 调用远程 team_memory_service 的 /mcp/ REST 端点。

    base_url 指向服务根，例如 http://localhost:9111 或 https://tm.example.com
    认证通过 TEAM_MEMORY_API_KEY 环境变量读取，写入 Authorization: Bearer header。
    """

    # MCP 操作名 → 实际 REST 路径（相对于 /mcp/）
    _ROUTES: dict[str, tuple[str, str]] = {
        "draft_save":          ("POST", "/mcp/draft-save"),
        "draft_publish":       ("POST", "/mcp/draft-publish"),
        "save":                ("POST", "/mcp/save"),
        "recall":              ("POST", "/mcp/recall"),
        "context":             ("POST", "/mcp/context"),
        "archive_upsert":      ("POST", "/mcp/archive-upsert"),
        "get_archive":         ("GET",  "/mcp/archive/{archive_id}"),
        "feedback":            ("POST", "/mcp/feedback"),
        # update_experience 暂无 /mcp/ 端点，fallback 到 REST API
        "update_experience":   ("PUT",  "/api/v1/experiences/{experience_id}"),
    }

    def __init__(self, base_url: str, user: str = "daemon") -> None:
        self._base_url = base_url.rstrip("/")
        self._user = user
        import os
        self._api_key = os.environ.get("TEAM_MEMORY_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        path_params: dict[str, str] | None = None,
    ) -> Any:
        """发送 HTTP 请求到 team_memory_service。"""
        if path_params:
            path = path.format(**path_params)
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                resp = await client.get(url, headers=self._headers())
            else:
                resp = await client.request(method, url, json=body or {}, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def draft_save(
        self,
        *,
        title: str,
        content: str,
        tags: list[str] | None = None,
        project: str | None = None,
        group_key: str | None = None,
        conversation_id: str | None = None,
        skip_dedup: bool = False,
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
        if skip_dedup:
            body["skip_dedup"] = skip_dedup
        return await self._request("POST", "/mcp/draft-save", body)

    async def draft_publish(
        self,
        *,
        draft_id: str,
        refined_content: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"draft_id": draft_id}
        if refined_content is not None:
            body["refined_content"] = refined_content
        return await self._request("POST", "/mcp/draft-publish", body)

    async def save(
        self,
        *,
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
        body: dict[str, Any] = {}
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
        if scope is not None:
            body["scope"] = scope
        if experience_type is not None:
            body["experience_type"] = experience_type
        if project is not None:
            body["project"] = project
        if group_key is not None:
            body["group_key"] = group_key
        return await self._request("POST", "/mcp/save", body)

    async def recall(
        self,
        *,
        query: str | None = None,
        problem: str | None = None,
        file_path: str | None = None,
        language: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        max_results: int = 5,
        project: str | None = None,
    ) -> list[dict]:
        body: dict[str, Any] = {}
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
        if max_results != 5:
            body["max_results"] = max_results
        if project is not None:
            body["project"] = project
        raw = await self._request("POST", "/mcp/recall", body)
        if isinstance(raw, dict):
            return raw.get("results", [])
        return []

    async def context(
        self,
        *,
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
        return await self._request("POST", "/mcp/context", body)

    async def update_experience(
        self,
        *,
        experience_id: str,
        title: str | None = None,
        problem: str | None = None,
        solution: str | None = None,
        tags: list[str] | None = None,
        experience_type: str | None = None,
        exp_status: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if problem is not None:
            body["problem"] = problem
        if solution is not None:
            body["solution"] = solution
        if tags is not None:
            body["tags"] = tags
        if experience_type is not None:
            body["experience_type"] = experience_type
        if exp_status is not None:
            body["exp_status"] = exp_status
        return await self._request(
            "PUT", "/api/v1/experiences/{experience_id}", body,
            path_params={"experience_id": experience_id},
        )


# ---------------------------------------------------------------------------
# create_sink 工厂
# ---------------------------------------------------------------------------


def create_sink(config: dict[str, Any]) -> TMSink:
    """根据配置字典创建 TMSink 实例。

    config 需要包含:
      - mode: "local" | "remote"
      - user: 用户名（可选，默认 "daemon"）
      - base_url: 远程地址（mode="remote" 时必需）
    """
    mode = config.get("mode", "local")
    user = config.get("user", "daemon")

    if mode == "local":
        return LocalTMSink(user=user)
    elif mode == "remote":
        base_url = config.get("base_url")
        if not base_url:
            raise ValueError("Remote mode requires 'base_url' in config")
        return RemoteTMSink(base_url=base_url, user=user)
    else:
        raise ValueError(f"Unsupported tm mode: {mode!r}, expected 'local' or 'remote'")
