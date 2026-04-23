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
    ) -> dict:
        return await _op_draft_save(
            self._user,
            title=title,
            content=content,
            tags=tags,
            project=project,
            group_key=group_key,
            conversation_id=conversation_id,
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
    """通过 HTTP POST 调用远程 TM 服务的 MCP 端点。"""

    def __init__(self, base_url: str, user: str = "daemon") -> None:
        self._base_url = base_url.rstrip("/")
        self._user = user

    async def _post(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """发送 POST 请求到远程 TM 服务。"""
        url = f"{self._base_url}/{tool_name}"
        payload = {"arguments": arguments}
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
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
    ) -> dict:
        args: dict[str, Any] = {"title": title, "content": content}
        if tags is not None:
            args["tags"] = tags
        if project is not None:
            args["project"] = project
        if group_key is not None:
            args["group_key"] = group_key
        if conversation_id is not None:
            args["conversation_id"] = conversation_id
        return await self._post("memory_draft_save", args)

    async def draft_publish(
        self,
        *,
        draft_id: str,
        refined_content: str | None = None,
    ) -> dict:
        args: dict[str, Any] = {"draft_id": draft_id}
        if refined_content is not None:
            args["refined_content"] = refined_content
        return await self._post("memory_draft_publish", args)

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
        args: dict[str, Any] = {}
        if title is not None:
            args["title"] = title
        if problem is not None:
            args["problem"] = problem
        if solution is not None:
            args["solution"] = solution
        if content is not None:
            args["content"] = content
        if tags is not None:
            args["tags"] = tags
        if scope is not None:
            args["scope"] = scope
        if experience_type is not None:
            args["experience_type"] = experience_type
        if project is not None:
            args["project"] = project
        if group_key is not None:
            args["group_key"] = group_key
        return await self._post("memory_save", args)

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
        args: dict[str, Any] = {}
        if query is not None:
            args["query"] = query
        if problem is not None:
            args["problem"] = problem
        if file_path is not None:
            args["file_path"] = file_path
        if language is not None:
            args["language"] = language
        if framework is not None:
            args["framework"] = framework
        if tags is not None:
            args["tags"] = tags
        if max_results != 5:
            args["max_results"] = max_results
        if project is not None:
            args["project"] = project
        raw = await self._post("memory_recall", args)
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
        args: dict[str, Any] = {}
        if file_paths is not None:
            args["file_paths"] = file_paths
        if task_description is not None:
            args["task_description"] = task_description
        if project is not None:
            args["project"] = project
        return await self._post("memory_context", args)


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
