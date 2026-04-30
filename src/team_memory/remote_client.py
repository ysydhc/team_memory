"""Remote HTTP client for team_memory MCP server.

When TEAM_MEMORY_REMOTE_URL is set, the MCP server delegates all op_*
calls to the remote team_memory_service via its /mcp/ REST endpoints
instead of bootstrapping a local AppContext and connecting to PostgreSQL.

Usage (automatic — set env var before starting MCP server):
    export TEAM_MEMORY_REMOTE_URL=http://your-server:9111
    export TEAM_MEMORY_API_KEY=your-key
    python -m team_memory.server   # stdio mode, no local DB needed

Design:
    - RemoteMCPClient wraps every op_* function used in server.py
    - setup_remote_ops() monkey-patches team_memory.services.memory_operations
      so server.py tool handlers transparently call remote HTTP
    - No AppContext / bootstrap / PostgreSQL needed in remote mode
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("team_memory.remote_client")


class RemoteMCPClient:
    """HTTP client that calls /mcp/ endpoints on a remote team_memory_service.

    Mirrors the op_* function signatures used in memory_operations so it
    can be used as a drop-in patch target.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _post(self, path: str, body: dict[str, Any]) -> dict:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, params=params, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # op_* wrappers — same signatures as memory_operations
    # ------------------------------------------------------------------

    async def op_save(
        self,
        user: str,
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
        query: str | None = None,
        problem: str | None = None,
        file_path: str | None = None,
        language: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
        max_results: int = 5,
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
    ) -> dict:
        body: dict[str, Any] = {
            "experience_id": experience_id,
            "rating": rating,
        }
        if comment is not None:
            body["comment"] = comment
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
    ) -> dict:
        body: dict[str, Any] = {"draft_id": draft_id}
        if refined_content is not None:
            body["refined_content"] = refined_content
        return await self._post("/api/v1/mcp/draft-publish", body)


def setup_remote_ops(base_url: str, api_key: str) -> RemoteMCPClient:
    """Monkey-patch memory_operations with remote HTTP implementations.

    Called by server.py main() when TEAM_MEMORY_REMOTE_URL is set.
    After this call, all MCP tool handlers transparently delegate to the
    remote team_memory_service — no local DB or AppContext needed.

    Returns the RemoteMCPClient for testing / inspection.
    """
    from team_memory.services import memory_operations

    client = RemoteMCPClient(base_url=base_url, api_key=api_key)

    # Patch each op_* used in server.py
    memory_operations.op_save = client.op_save  # type: ignore[method-assign]
    memory_operations.op_recall = client.op_recall  # type: ignore[method-assign]
    memory_operations.op_context = client.op_context  # type: ignore[method-assign]
    memory_operations.op_get_archive = client.op_get_archive  # type: ignore[method-assign]
    memory_operations.op_archive_upsert = client.op_archive_upsert  # type: ignore[method-assign]
    memory_operations.op_feedback = client.op_feedback  # type: ignore[method-assign]
    memory_operations.op_draft_save = client.op_draft_save  # type: ignore[method-assign]
    memory_operations.op_draft_publish = client.op_draft_publish  # type: ignore[method-assign]

    logger.info(
        "MCP server running in REMOTE mode — delegating all ops to %s", base_url
    )
    return client
