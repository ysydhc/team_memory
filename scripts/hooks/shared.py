"""TMClient — async HTTP client for TeamMemory MCP endpoints.

Wraps the draft_save / draft_publish / memory_context / memory_recall
MCP-over-HTTP calls used by the hook pipeline.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class HookInput:
    """Parsed input from a Cursor / Claude Code hook (stdin JSON).

    Attributes:
        workspace_roots: List of workspace root paths.
        prompt: User prompt text (beforeSubmitPrompt only).
        conversation_id: Unique conversation / session identifier.
    """

    workspace_roots: list[str] = field(default_factory=list)
    prompt: str = ""
    conversation_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookInput:
        """Create HookInput from a raw dict (typically parsed from stdin JSON)."""
        return cls(
            workspace_roots=data.get("workspace_roots", []),
            prompt=data.get("prompt", ""),
            conversation_id=data.get("conversation_id", ""),
        )


@dataclass
class PipelineConfig:
    """Configuration for the hook pipeline.

    Attributes:
        tm_url: Base URL for the TeamMemory MCP-over-HTTP endpoint.
        session_start_top_k: Number of context items to fetch on session start.
        max_context_chars: Maximum characters for additional_context output.
    """

    tm_url: str = "http://localhost:3900"
    session_start_top_k: int = 3
    max_context_chars: int = 2000


# ---------------------------------------------------------------------------
# TMClient
# ---------------------------------------------------------------------------

class TMClient:
    """Async HTTP client for TeamMemory MCP tools.

    Usage::

        tm = TMClient("http://localhost:3900")
        result = await tm.draft_save("title", "content", project="team_doc")
        await tm.draft_publish(result["id"], refined_content="...")
        context = await tm.get_context(project="team_doc")
        results = await tm.recall(query="之前的问题", project="team_doc")
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def _call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """POST to the MCP-over-HTTP endpoint and return parsed JSON."""
        url = f"{self._base_url}/{tool_name}"
        payload = {"arguments": arguments}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def draft_save(
        self,
        title: str,
        content: str,
        project: str | None = None,
        group_key: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Call the memory_draft_save MCP tool.

        Returns:
            Dict with at least an ``id`` key identifying the saved draft.
        """
        args: dict[str, Any] = {"title": title, "content": content}
        if project is not None:
            args["project"] = project
        if group_key is not None:
            args["group_key"] = group_key
        if conversation_id is not None:
            args["conversation_id"] = conversation_id
        return await self._call("memory_draft_save", args)

    async def draft_publish(
        self,
        draft_id: str,
        refined_content: str | None = None,
    ) -> dict[str, Any]:
        """Call the memory_draft_publish MCP tool.

        Args:
            draft_id: The draft ID returned by draft_save.
            refined_content: Optional refined content to replace the draft body.

        Returns:
            Parsed JSON response from the MCP endpoint.
        """
        args: dict[str, Any] = {"draft_id": draft_id}
        if refined_content is not None:
            args["refined_content"] = refined_content
        return await self._call("memory_draft_publish", args)

    async def get_context(self, project: str | None = None) -> dict[str, Any] | None:
        """Call the memory_context MCP tool.

        Retrieves project-relevant memories for injection into a new session.

        Args:
            project: Optional project name to scope the context query.

        Returns:
            Parsed JSON response, or None if the call fails.
        """
        args: dict[str, Any] = {}
        if project is not None:
            args["project"] = project
        return await self._call("memory_context", args)

    async def recall(
        self,
        query: str,
        project: str | None = None,
        max_results: int = 3,
    ) -> dict[str, Any] | None:
        """Call the memory_recall MCP tool.

        Searches TeamMemory for experiences matching the query.

        Args:
            query: Search query text (typically the user's prompt).
            project: Optional project name to scope the search.
            max_results: Maximum number of results to return.

        Returns:
            Parsed JSON response, or None if the call fails.
        """
        args: dict[str, Any] = {"query": query, "max_results": max_results}
        if project is not None:
            args["project"] = project
        return await self._call("memory_recall", args)
