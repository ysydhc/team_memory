"""HermesPipeline — thin version, forwards to TM Daemon HTTP API.

Hermes does not use external Cursor/Claude Code Hook mechanisms.
Instead, it calls this pipeline directly for retrieval and draft management.

This thin version delegates all heavy logic to the TM Daemon HTTP API.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx


class HermesPipeline:
    """Hermes-side memory pipeline — thin HTTP forwarder.

    Delegates on_turn_start, on_turn_end, and on_session_end
    to the TM Daemon HTTP API.

    Usage::

        pipeline = HermesPipeline()
        result = await pipeline.on_turn_start("之前的问题", workspace_roots=["/path/to/project"])
        ...
        result = await pipeline.on_turn_end("sess-1", "解决了", workspace_roots=["/path/to/project"])
        ...
        result = await pipeline.on_session_end("sess-1")
    """

    def __init__(self, daemon_url: str = "http://127.0.0.1:3901") -> None:
        self._daemon_url = daemon_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._daemon_url, timeout=10.0)

    @staticmethod
    def _default_workspace_roots() -> list[str]:
        """Return [os.getcwd()] as default workspace roots."""
        try:
            return [os.getcwd()]
        except OSError:
            return []

    async def on_turn_start(
        self,
        user_message: str,
        project: str | None = None,
        workspace_roots: list[str] | None = None,
    ) -> dict[str, Any]:
        """Called at the start of each conversation turn for auto-retrieval.

        Forwards to daemon /hooks/before_prompt endpoint.

        Args:
            user_message: The user's input message for this turn.
            project: Optional project name to scope the retrieval.
            workspace_roots: Optional list of workspace root paths.
                Defaults to [os.getcwd()] so _resolve_project can match.

        Returns:
            Dict with retrieval results from the daemon.
        """
        payload = {
            "prompt": user_message,
            "project": project or "",
            "workspace_roots": workspace_roots or self._default_workspace_roots(),
        }
        try:
            resp = await self._client.post("/hooks/before_prompt", json=payload)
            return resp.json()
        except httpx.ConnectError:
            return {"action": "ok", "message": "daemon not running"}
        except Exception as e:
            return {"action": "error", "message": str(e)}

    async def on_turn_end(
        self,
        session_id: str,
        agent_response: str,
        project: str | None = None,
        workspace_roots: list[str] | None = None,
        recent_tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Called at the end of each conversation turn to update drafts.

        Forwards to daemon /hooks/after_response endpoint.

        Args:
            session_id: The conversation / session identifier.
            agent_response: The agent's response text for this turn.
            project: Optional project name.
            workspace_roots: Optional list of workspace root paths.
                Defaults to [os.getcwd()] so _resolve_project can match.
            recent_tools: Optional list of recent tool invocations (passed through).

        Returns:
            Dict with draft save/publish result from the daemon.
        """
        payload = {
            "conversation_id": session_id,
            "prompt": agent_response,
            "project": project or "",
            "workspace_roots": workspace_roots or self._default_workspace_roots(),
            "recent_tools": recent_tools or [],
        }
        try:
            resp = await self._client.post("/hooks/after_response", json=payload)
            return resp.json()
        except httpx.ConnectError:
            return {"action": "ok", "message": "daemon not running"}
        except Exception as e:
            return {"action": "error", "message": str(e)}

    async def on_session_end(self, session_id: str) -> dict[str, Any] | None:
        """Force-publish any remaining unpublished drafts at session end.

        Forwards to daemon /hooks/session_end endpoint.

        Args:
            session_id: The conversation / session identifier.

        Returns:
            The result dict from the daemon, or None if daemon not running.
        """
        payload = {
            "conversation_id": session_id,
        }
        try:
            resp = await self._client.post("/hooks/session_end", json=payload)
            return resp.json()
        except httpx.ConnectError:
            return None
        except Exception as e:
            return {"action": "error", "message": str(e)}

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> HermesPipeline:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
