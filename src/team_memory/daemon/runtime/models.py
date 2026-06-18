"""Pydantic request/response models for daemon runtime endpoints.

Shared across route sub-modules.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("daemon.app")


class HookPayload(BaseModel):
    """Payload for hook callback endpoints.

    Lenient mode: accepts extra fields and logs them for contract derivation.
    宽松模式: 接受额外字段并记录，用于契约推导。
    """

    model_config = ConfigDict(extra="allow")

    conversation_id: str = ""
    prompt: str = ""
    response_text: str = ""
    agent_response: str = ""  # Cursor afterAgentResponse format
    content: str = ""  # Some platforms use content
    workspace_roots: list[str] = Field(default_factory=list)
    model: str = ""

    @model_validator(mode="before")
    @classmethod
    def _log_extra_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = {
                "conversation_id",
                "prompt",
                "response_text",
                "agent_response",
                "content",
                "workspace_roots",
                "model",
            }
            extra = set(data.keys()) - known
            if extra:
                logger.info("[CONTRACT] HookPayload extra fields: %s", sorted(extra))
        return data


class DraftSavePayload(BaseModel):
    """Payload for /draft/save endpoint."""

    model_config = ConfigDict(extra="allow")

    title: str
    content: str
    project: str | None = None
    group_key: str | None = None
    conversation_id: str | None = None


class DraftPublishPayload(BaseModel):
    """Payload for /draft/publish endpoint."""

    draft_id: str
    refined_content: str | None = None


class MCPCallbackPayload(BaseModel):
    """Payload for /hooks/mcp_callback — PostToolUse hook reports MCP recall results.

    Called by Claude Code PostToolUse / Cursor afterMCPExecution / Hermes post_tool_call
    after the AI successfully invokes a TM recall tool.  The hook extracts the
    ``search_log_id`` from the tool_response and registers it with the daemon so
    that the subsequent ``after_response`` can correlate the MCP search_log with
    the AI's final answer.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    tool_use_id: str = ""
    search_log_id: str = ""
    tool_name: str = ""
    result_ids: list[dict] = Field(default_factory=list)


class RecallQuery(BaseModel):
    """Query params for /recall endpoint."""

    query: str | None = None
    project: str | None = None
    max_results: int = 5
