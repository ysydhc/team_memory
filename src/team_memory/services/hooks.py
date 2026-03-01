"""Hook interface protocol for automatic experience capture.

This module defines the hook interface that can be used by external systems
(e.g., Claude Code hooks, IDE plugins) to automatically capture and compress
agent operations into semantic observations.

Usage:
    When implementing hooks, each hook should:
    1. Implement the HookHandler protocol
    2. Register with the HookRegistry (via init_hook_registry or get_hook_registry)
    3. Process events and optionally save observations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol


class HookEvent(Enum):
    """Supported hook trigger points."""

    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TASK_COMPLETE = "task_complete"
    EXPERIENCE_SAVED = "experience_saved"


@dataclass
class HookContext:
    """Context passed to hook handlers."""

    event: HookEvent
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    session_id: str | None = None
    user: str | None = None
    project: str | None = None
    metadata: dict = field(default_factory=dict)
    # P3-7: optional API key label for usage analytics (e.g. from TEAM_MEMORY_API_KEY_NAME)
    api_key_name: str | None = None


@dataclass
class Observation:
    """A compressed semantic observation from hook processing."""

    content: str
    observation_type: str  # tool_usage, error, decision, pattern
    source: str = "hook"
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class HookHandler(Protocol):
    """Protocol for hook handlers.

    Implementors should process the context and optionally return
    observations to be saved to the knowledge base.
    """

    async def handle(self, context: HookContext) -> list[Observation] | None:
        """Process a hook event and optionally return observations."""
        ...

    @property
    def supported_events(self) -> list[HookEvent]:
        """Return the list of events this handler is interested in."""
        ...


class OTELExporter(Protocol):
    """Reserved interface for future OpenTelemetry integration."""

    async def export_metric(self, name: str, value: float, labels: dict) -> None:
        ...


class NoopExporter:
    """No-op implementation of OTELExporter."""

    async def export_metric(self, name: str, value: float, labels: dict) -> None:
        pass


class HookRegistry:
    """Central registry for hook handlers."""

    def __init__(self, otel_exporter: OTELExporter | None = None) -> None:
        self._handlers: dict[HookEvent, list[HookHandler]] = {}
        self._otel = otel_exporter or NoopExporter()

    def register(self, handler: HookHandler) -> None:
        """Register a handler for its supported events."""
        for event in handler.supported_events:
            self._handlers.setdefault(event, []).append(handler)

    async def fire(self, context: HookContext) -> list[Observation]:
        """Fire an event to all registered handlers and collect observations."""
        observations: list[Observation] = []
        for handler in self._handlers.get(context.event, []):
            try:
                result = await handler.handle(context)
                if result:
                    observations.extend(result)
            except Exception:
                pass  # Don't let hook failures break main flow
        return observations


class UsageTrackingHandler:
    """Tracks tool usage by writing to tool_usage_logs table."""

    def __init__(
        self,
        session_factory: object,
        project: str = "default",
    ) -> None:
        self._session_factory = session_factory
        self._project = project
        self._pending: dict[str, HookContext] = {}

    @property
    def supported_events(self) -> list[HookEvent]:
        return [HookEvent.PRE_TOOL_CALL, HookEvent.POST_TOOL_CALL]

    async def handle(self, context: HookContext) -> list[Observation] | None:
        if context.event == HookEvent.PRE_TOOL_CALL:
            key = f"{context.session_id}:{context.tool_name}"
            self._pending[key] = context
            return None

        if context.event == HookEvent.POST_TOOL_CALL:
            key = f"{context.session_id}:{context.tool_name}"
            pre_ctx = self._pending.pop(key, None)
            duration_ms = None
            if pre_ctx:
                delta = context.timestamp - pre_ctx.timestamp
                duration_ms = int(delta.total_seconds() * 1000)

            success = context.metadata.get("success", True)
            error_msg = context.metadata.get("error_message")

            try:
                from team_memory.storage.models import ToolUsageLog

                async with self._session_factory() as session:
                    log = ToolUsageLog(
                        tool_name=context.tool_name or "unknown",
                        tool_type="mcp",
                        user=context.user or "anonymous",
                        project=context.project or self._project,
                        duration_ms=duration_ms,
                        success=success,
                        error_message=error_msg,
                        session_id=context.session_id,
                        metadata_extra=context.metadata,
                        api_key_name=context.api_key_name,
                    )
                    session.add(log)
                    await session.commit()
            except Exception:
                pass  # Best-effort tracking
            return None

        return None


# Observation compression prompt template for future LLM-based compression
OBSERVATION_COMPRESS_PROMPT = """你是一个编程助手的观察压缩器。
将以下工具调用序列压缩为简洁的语义观察。

工具调用序列:
{tool_calls}

请提取:
1. 核心操作意图（不是具体命令）
2. 关键发现或决策
3. 可复用的模式

以 JSON 格式返回:
{{
  "observations": [
    {{
      "type": "tool_usage|error|decision|pattern",
      "content": "简洁描述",
      "confidence": 0.0-1.0
    }}
  ]
}}"""


_registry: HookRegistry | None = None


def get_hook_registry() -> HookRegistry:
    """Return the singleton hook registry, creating it if needed."""
    global _registry
    if _registry is None:
        _registry = HookRegistry()
    return _registry


def init_hook_registry(
    session_factory: object | None = None,
    project: str = "default",
    otel_exporter: OTELExporter | None = None,
) -> HookRegistry:
    """Initialize the singleton hook registry with optional handlers."""
    global _registry
    _registry = HookRegistry(otel_exporter=otel_exporter)
    if session_factory:
        handler = UsageTrackingHandler(session_factory, project)
        _registry.register(handler)
    return _registry
