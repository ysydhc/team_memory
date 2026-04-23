"""Stub pipeline module — full implementation in Task 1-2.

Provides async stubs for the daemon pipeline endpoints so that the
FastAPI app is testable without real pipeline logic.
"""

from __future__ import annotations

from typing import Any


async def process_after_response(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Process agent response (draft pipeline)."""
    return {"action": "ok"}


async def process_session_start(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Inject project context at session start."""
    return {"additional_context": "", "project": None}


async def process_before_prompt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Retrieve relevant memories before prompt."""
    return {"results": [], "project": None}


async def process_session_end(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Flush remaining drafts at session end."""
    return {"action": "ok"}
