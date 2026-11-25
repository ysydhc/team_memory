"""Tenant context management (L1 storage layer).

Provides a process-safe ``ContextVar`` holding the active tenant UUID for the
current async task / HTTP request / MCP invocation, plus deterministic defaults
for the built-in ``default`` tenant so existing single-tenant callers stay 100%
backward compatible.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

# Deterministic UUID for the default tenant created by migration 023.
DEFAULT_TENANT_ID: uuid.UUID = uuid.UUID("00000000-0000-4000-8000-000000000001")
DEFAULT_TENANT_SLUG: str = "default"
DEFAULT_TENANT_NAME: str = "Default Workspace"

_current_tenant_id: ContextVar[uuid.UUID | None] = ContextVar("current_tenant_id", default=None)


def parse_tenant_uuid(value: uuid.UUID | str | None) -> uuid.UUID | None:
    """Parse a UUID or string into ``uuid.UUID``, returning ``None`` if empty."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    return uuid.UUID(raw)


def get_active_tenant_id() -> uuid.UUID | None:
    """Return the explicitly set tenant UUID, or ``None`` if unscoped."""
    return _current_tenant_id.get()


def get_current_tenant_id() -> uuid.UUID:
    """Return the active tenant UUID, falling back to ``DEFAULT_TENANT_ID``."""
    return _current_tenant_id.get() or DEFAULT_TENANT_ID


def set_current_tenant_id(tenant_id: uuid.UUID | str | None) -> Token[uuid.UUID | None]:
    """Set the active tenant UUID in the current async context."""
    parsed = parse_tenant_uuid(tenant_id)
    return _current_tenant_id.set(parsed)


def reset_current_tenant_id(token: Token[uuid.UUID | None]) -> None:
    """Reset the active tenant UUID using the token returned by ``set_current_tenant_id``."""
    _current_tenant_id.reset(token)


@contextmanager
def tenant_scope(tenant_id: uuid.UUID | str | None) -> Iterator[uuid.UUID | None]:
    """Temporarily bind ``tenant_id`` to the current execution context."""
    parsed = parse_tenant_uuid(tenant_id)
    token = _current_tenant_id.set(parsed)
    try:
        yield parsed
    finally:
        _current_tenant_id.reset(token)
