"""Audit log helpers for sensitive operations (P2-3)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from team_memory.storage.models import AuditLog


async def write_audit_log(
    session: AsyncSession,
    user_name: str,
    action: str,
    target_type: str,
    target_id: str | None = None,
    detail: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """Append one row to audit_logs. Caller is responsible for commit."""
    entry = AuditLog(
        user_name=user_name,
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail,
        ip_address=ip_address,
    )
    session.add(entry)
    await session.flush()
