"""FastAPI dependencies for web layer (auth glue).

require_role and require_admin depend on get_current_user; they belong in web
as glue between auth logic and FastAPI. auth.permissions keeps pure logic
(ROLE_PERMISSIONS, has_permission).
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from team_memory.auth.permissions import has_permission
from team_memory.auth.provider import User

logger = logging.getLogger("team_memory.web")


def require_role(*actions: str):
    """FastAPI dependency factory that checks if the current user has
    the required role permissions for the given action(s).

    Usage:
        @router.post("/experiences")
        async def create(user=Depends(require_role("create"))):
            ...

    Multiple actions (requires ALL):
        @router.put("/experiences/{id}")
        async def edit(user=Depends(require_role("update", "delete"))):
            ...
    """
    async def _check_role(request: Request) -> User:
        # Retrieve user from request state (set by get_current_user)
        user: User | None = getattr(request.state, "user", None)

        if user is None:
            # Authenticate via get_current_user logic (lazy import to avoid circular)
            from team_memory.web.app import get_current_user
            user = await get_current_user(request)
            request.state.user = user

        # Check permissions
        for action in actions:
            if not has_permission(user.role, action):
                logger.warning(
                    "Permission denied: user=%s role=%s action=%s",
                    user.name, user.role, action,
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: role '{user.role}' cannot perform '{action}'",
                )

        return user

    return _check_role


def require_admin():
    """Convenience dependency: require admin role."""
    return require_role("admin")
