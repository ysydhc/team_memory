"""Role-based access control (RBAC) for team_memory.

Lightweight permission system built on top of API Key roles.
No separate users table needed — each API Key carries a role.

Roles:
    admin  — Full access to all operations
    editor — Create, update, delete, review experiences
    viewer — Read-only access (search, browse)
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

logger = logging.getLogger("team_memory.auth")

# ============================================================
# Permission Matrix
# ============================================================

# Actions map to operation categories
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"*"},  # Wildcard: all operations
    "editor": {
        "read", "search", "create", "update", "delete",
        "review", "publish", "feedback", "export", "import",
    },
    "viewer": {"read", "search", "feedback"},
}


def has_permission(role: str, action: str) -> bool:
    """Check if a role has permission for a given action.

    Args:
        role: User role (admin, editor, viewer).
        action: Action to check (read, create, update, delete, etc.).

    Returns:
        True if the role has permission, False otherwise.
    """
    perms = ROLE_PERMISSIONS.get(role, set())
    return "*" in perms or action in perms


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
    from team_memory.auth.provider import User

    async def _check_role(request: Request) -> User:
        # Retrieve user from request state (set by get_current_user)
        user: User | None = getattr(request.state, "user", None)

        if user is None:
            # Authenticate via get_current_user logic
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
