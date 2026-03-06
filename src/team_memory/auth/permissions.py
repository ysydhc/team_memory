"""Role-based access control (RBAC) for team_memory.

Lightweight permission system built on top of API Key roles.
No separate users table needed — each API Key carries a role.

Roles:
    admin  — Full access to all operations
    editor — Create, update, delete, review experiences
    viewer — Read-only access (search, browse)

FastAPI dependencies (require_role, require_admin) live in web.dependencies
because they depend on get_current_user (web layer).
"""

from __future__ import annotations

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
