"""Default admin initialization for db_api_key auth.

Ensures an admin user exists on first start when TEAM_MEMORY_ADMIN_PASSWORD is set.
"""

from __future__ import annotations

import os


async def is_api_keys_empty(db_url: str) -> bool:
    """Return True if api_keys table has no rows."""
    from sqlalchemy import func, select

    from team_memory.storage.database import get_session
    from team_memory.storage.models import ApiKey

    async with get_session(db_url) as session:
        r = await session.execute(select(func.count()).select_from(ApiKey))
        return r.scalar() == 0


async def ensure_default_admin(db_url: str, password: str) -> bool:
    """Create admin user if api_keys table is empty.

    password is required. Returns True if admin was created, False if table
    already had users (no-op).
    """
    from sqlalchemy import func, select

    from team_memory.auth.provider import _hash_password
    from team_memory.storage.database import get_session
    from team_memory.storage.models import ApiKey

    async with get_session(db_url) as session:
        r = await session.execute(select(func.count()).select_from(ApiKey))
        if r.scalar() > 0:
            return False
        admin_username = os.environ.get("TEAM_MEMORY_DEFAULT_ADMIN", "admin")
        pwd_hash = _hash_password(password)
        admin = ApiKey(
            user_name=admin_username,
            role="admin",
            is_active=True,
            password_hash=pwd_hash,
        )
        session.add(admin)
        await session.flush()
        return True
