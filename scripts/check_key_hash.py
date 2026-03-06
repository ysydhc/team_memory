#!/usr/bin/env python3
"""Check which user a key_hash belongs to and verify admin key association."""

import asyncio
import hashlib
import os
import sys

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


async def main():
    from team_memory.config import get_settings
    from team_memory.storage.database import get_session
    from team_memory.storage.models import ApiKey
    from sqlalchemy import select

    settings = get_settings()
    db_url = settings.database.url

    target_hash = "0D5007FEF6A98F5A99ED521327C9A698"
    # SHA256 is 64 chars; user might have given 32
    if len(target_hash) == 32:
        target_prefix = target_hash
    else:
        target_prefix = target_hash[:32]

    print(f"查找 key_hash 前缀: {target_prefix}...")
    print()

    async with get_session(db_url) as session:
        # Find by prefix (key_hash is full 64-char SHA256)
        result = await session.execute(
            select(ApiKey).where(ApiKey.key_hash.like(f"{target_prefix}%"))
        )
        keys = result.scalars().all()
        if keys:
            for k in keys:
                print(f"  user_name={k.user_name}, role={k.role}, is_active={k.is_active}")
        else:
            print("  未找到匹配的 key_hash")

        # Also show admin row
        print()
        print("admin 账户当前状态:")
        result2 = await session.execute(
            select(ApiKey).where(ApiKey.user_name == "admin")
        )
        admin = result2.scalar_one_or_none()
        if admin:
            prefix = (admin.key_hash or "")[:16]
            print(f"  key_hash 前缀: {prefix}... (共 {len(admin.key_hash or '')} 字符)")
            print(f"  is_active={admin.is_active}")
        else:
            print("  未找到 admin 用户")

    # If user has raw key in env, show its hash
    raw = os.environ.get("TEAM_MEMORY_API_KEY", "")
    if raw:
        h = hash_key(raw)
        print()
        print(f"TEAM_MEMORY_API_KEY 的哈希: {h}")
        print(f"  与目标匹配: {h.startswith(target_prefix) or target_prefix in h}")


if __name__ == "__main__":
    asyncio.run(main())
