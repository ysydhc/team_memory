"""Alembic environment configuration for team_memory.

Supports async PostgreSQL via asyncpg. Reads database URL from
alembic.ini or TEAM_MEMORY_DATABASE__URL environment variable.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import all models so Alembic can detect them for autogenerate
from team_memory.storage.models import Base  # noqa: F401

# Alembic Config object
config = context.config

# Setup Python logging from config file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate
target_metadata = Base.metadata

# Allow overriding database URL via environment variable
env_url = os.environ.get("TEAM_MEMORY_DATABASE__URL")
if env_url:
    config.set_main_option("sqlalchemy.url", env_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations with a given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        # Acquire advisory lock to prevent concurrent migrations
        await connection.execute(text("SELECT pg_advisory_lock(1)"))
        try:
            await connection.run_sync(do_run_migrations)
        finally:
            await connection.execute(text("SELECT pg_advisory_unlock(1)"))
        await connection.commit()

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
