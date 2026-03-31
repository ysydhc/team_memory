"""Ensure personal_memories exists (repair DBs at 002 without this table).

Revision ID: 003_personal_memories_if_missing
Revises: 002_mvp_cleanup
Create Date: 2026-03-30
"""

from alembic import op

revision = "003_personal_memories_if_missing"
down_revision = "002_mvp_cleanup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS personal_memories (
            id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
            user_id VARCHAR(100) NOT NULL,
            content TEXT NOT NULL,
            scope VARCHAR(20) NOT NULL DEFAULT 'generic',
            context_hint VARCHAR(500),
            embedding vector(768),
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_personal_memories_user_id "
        "ON personal_memories (user_id)"
    )


def downgrade() -> None:
    raise NotImplementedError("Downgrade not supported for repair migration 003")
