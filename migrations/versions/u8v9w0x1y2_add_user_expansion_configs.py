"""Add user_expansion_configs table for per-user query expansion.

Revision ID: u8v9w0x1y2
Revises: t7u8v9w0x1
Create Date: 2026-03-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "u8v9w0x1y2"
down_revision: Union[str, Sequence[str], None] = "t7u8v9w0x1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_expansion_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("tag_synonyms", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("user_id", name="uq_user_expansion_configs_user_id"),
    )
    op.create_index(
        "ix_user_expansion_configs_user_id",
        "user_expansion_configs",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_user_expansion_configs_user_id", table_name="user_expansion_configs")
    op.drop_table("user_expansion_configs")
