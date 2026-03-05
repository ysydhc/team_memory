"""Add custom_installable_contents table for user-edited rules/prompts.

Revision ID: r5s6t7u8v9
Revises: q4r5s6t7u8
Create Date: 2025-03-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "r5s6t7u8v9"
down_revision: Union[str, Sequence[str], None] = "q4r5s6t7u8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "custom_installable_contents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project", sa.String(100), nullable=False),
        sa.Column("item_id", sa.String(200), nullable=False),
        sa.Column("item_type", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("updated_by", sa.String(100), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", "item_id", name="uq_custom_installable_project_item"),
    )
    op.create_index(
        "ix_custom_installable_contents_project",
        "custom_installable_contents",
        ["project"],
        unique=False,
    )
    op.create_index(
        "ix_custom_installable_contents_item_id",
        "custom_installable_contents",
        ["item_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_custom_installable_contents_item_id", "custom_installable_contents")
    op.drop_index("ix_custom_installable_contents_project", "custom_installable_contents")
    op.drop_table("custom_installable_contents")
