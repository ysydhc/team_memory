"""Add parent_id to experiences for parent-child hierarchy.

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2026-02-11
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "e4f5a6b7c8d9"
down_revision = "d3e4f5a6b7c8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add parent_id column: self-referencing FK to experiences.id
    op.add_column(
        "experiences",
        sa.Column(
            "parent_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    # Index for fast lookup of children by parent_id
    op.create_index(
        "ix_experiences_parent_id",
        "experiences",
        ["parent_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_experiences_parent_id", table_name="experiences")
    op.drop_column("experiences", "parent_id")
