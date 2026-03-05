"""Add lifecycle features: last_used_at column + experience_versions table.

Revision ID: f5a6b7c8d9e0
Revises: e4f5a6b7c8d9
Create Date: 2026-02-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "f5a6b7c8d9e0"
down_revision = "e4f5a6b7c8d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # B1: Add last_used_at column to experiences (defaults to created_at via server)
    op.add_column(
        "experiences",
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # Backfill last_used_at from updated_at for existing rows
    op.execute(
        "UPDATE experiences SET last_used_at = COALESCE(updated_at, created_at) WHERE last_used_at IS NULL"
    )

    # Index for efficient stale experience scanning
    op.create_index(
        "ix_experiences_last_used_at",
        "experiences",
        ["last_used_at"],
        unique=False,
    )

    # B3: Create experience_versions table for version history
    op.create_table(
        "experience_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "experience_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("version_number", sa.Integer, nullable=False),
        sa.Column("snapshot", JSONB, nullable=False),
        sa.Column("changed_by", sa.String(100), nullable=False),
        sa.Column("change_summary", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("experience_versions")
    op.drop_index("ix_experiences_last_used_at", table_name="experiences")
    op.drop_column("experiences", "last_used_at")
