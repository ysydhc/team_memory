"""Add archive_upload_failures for upload MVP human remediation.

Revision ID: 005_upload_failures
Revises: 004_profile_kind
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "005_upload_failures"
down_revision = "004_profile_kind"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "archive_upload_failures",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "archive_id",
            UUID(as_uuid=True),
            sa.ForeignKey("archives.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.Column("source", sa.String(20), server_default="web", nullable=False),
        sa.Column("error_code", sa.String(50), nullable=False),
        sa.Column("error_message", sa.String(500), nullable=False),
        sa.Column("client_filename_hint", sa.String(500), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_archive_upload_failures_archive_id",
        "archive_upload_failures",
        ["archive_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_archive_upload_failures_archive_id", table_name="archive_upload_failures")
    op.drop_table("archive_upload_failures")
