"""Add source_path to archive_attachments for local file mapping.

Revision ID: 007_attachment_source_path
Revises: 006_archive_knowledge
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "007_attachment_source_path"
down_revision = "006_archive_knowledge"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "archive_attachments",
        sa.Column("source_path", sa.String(1000), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("archive_attachments", "source_path")
