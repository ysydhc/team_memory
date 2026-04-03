"""Add content_type, value_summary, tags to archives for knowledge classification.

Revision ID: 006_archive_knowledge
Revises: 005_upload_failures
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision = "006_archive_knowledge"
down_revision = "005_upload_failures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "archives",
        sa.Column(
            "content_type",
            sa.String(50),
            nullable=False,
            server_default="session_archive",
        ),
    )
    op.add_column(
        "archives",
        sa.Column("value_summary", sa.String(500), nullable=True),
    )
    op.add_column(
        "archives",
        sa.Column("tags", ARRAY(sa.String), nullable=True),
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_archives_title_project "
        "ON archives (title, project) WHERE title IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_index("uq_archives_title_project", table_name="archives")
    op.drop_column("archives", "tags")
    op.drop_column("archives", "value_summary")
    op.drop_column("archives", "content_type")
