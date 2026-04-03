"""Add raw_conversation to archives for user behavior pattern extraction.

Revision ID: 008_archive_raw_conversation
Revises: 007_attachment_source_path
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "008_archive_raw_conversation"
down_revision = "007_attachment_source_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("archives", sa.Column("raw_conversation", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("archives", "raw_conversation")
