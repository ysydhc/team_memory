"""Background task queue table.

Revision ID: 009_background_tasks
Revises: 008_archive_raw_conversation
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "009_background_tasks"
down_revision = "008_archive_raw_conversation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "background_tasks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("task_type", sa.String(50), nullable=False, index=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending", index=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
    )


def downgrade() -> None:
    op.drop_table("background_tasks")
