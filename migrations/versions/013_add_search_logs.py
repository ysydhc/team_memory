"""Add search_logs table for evaluation tracking.

Revision ID: 013_add_search_logs
Revises: 012_add_promoted_status
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "013_add_search_logs"
down_revision = "012_add_promoted_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "search_logs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("intent_type", sa.String(20), server_default="unknown"),
        sa.Column("project", sa.String(100), server_default="default"),
        sa.Column("source", sa.String(20), server_default="mcp"),
        sa.Column("result_ids", sa.JSON(), nullable=True),
        sa.Column("was_used", sa.Boolean(), nullable=True),
        sa.Column("agent_response_snippet", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("search_logs")
