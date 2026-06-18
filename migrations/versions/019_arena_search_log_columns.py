"""Add relevance_score and helpfulness_score to search_logs.

- relevance_score FLOAT nullable — measures how relevant results were to the query
- helpfulness_score FLOAT nullable — measures how helpful results were to the user

Revision ID: 019_arena_search_log_columns
Revises: 0190de072338
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "019_arena_search_log_columns"
down_revision = "0190de072338"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "search_logs",
        sa.Column("relevance_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "search_logs",
        sa.Column("helpfulness_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("search_logs", "helpfulness_score")
    op.drop_column("search_logs", "relevance_score")
