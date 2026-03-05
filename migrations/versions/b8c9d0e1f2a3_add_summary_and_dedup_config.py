"""Add summary column to experiences table (P0-4 Memory Compaction).

Supports LLM-generated summaries for long experiences to reduce
context window consumption during search result delivery.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-02-14
"""

import sqlalchemy as sa
from alembic import op

revision = "b8c9d0e1f2a3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add summary column
    op.add_column(
        "experiences",
        sa.Column("summary", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("experiences", "summary")
