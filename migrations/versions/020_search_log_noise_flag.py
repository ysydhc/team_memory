"""Add noise classification flags to search_logs.

- is_noise BOOLEAN nullable — True when the query is not a real retrieval
  intent (test command, confirmation, multi-step instruction, chit-chat).
  NULL means "not yet classified" (legacy rows).
- noise_reason VARCHAR(20) nullable — how it was flagged: 'rule' | 'llm'.

Quality/usage stats exclude rows where is_noise IS TRUE so that conversational
noise no longer drags down hit-rate, use-rate, and faithfulness averages.

Revision ID: 020_search_log_noise_flag
Revises: 019_arena_search_log_columns
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "020_search_log_noise_flag"
down_revision = "019_arena_search_log_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "search_logs",
        sa.Column("is_noise", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "search_logs",
        sa.Column("noise_reason", sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("search_logs", "noise_reason")
    op.drop_column("search_logs", "is_noise")
