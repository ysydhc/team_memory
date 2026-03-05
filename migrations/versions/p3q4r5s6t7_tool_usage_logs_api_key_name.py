"""P3-7: Add api_key_name to tool_usage_logs for per-key usage tracking.

Revision ID: p3q4r5s6t7
Revises: o1p2q3r4s5
Create Date: 2026-03-01

"""
from alembic import op
import sqlalchemy as sa

revision = "p3q4r5s6t7"
down_revision = "o1p2q3r4s5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tool_usage_logs",
        sa.Column("api_key_name", sa.String(100), nullable=True),
    )
    op.create_index(
        "ix_tool_usage_logs_api_key_name",
        "tool_usage_logs",
        ["api_key_name"],
    )


def downgrade() -> None:
    op.drop_index("ix_tool_usage_logs_api_key_name", table_name="tool_usage_logs")
    op.drop_column("tool_usage_logs", "api_key_name")
