"""Add tool_usage_logs table for analytics tracking.

Revision ID: m9n0o1p2q3
Revises: l8m9n0o1p2
Create Date: 2026-02-28 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "m9n0o1p2q3"
down_revision = "l8m9n0o1p2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_usage_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tool_name", sa.String(100), nullable=False, index=True),
        sa.Column("tool_type", sa.String(20), nullable=False, server_default="mcp"),
        sa.Column("user", sa.String(100), nullable=False, server_default="anonymous"),
        sa.Column(
            "project", sa.String(100), nullable=False, server_default="default"
        ),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("session_id", sa.String(100), nullable=True),
        sa.Column("metadata_extra", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            index=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("tool_usage_logs")
