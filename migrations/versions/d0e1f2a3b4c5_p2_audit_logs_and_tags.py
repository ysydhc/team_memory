"""P2: Add audit_logs table and tag_synonyms config.

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-02-09
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "d0e1f2a3b4c5"
down_revision = "c9d0e1f2a3b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # P2-3: Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_name", sa.String(100), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("target_type", sa.String(50), nullable=False),
        sa.Column("target_id", sa.String(100), nullable=True),
        sa.Column("detail", JSONB, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_audit_logs_user", "audit_logs", ["user_name"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    # P3-7: Add extra columns to query_logs
    op.add_column("query_logs", sa.Column("api_key_id", sa.Integer, nullable=True))
    op.add_column("query_logs", sa.Column("endpoint", sa.String(200), nullable=True))
    op.add_column("query_logs", sa.Column("method", sa.String(10), nullable=True))


def downgrade() -> None:
    op.drop_column("query_logs", "method")
    op.drop_column("query_logs", "endpoint")
    op.drop_column("query_logs", "api_key_id")
    op.drop_table("audit_logs")
