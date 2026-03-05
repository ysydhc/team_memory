"""Experience type system: add experience_type, severity, category,
progress_status, structured_data, git_refs, related_links.
Also make solution nullable for incomplete experiences.

Revision ID: g3h4i5j6k7l8
Revises: f2a3b4c5d6e7
Create Date: 2026-02-15
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "g3h4i5j6k7l8"
down_revision = "f2a3b4c5d6e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. New columns (all nullable or with defaults — non-breaking)
    op.add_column(
        "experiences",
        sa.Column(
            "experience_type",
            sa.String(30),
            nullable=False,
            server_default="general",
        ),
    )
    op.add_column(
        "experiences",
        sa.Column("severity", sa.String(10), nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column("category", sa.String(50), nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column("progress_status", sa.String(30), nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column("structured_data", JSONB, nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column("git_refs", JSONB, nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column("related_links", JSONB, nullable=True),
    )

    # 2. Make solution nullable (allow incomplete experiences)
    op.alter_column(
        "experiences",
        "solution",
        existing_type=sa.Text(),
        nullable=True,
    )

    # 3. Indexes for efficient filtering
    op.create_index("idx_exp_type", "experiences", ["experience_type"])
    op.create_index(
        "idx_exp_severity",
        "experiences",
        ["severity"],
        postgresql_where=sa.text("severity IS NOT NULL"),
    )
    op.create_index(
        "idx_exp_category",
        "experiences",
        ["category"],
        postgresql_where=sa.text("category IS NOT NULL"),
    )
    op.create_index(
        "idx_exp_progress",
        "experiences",
        ["progress_status"],
        postgresql_where=sa.text("progress_status IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_exp_progress", table_name="experiences")
    op.drop_index("idx_exp_category", table_name="experiences")
    op.drop_index("idx_exp_severity", table_name="experiences")
    op.drop_index("idx_exp_type", table_name="experiences")

    op.alter_column(
        "experiences",
        "solution",
        existing_type=sa.Text(),
        nullable=False,
        server_default="",
    )

    op.drop_column("experiences", "related_links")
    op.drop_column("experiences", "git_refs")
    op.drop_column("experiences", "structured_data")
    op.drop_column("experiences", "progress_status")
    op.drop_column("experiences", "category")
    op.drop_column("experiences", "severity")
    op.drop_column("experiences", "experience_type")
