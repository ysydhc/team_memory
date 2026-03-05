"""P1: Add scope field, experience_links table, and review_history table.

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-02-09
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "c9d0e1f2a3b4"
down_revision = "b8c9d0e1f2a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # P1-2: Add scope field to experiences (personal / team)
    op.add_column(
        "experiences",
        sa.Column("scope", sa.String(20), nullable=False, server_default="team"),
    )

    # P1-3: Create experience_links table for relationship graph
    op.create_table(
        "experience_links",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_id", UUID(as_uuid=True), sa.ForeignKey("experiences.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), sa.ForeignKey("experiences.id", ondelete="CASCADE"), nullable=False),
        sa.Column("link_type", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("created_by", sa.String(100), nullable=True),
        sa.UniqueConstraint("source_id", "target_id", "link_type", name="uq_experience_link"),
    )
    op.create_index("ix_experience_links_source", "experience_links", ["source_id"])
    op.create_index("ix_experience_links_target", "experience_links", ["target_id"])

    # P1-6: Create review_history table
    op.create_table(
        "review_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("experience_id", UUID(as_uuid=True), sa.ForeignKey("experiences.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reviewer", sa.String(100), nullable=False),
        sa.Column("action", sa.String(20), nullable=False),  # approved, rejected
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_review_history_experience", "review_history", ["experience_id"])

    # P1-6: Add reviewed_at timestamp to experiences
    op.add_column(
        "experiences",
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("experiences", "reviewed_at")
    op.drop_table("review_history")
    op.drop_table("experience_links")
    op.drop_column("experiences", "scope")
