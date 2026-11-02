"""Add search_log stage telemetry columns and bi-temporal / feedback columns to experiences and relationships.

Revision ID: 022_telemetry_and_bitemporal
Revises: 021_merge_project_name_variants
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "022_telemetry_and_bitemporal"
down_revision = "021_merge_project_name_variants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. search_logs telemetry columns
    op.add_column(
        "search_logs",
        sa.Column("stage_metrics", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "search_logs",
        sa.Column("stage_counts", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "search_logs",
        sa.Column("returned_tokens", sa.Integer(), nullable=True),
    )

    # 2. experiences bi-temporal & lifecycle columns
    op.add_column(
        "experiences",
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
    )
    op.add_column(
        "experiences",
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_experiences_valid_until",
        "experiences",
        ["valid_until"],
    )
    op.add_column(
        "experiences",
        sa.Column(
            "superseded_by_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "experiences",
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column(
            "mutation_history",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=True,
        ),
    )
    op.add_column(
        "experiences",
        sa.Column(
            "negative_feedback_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )

    # 3. relationships temporal validity columns
    op.add_column(
        "relationships",
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=True,
        ),
    )
    op.add_column(
        "relationships",
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("relationships", "valid_until")
    op.drop_column("relationships", "valid_from")

    op.drop_column("experiences", "negative_feedback_count")
    op.drop_column("experiences", "mutation_history")
    op.drop_column("experiences", "last_verified_at")
    op.drop_column("experiences", "superseded_by_id")
    op.drop_index("ix_experiences_valid_until", table_name="experiences")
    op.drop_column("experiences", "valid_until")
    op.drop_column("experiences", "valid_from")

    op.drop_column("search_logs", "returned_tokens")
    op.drop_column("search_logs", "stage_counts")
    op.drop_column("search_logs", "stage_metrics")
