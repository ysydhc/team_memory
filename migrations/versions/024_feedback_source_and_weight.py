"""Add source, weight, search_log_id to experience_feedbacks and implicit counters to experiences.

Revision ID: 024_feedback_source_and_weight
Revises: 023_multitenant_rls
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "024_feedback_source_and_weight"
down_revision = "023_multitenant_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. experience_feedbacks: distinguish explicit vs implicit feedback + idempotency key
    op.add_column(
        "experience_feedbacks",
        sa.Column(
            "source",
            sa.String(32),
            nullable=False,
            server_default="explicit_user",
        ),
    )
    op.add_column(
        "experience_feedbacks",
        sa.Column(
            "weight",
            sa.Float(),
            nullable=False,
            server_default="1.0",
        ),
    )
    op.add_column(
        "experience_feedbacks",
        sa.Column(
            "search_log_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_index(
        "uq_exp_feedback_tenant_exp_log_source",
        "experience_feedbacks",
        ["tenant_id", "experience_id", "search_log_id", "source"],
        unique=True,
        postgresql_where=sa.text("search_log_id IS NOT NULL"),
    )
    op.create_index(
        "ix_exp_feedback_tenant_source_created",
        "experience_feedbacks",
        ["tenant_id", "source", "created_at"],
    )

    # 2. experiences: implicit positive / negative counters (weighted 0.5x in scoring.py)
    op.add_column(
        "experiences",
        sa.Column(
            "implicit_positive_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "experiences",
        sa.Column(
            "implicit_negative_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("experiences", "implicit_negative_count")
    op.drop_column("experiences", "implicit_positive_count")
    op.drop_index("ix_exp_feedback_tenant_source_created", table_name="experience_feedbacks")
    op.drop_index("uq_exp_feedback_tenant_exp_log_source", table_name="experience_feedbacks")
    op.drop_column("experience_feedbacks", "search_log_id")
    op.drop_column("experience_feedbacks", "weight")
    op.drop_column("experience_feedbacks", "source")
