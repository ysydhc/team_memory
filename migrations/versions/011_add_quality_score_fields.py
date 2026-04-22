"""Add quality score fields to experiences table.

Revision ID: 011_add_quality_score_fields
Revises: 010_indexes_and_constraints
Create Date: 2026-04-13

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "011_add_quality_score_fields"
down_revision = "010_indexes_and_constraints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add quality score fields to experiences table."""
    # Add quality_score column
    op.add_column(
        "experiences",
        sa.Column("quality_score", sa.Float(), nullable=False, server_default="100.0"),
    )

    # Add quality_tier column
    op.add_column(
        "experiences",
        sa.Column("quality_tier", sa.String(20), nullable=False, server_default="'Silver'"),
    )

    # Add last_scored_at column with NOW() as default
    op.add_column(
        "experiences",
        sa.Column(
            "last_scored_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("NOW()"),
        ),
    )

    # Add is_pinned column
    op.add_column(
        "experiences", sa.Column("is_pinned", sa.Boolean(), nullable=False, server_default="false")
    )


def downgrade() -> None:
    """Remove quality score fields from experiences table."""
    op.drop_column("experiences", "is_pinned")
    op.drop_column("experiences", "last_scored_at")
    op.drop_column("experiences", "quality_tier")
    op.drop_column("experiences", "quality_score")
