"""Change feedback from boolean helpful to integer rating (1-5).

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-02-10
"""

import sqlalchemy as sa
from alembic import op

revision = "d3e4f5a6b7c8"
down_revision = "c2d3e4f5a6b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add the new rating column (1-5)
    op.add_column(
        "experience_feedbacks",
        sa.Column("rating", sa.Integer(), nullable=True),
    )

    # Migrate data: helpful=True -> rating=5, helpful=False -> rating=1
    op.execute(
        "UPDATE experience_feedbacks SET rating = CASE WHEN helpful = true THEN 5 ELSE 1 END"
    )

    # Make rating NOT NULL after migration
    op.alter_column("experience_feedbacks", "rating", nullable=False)

    # Drop the old helpful column
    op.drop_column("experience_feedbacks", "helpful")

    # Recalculate avg_rating on experiences from the new rating column
    op.execute("""
        UPDATE experiences e
        SET avg_rating = COALESCE(
            (SELECT AVG(f.rating::float) FROM experience_feedbacks f WHERE f.experience_id = e.id),
            0.0
        )
    """)


def downgrade() -> None:
    # Re-add helpful column
    op.add_column(
        "experience_feedbacks",
        sa.Column("helpful", sa.Boolean(), nullable=True),
    )

    # Migrate back: rating >= 3 -> helpful=True, else False
    op.execute(
        "UPDATE experience_feedbacks SET helpful = CASE WHEN rating >= 3 THEN true ELSE false END"
    )

    op.alter_column("experience_feedbacks", "helpful", nullable=False)

    # Drop rating column
    op.drop_column("experience_feedbacks", "rating")

    # Recalculate avg_rating from helpful boolean
    op.execute("""
        UPDATE experiences e
        SET avg_rating = COALESCE(
            (SELECT AVG(CASE WHEN f.helpful THEN 1.0 ELSE 0.0 END) FROM experience_feedbacks f WHERE f.experience_id = e.id),
            0.0
        )
    """)
