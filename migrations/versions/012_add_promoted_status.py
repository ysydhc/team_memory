"""Add promoted status to exp_status CHECK constraint.

Revision ID: 012_add_promoted_status
Revises: 011_add_quality_score_fields
Create Date: 2026-04-23

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "012_add_promoted_status"
down_revision = "011_add_quality_score_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add CHECK constraint for exp_status allowing draft, published, promoted."""
    # Drop existing constraint if it exists (may not exist in older schemas)
    op.execute(
        "ALTER TABLE experiences DROP CONSTRAINT IF EXISTS ck_experiences_exp_status"
    )
    # Add CHECK constraint for valid exp_status values
    op.execute(
        "ALTER TABLE experiences ADD CONSTRAINT ck_experiences_exp_status "
        "CHECK (exp_status IN ('draft', 'published', 'promoted'))"
    )


def downgrade() -> None:
    """Remove the promoted status CHECK constraint."""
    op.execute(
        "ALTER TABLE experiences DROP CONSTRAINT IF EXISTS ck_experiences_exp_status"
    )
    # Re-add constraint without promoted
    op.execute(
        "ALTER TABLE experiences ADD CONSTRAINT ck_experiences_exp_status "
        "CHECK (exp_status IN ('draft', 'published'))"
    )
