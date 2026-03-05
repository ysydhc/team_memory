"""Add pinned and last_decay_date for quality scoring system.

Reset quality_score to 100 (new scoring scale 0-300).
"""

from alembic import op
import sqlalchemy as sa

revision = "o1p2q3r4s5"
down_revision = "n0o1p2q3r4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("experiences", sa.Column("pinned", sa.Boolean(), server_default="false", nullable=False))
    op.add_column("experiences", sa.Column("last_decay_date", sa.Date(), nullable=True))
    # Reset quality_score from old 0-5 scale to new 0-300 scale (initial=100)
    op.execute("UPDATE experiences SET quality_score = 100")
    op.execute("ALTER TABLE experiences ALTER COLUMN quality_score SET DEFAULT 100")


def downgrade() -> None:
    op.drop_column("experiences", "last_decay_date")
    op.drop_column("experiences", "pinned")
    op.execute("UPDATE experiences SET quality_score = 0")
    op.execute("ALTER TABLE experiences ALTER COLUMN quality_score SET DEFAULT 0")
