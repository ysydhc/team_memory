"""P4: Add embedding_dim field for dimension tracking.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-02-09
"""
import sqlalchemy as sa
from alembic import op

revision = "f2a3b4c5d6e7"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "experiences",
        sa.Column("embedding_dim", sa.Integer, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("experiences", "embedding_dim")
