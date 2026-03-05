"""P4: Add project field for team config isolation.

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-02-09
"""
import sqlalchemy as sa
from alembic import op

revision = "e1f2a3b4c5d6"
down_revision = "d0e1f2a3b4c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # P4-1: Add project field
    op.add_column(
        "experiences",
        sa.Column("project", sa.String(100), nullable=False, server_default="default"),
    )
    op.create_index("ix_experiences_project", "experiences", ["project"])


def downgrade() -> None:
    op.drop_index("ix_experiences_project")
    op.drop_column("experiences", "project")
