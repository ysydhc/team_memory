"""add archived to task_groups and delete task endpoint support

Revision ID: l8m9n0o1p2
Revises: k7l8m9n0o1
"""
from alembic import op
import sqlalchemy as sa

revision = "l8m9n0o1p2"
down_revision = "k7l8m9n0o1"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "task_groups",
        sa.Column("archived", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade():
    op.drop_column("task_groups", "archived")
