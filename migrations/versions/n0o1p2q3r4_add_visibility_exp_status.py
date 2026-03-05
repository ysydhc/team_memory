"""Add visibility and exp_status fields for new status model v2.

Revision ID: n0o1p2q3r4
Revises: m9n0o1p2q3
Create Date: 2026-02-28
"""

from alembic import op
import sqlalchemy as sa

revision = "n0o1p2q3r4"
down_revision = ("m9n0o1p2q3f", "ae259d6f86a2")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "experiences",
        sa.Column("visibility", sa.String(20), server_default="project", nullable=False),
    )
    op.add_column(
        "experiences",
        sa.Column("exp_status", sa.String(20), server_default="draft", nullable=False),
    )

    # Data migration: map old fields to new fields
    op.execute("""
        UPDATE experiences SET
            visibility = CASE
                WHEN scope = 'global' THEN 'global'
                WHEN scope = 'personal' THEN 'private'
                ELSE 'project'
            END,
            exp_status = CASE
                WHEN publish_status = 'draft' THEN 'draft'
                WHEN publish_status = 'personal' THEN 'published'
                WHEN publish_status = 'pending_team' THEN 'review'
                WHEN publish_status = 'published' THEN 'published'
                WHEN publish_status = 'rejected' THEN 'rejected'
                ELSE 'draft'
            END
    """)

    # For publish_status=personal, also set visibility=private
    op.execute("""
        UPDATE experiences SET visibility = 'private'
        WHERE publish_status = 'personal'
    """)


def downgrade() -> None:
    op.drop_column("experiences", "exp_status")
    op.drop_column("experiences", "visibility")
