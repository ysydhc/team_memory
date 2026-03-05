"""add_task_acceptance_criteria_acceptance_met

Revision ID: 5bdca099b9de
Revises: p3q4r5s6t7
Create Date: 2026-03-03 23:09:35.276622

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5bdca099b9de'
down_revision: Union[str, Sequence[str], None] = 'p3q4r5s6t7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "personal_tasks",
        sa.Column("acceptance_criteria", sa.Text(), nullable=True),
    )
    op.add_column(
        "personal_tasks",
        sa.Column("acceptance_met", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("personal_tasks", "acceptance_met")
    op.drop_column("personal_tasks", "acceptance_criteria")
