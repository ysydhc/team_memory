"""Add key_prefix and key_suffix to api_keys for masked display.

Revision ID: q4r5s6t7u8
Revises: 5bdca099b9de
Create Date: 2026-03-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "q4r5s6t7u8"
down_revision: Union[str, Sequence[str], None] = "5bdca099b9de"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("key_prefix", sa.String(4), nullable=True))
    op.add_column("api_keys", sa.Column("key_suffix", sa.String(4), nullable=True))


def downgrade() -> None:
    op.drop_column("api_keys", "key_suffix")
    op.drop_column("api_keys", "key_prefix")
