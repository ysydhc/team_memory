"""Add password_hash to api_keys and make key_hash nullable for registration flow.

Revision ID: i5j6k7l8m9n
Revises: h4i5j6k7l8m
Create Date: 2026-02-25
"""

import sqlalchemy as sa
from alembic import op

revision = "i5j6k7l8m9n"
down_revision = "h4i5j6k7l8m"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("password_hash", sa.String(256), nullable=True))
    op.alter_column("api_keys", "key_hash", existing_type=sa.String(256), nullable=True)
    op.drop_constraint("api_keys_key_hash_key", "api_keys", type_="unique")
    op.create_unique_constraint("uq_api_keys_key_hash", "api_keys", ["key_hash"])
    op.create_unique_constraint("uq_api_keys_user_name", "api_keys", ["user_name"])


def downgrade() -> None:
    op.drop_constraint("uq_api_keys_user_name", "api_keys", type_="unique")
    op.drop_constraint("uq_api_keys_key_hash", "api_keys", type_="unique")
    op.create_unique_constraint("api_keys_key_hash_key", "api_keys", ["key_hash"])
    op.alter_column("api_keys", "key_hash", existing_type=sa.String(256), nullable=False)
    op.drop_column("api_keys", "password_hash")
