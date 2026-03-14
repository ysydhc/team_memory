"""experience_file_locations

Revision ID: 6ab06751f40e
Revises: w0x1y2z3a4
Create Date: 2026-03-14 22:19:24.892440

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "6ab06751f40e"
down_revision: Union[str, Sequence[str], None] = "w0x1y2z3a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experience_file_locations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experience_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("path", sa.String(1000), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("content_fingerprint", sa.String(128), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("file_mtime_at_bind", sa.Float(), nullable=True),
        sa.Column("file_content_hash_at_bind", sa.String(64), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["experience_id"],
            ["experiences.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_experience_file_locations_path",
        "experience_file_locations",
        ["path"],
        unique=False,
    )
    op.create_index(
        "ix_experience_file_locations_content_fingerprint",
        "experience_file_locations",
        ["content_fingerprint"],
        unique=False,
    )
    op.create_index(
        "ix_exp_file_locations_exp_id_path",
        "experience_file_locations",
        ["experience_id", "path"],
        unique=False,
    )
    op.create_index(
        "ix_exp_file_locations_path_fingerprint",
        "experience_file_locations",
        ["path", "content_fingerprint"],
        unique=False,
    )


def downgrade() -> None:
    """Drop experience_file_locations table and indexes.

    WARNING: Downgrade deletes all data in experience_file_locations.
    """
    op.drop_index(
        "ix_exp_file_locations_path_fingerprint",
        table_name="experience_file_locations",
    )
    op.drop_index(
        "ix_exp_file_locations_exp_id_path",
        table_name="experience_file_locations",
    )
    op.drop_index(
        "ix_experience_file_locations_content_fingerprint",
        table_name="experience_file_locations",
    )
    op.drop_index(
        "ix_experience_file_locations_path",
        table_name="experience_file_locations",
    )
    op.drop_table("experience_file_locations")
