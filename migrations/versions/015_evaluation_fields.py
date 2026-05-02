"""Add evaluation fields: judgment_source, recall_count, used_count.

- search_logs: add judgment_source (text, nullable)
- experiences: rename use_count → recall_count, add used_count

Revision ID: 015_evaluation_fields
Revises: 014_add_entity_graph
"""

from alembic import op
import sqlalchemy as sa

revision = "015_evaluation_fields"
down_revision = "014_add_entity_graph"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- search_logs: add judgment_source ---
    op.add_column(
        "search_logs",
        sa.Column("judgment_source", sa.String(20), nullable=True),
    )

    # --- experiences: rename use_count → recall_count ---
    op.alter_column("experiences", "use_count", new_column_name="recall_count")

    # --- experiences: add used_count ---
    op.add_column(
        "experiences",
        sa.Column("used_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    # --- experiences: drop used_count ---
    op.drop_column("experiences", "used_count")

    # --- experiences: rename recall_count → use_count ---
    op.alter_column("experiences", "recall_count", new_column_name="use_count")

    # --- search_logs: drop judgment_source ---
    op.drop_column("search_logs", "judgment_source")
