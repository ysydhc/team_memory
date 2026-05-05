"""Add response_buffer table + faithfulness_score on search_logs.

- response_buffer: temporary storage for pending faithfulness evaluations
- search_logs: add faithfulness_score (float, nullable)

Revision ID: 016_response_buffer
Revises: 015_evaluation_fields
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON

revision = "016_response_buffer"
down_revision = "015_evaluation_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- response_buffer table ---
    op.create_table(
        "response_buffer",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("agent_response", sa.Text(), nullable=False),
        sa.Column("result_ids", JSON, nullable=True),
        sa.Column("search_log_id", UUID(as_uuid=True), nullable=True),
        sa.Column("evaluated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("faithfulness_score", sa.Float(), nullable=True),
        sa.Column("judge_reasoning", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_response_buffer_evaluated", "response_buffer", ["evaluated"])
    op.create_index("ix_response_buffer_created_at", "response_buffer", ["created_at"])

    # --- search_logs: add faithfulness_score ---
    op.add_column(
        "search_logs",
        sa.Column("faithfulness_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("search_logs", "faithfulness_score")
    op.drop_table("response_buffer")
