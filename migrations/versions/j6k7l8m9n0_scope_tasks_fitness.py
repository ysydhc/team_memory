"""scope extension + task_groups + personal_tasks + fitness_score

Revision ID: j6k7l8m9n0
Revises: i5j6k7l8m9n
Create Date: 2026-02-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY

revision = "j6k7l8m9n0"
down_revision = "i5j6k7l8m9n"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add fitness_score to experience_feedbacks
    op.add_column(
        "experience_feedbacks",
        sa.Column("fitness_score", sa.Integer(), nullable=True),
    )

    # 2. Create task_groups table
    op.create_table(
        "task_groups",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("project", sa.String(100), nullable=False, server_default="default"),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("source_doc", sa.String(500), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("sort_order", sa.Integer(), server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    # 3. Create personal_tasks table
    op.create_table(
        "personal_tasks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "group_id",
            UUID(as_uuid=True),
            sa.ForeignKey("task_groups.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="wait"),
        sa.Column("priority", sa.String(20), nullable=False, server_default="medium"),
        sa.Column("importance", sa.Integer(), server_default="3"),
        sa.Column("project", sa.String(100), nullable=False, server_default="default"),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sort_order", sa.Integer(), server_default="0"),
        sa.Column("labels", ARRAY(sa.String()), nullable=True),
        sa.Column(
            "experience_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "sediment_experience_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    # 4. Add indexes for common queries
    op.create_index(
        "ix_personal_tasks_user_project",
        "personal_tasks",
        ["user_id", "project"],
    )
    op.create_index(
        "ix_personal_tasks_status",
        "personal_tasks",
        ["status"],
    )
    op.create_index(
        "ix_task_groups_source_doc",
        "task_groups",
        ["source_doc"],
    )


def downgrade() -> None:
    op.drop_index("ix_task_groups_source_doc", table_name="task_groups")
    op.drop_index("ix_personal_tasks_status", table_name="personal_tasks")
    op.drop_index("ix_personal_tasks_user_project", table_name="personal_tasks")
    op.drop_table("personal_tasks")
    op.drop_table("task_groups")
    op.drop_column("experience_feedbacks", "fitness_score")
