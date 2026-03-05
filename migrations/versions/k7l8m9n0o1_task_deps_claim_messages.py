"""task_dependencies + assignee/claimed_at + task_messages + hierarchy fields

Revision ID: k7l8m9n0o1
Revises: j6k7l8m9n0
Create Date: 2026-02-27
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "k7l8m9n0o1"
down_revision = "j6k7l8m9n0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Phase 2: task_dependencies
    op.create_table(
        "task_dependencies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_task_id", UUID(as_uuid=True), sa.ForeignKey("personal_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_task_id", UUID(as_uuid=True), sa.ForeignKey("personal_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dep_type", sa.String(30), nullable=False),  # blocks, related, discovered_from
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_by", sa.String(100), nullable=True),
    )
    op.create_index("ix_task_deps_source", "task_dependencies", ["source_task_id"])
    op.create_index("ix_task_deps_target", "task_dependencies", ["target_task_id"])

    # Phase 2: assignee + claimed_at on personal_tasks
    op.add_column("personal_tasks", sa.Column("assignee", sa.String(100), nullable=True))
    op.add_column("personal_tasks", sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True))

    # Phase 3: task_messages
    op.create_table(
        "task_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("task_id", UUID(as_uuid=True), sa.ForeignKey("personal_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("thread_id", UUID(as_uuid=True), nullable=True),
        sa.Column("author", sa.String(100), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_task_messages_task", "task_messages", ["task_id"])

    # Phase 4: hierarchy fields on personal_tasks
    op.add_column("personal_tasks", sa.Column("parent_task_id", UUID(as_uuid=True), nullable=True))
    op.add_column("personal_tasks", sa.Column("path", sa.String(200), nullable=True))
    op.add_column("personal_tasks", sa.Column("display_id", sa.String(50), nullable=True))
    op.add_column("personal_tasks", sa.Column("content_hash", sa.String(64), nullable=True))
    op.create_foreign_key(
        "fk_personal_tasks_parent",
        "personal_tasks",
        "personal_tasks",
        ["parent_task_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_personal_tasks_content_hash", "personal_tasks", ["content_hash"])
    op.create_index("ix_personal_tasks_parent", "personal_tasks", ["parent_task_id"])


def downgrade() -> None:
    op.drop_index("ix_personal_tasks_parent")
    op.drop_index("ix_personal_tasks_content_hash")
    op.drop_constraint("fk_personal_tasks_parent", "personal_tasks", type_="foreignkey")
    op.drop_column("personal_tasks", "content_hash")
    op.drop_column("personal_tasks", "display_id")
    op.drop_column("personal_tasks", "path")
    op.drop_column("personal_tasks", "parent_task_id")
    op.drop_table("task_messages")
    op.drop_column("personal_tasks", "claimed_at")
    op.drop_column("personal_tasks", "assignee")
    op.drop_index("ix_task_deps_target")
    op.drop_index("ix_task_deps_source")
    op.drop_table("task_dependencies")
