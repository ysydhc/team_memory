"""Add vector indexes, column indexes, unique constraints, and check constraints.

Revision ID: 010_indexes_and_constraints
Revises: 009_background_tasks
"""

from __future__ import annotations

from alembic import op

revision = "010_indexes_and_constraints"
down_revision = "009_background_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # D1: HNSW vector indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_experiences_embedding "
        "ON experiences USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_archives_embedding "
        "ON archives USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_personal_memories_embedding "
        "ON personal_memories USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    # D5: Unique constraint for archive upsert
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_archives_title_project ON archives (title, project)"
    )

    # D6-D9: Column indexes
    op.create_index("ix_experiences_created_by", "experiences", ["created_by"])
    op.create_index("ix_archives_project", "archives", ["project"])
    op.create_index("ix_archives_status", "archives", ["status"])

    # D10: BackgroundTask.id server_default
    op.execute("ALTER TABLE background_tasks ALTER COLUMN id SET DEFAULT gen_random_uuid()")

    # D13: Partial index for soft delete
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_experiences_active "
        "ON experiences(id) WHERE is_deleted = false"
    )

    # D15: CHECK constraints on feedback
    op.execute(
        "ALTER TABLE experience_feedbacks "
        "ADD CONSTRAINT ck_feedback_rating CHECK (rating >= 1 AND rating <= 5)"
    )
    op.execute(
        "ALTER TABLE experience_feedbacks "
        "ADD CONSTRAINT ck_feedback_fitness "
        "CHECK (fitness_score IS NULL OR (fitness_score >= 1 AND fitness_score <= 5))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE experience_feedbacks DROP CONSTRAINT IF EXISTS ck_feedback_fitness")
    op.execute("ALTER TABLE experience_feedbacks DROP CONSTRAINT IF EXISTS ck_feedback_rating")
    op.execute("DROP INDEX IF EXISTS ix_experiences_active")
    op.execute("ALTER TABLE background_tasks ALTER COLUMN id DROP DEFAULT")
    op.drop_index("ix_archives_status")
    op.drop_index("ix_archives_project")
    op.drop_index("ix_experiences_created_by")
    op.execute("DROP INDEX IF EXISTS uq_archives_title_project")
    op.execute("DROP INDEX IF EXISTS ix_personal_memories_embedding")
    op.execute("DROP INDEX IF EXISTS ix_archives_embedding")
    op.execute("DROP INDEX IF EXISTS ix_experiences_embedding")
