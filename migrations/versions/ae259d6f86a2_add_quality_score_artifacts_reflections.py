"""add quality_score artifacts reflections

Revision ID: ae259d6f86a2
Revises: l8m9n0o1p2
Create Date: 2026-02-27 20:07:27.607352

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'ae259d6f86a2'
down_revision: Union[str, Sequence[str], None] = 'l8m9n0o1p2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add quality_score column and artifact/reflection tables."""
    op.add_column(
        'experiences',
        sa.Column('quality_score', sa.Integer(), server_default='0', nullable=False),
    )

    op.create_table(
        'experience_artifacts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('experience_id', sa.UUID(), nullable=False),
        sa.Column('artifact_type', sa.String(length=30), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('context_before', sa.Text(), nullable=True),
        sa.Column('context_after', sa.Text(), nullable=True),
        sa.Column('source_ref', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['experience_id'], ['experiences.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_experience_artifacts_experience_id',
        'experience_artifacts', ['experience_id'],
    )

    op.create_table(
        'experience_reflections',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('task_id', sa.UUID(), nullable=True),
        sa.Column('experience_id', sa.UUID(), nullable=True),
        sa.Column('success_points', sa.Text(), nullable=True),
        sa.Column('failure_points', sa.Text(), nullable=True),
        sa.Column('improvements', sa.Text(), nullable=True),
        sa.Column('generalized_strategy', sa.Text(), nullable=True),
        sa.Column('judge_score', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['experience_id'], ['experiences.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['task_id'], ['personal_tasks.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_experience_reflections_experience_id',
        'experience_reflections', ['experience_id'],
    )
    op.create_index(
        'ix_experience_reflections_task_id',
        'experience_reflections', ['task_id'],
    )


def downgrade() -> None:
    """Remove quality_score column and artifact/reflection tables."""
    op.drop_index('ix_experience_reflections_task_id', table_name='experience_reflections')
    op.drop_index('ix_experience_reflections_experience_id', table_name='experience_reflections')
    op.drop_table('experience_reflections')
    op.drop_index('ix_experience_artifacts_experience_id', table_name='experience_artifacts')
    op.drop_table('experience_artifacts')
    op.drop_column('experiences', 'quality_score')
