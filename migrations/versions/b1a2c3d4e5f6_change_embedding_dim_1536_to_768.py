"""change embedding dimension from 1536 to 768 for Ollama nomic-embed-text

Revision ID: b1a2c3d4e5f6
Revises: 06387edfbe6e
Create Date: 2026-02-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b1a2c3d4e5f6'
down_revision: Union[str, Sequence[str]] = '06387edfbe6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Change embedding column from vector(1536) to vector(768).

    This migration:
    1. Drops the IVFFlat index (requires fixed dimension)
    2. Drops the old column and adds a new one with correct dimension
    3. Recreates the IVFFlat index with the new dimension

    WARNING: All existing embedding data will be lost.
    If you have real data, re-encode all embeddings after running this migration.
    """
    # Drop the IVFFlat index first (it depends on the column type)
    op.execute("DROP INDEX IF EXISTS idx_exp_embedding")

    # Alter the column type from vector(1536) to vector(768)
    op.execute(
        "ALTER TABLE experiences "
        "ALTER COLUMN embedding TYPE vector(768) "
        "USING NULL"  # Reset existing embeddings (dimension mismatch)
    )

    # Recreate the IVFFlat index with the new dimension
    op.execute("""
        CREATE INDEX idx_exp_embedding ON experiences
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)


def downgrade() -> None:
    """Revert embedding column from vector(768) back to vector(1536)."""
    op.execute("DROP INDEX IF EXISTS idx_exp_embedding")

    op.execute(
        "ALTER TABLE experiences "
        "ALTER COLUMN embedding TYPE vector(1536) "
        "USING NULL"
    )

    op.execute("""
        CREATE INDEX idx_exp_embedding ON experiences
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)
