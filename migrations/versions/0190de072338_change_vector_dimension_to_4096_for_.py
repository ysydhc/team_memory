"""Change vector dimension to 4096 for qwen3-embedding-8b

Revision ID: 0190de072338
Revises: 018_persona_priority
Create Date: 2026-05-15 22:56:25.700415

"""

from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0190de072338"
down_revision: Union[str, Sequence[str], None] = "018_persona_priority"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("archives", "embedding")
    op.add_column(
        "archives",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=4096), nullable=True),
    )
    op.drop_column("experiences", "embedding")
    op.add_column(
        "experiences",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=4096), nullable=True),
    )
    op.drop_column("personal_memories", "embedding")
    op.add_column(
        "personal_memories",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=4096), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("archives", "embedding")
    op.add_column(
        "archives",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=True),
    )
    op.drop_column("experiences", "embedding")
    op.add_column(
        "experiences",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=True),
    )
    op.drop_column("personal_memories", "embedding")
    op.add_column(
        "personal_memories",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=True),
    )
