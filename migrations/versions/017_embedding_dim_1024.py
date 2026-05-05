"""Change embedding dimension from 768 to 1024 (qwen3-embedding:0.6b).

- experiences.embedding: vector(768) → vector(1024)
- archives.embedding: vector(768) → vector(1024)
- personal_memories.embedding: vector(768) → vector(1024)
- Drop and recreate all embedding indexes

Revision ID: 017_embedding_dim_1024
Revises: 016_response_buffer
"""

from alembic import op

revision = "017_embedding_dim_1024"
down_revision = "016_response_buffer"
branch_labels = None
depends_on = None

# Tables and their embedding indexes
TABLES = ["experiences", "archives", "personal_memories"]

INDEXES = {
    "experiences": [
        ("idx_exp_embedding", "ivfflat", "WITH (lists='100')"),
        ("ix_experiences_embedding", "hnsw", "WITH (m='16', ef_construction=64)"),
    ],
    "archives": [
        ("idx_archives_embedding", "ivfflat", "WITH (lists='100')"),
        ("ix_archives_embedding", "hnsw", "WITH (m='16', ef_construction=64)"),
    ],
    "personal_memories": [
        ("ix_personal_memories_embedding", "hnsw", "WITH (m='16', ef_construction=64)"),
    ],
}


def upgrade() -> None:
    # 1. Clear existing embeddings (must happen before ALTER TYPE)
    for table in TABLES:
        op.execute(f"UPDATE {table} SET embedding = NULL")

    # 2. Drop all embedding indexes
    for table, indexes in INDEXES.items():
        for idx_name, _, _ in indexes:
            op.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # 3. Alter column type for each table
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector(1024)")

    # 4. Recreate indexes
    for table, indexes in INDEXES.items():
        for idx_name, method, params in indexes:
            op.execute(
                f"CREATE INDEX {idx_name} ON {table} USING {method} "
                f"(embedding vector_cosine_ops) {params}"
            )


def downgrade() -> None:
    # Clear embeddings before altering type
    for table in TABLES:
        op.execute(f"UPDATE {table} SET embedding = NULL")

    # Drop indexes
    for table, indexes in INDEXES.items():
        for idx_name, _, _ in indexes:
            op.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Revert column type
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector(768)")

    # Recreate indexes
    for table, indexes in INDEXES.items():
        for idx_name, method, params in indexes:
            op.execute(
                f"CREATE INDEX {idx_name} ON {table} USING {method} "
                f"(embedding vector_cosine_ops) {params}"
            )
