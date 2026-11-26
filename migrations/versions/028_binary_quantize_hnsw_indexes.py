"""Add pgvector binary-quantized HNSW expression indexes for 4096-dim vectors.

Revision ID: 028_binary_quantize_hnsw
Revises: 027_fts_trigger_preserve

pgvector HNSW on float32 ``vector`` is capped at 2000 dimensions, whereas
``binary_quantize(embedding)::bit(N)`` with ``bit_hamming_ops`` supports up to
64,000 dimensions (pgvector >= 0.7.0, iterative_scan >= 0.8.0).

This migration inspects the live ``embedding`` column dimension on
``experiences``, ``archives``, and ``personal_memories`` so it safely creates
partial HNSW expression indexes matching the actual column dimension (4096 in
production) and skips any table whose ``embedding`` column is missing or
un-dimensioned.
"""

from __future__ import annotations

from alembic import op

revision = "028_binary_quantize_hnsw"
down_revision = "027_fts_trigger_preserve"
branch_labels = None
depends_on = None

INDEX_SPECS: tuple[tuple[str, str], ...] = (
    ("experiences", "ix_experiences_embedding_bq_hnsw"),
    ("archives", "ix_archives_embedding_bq_hnsw"),
    ("personal_memories", "ix_personal_memories_embedding_bq_hnsw"),
)


def upgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            tbl TEXT;
            idx_name TEXT;
            col_type TEXT;
            dim_match TEXT[];
            vec_dim INT;
        BEGIN
            FOR tbl, idx_name IN VALUES
                ('experiences', 'ix_experiences_embedding_bq_hnsw'),
                ('archives', 'ix_archives_embedding_bq_hnsw'),
                ('personal_memories', 'ix_personal_memories_embedding_bq_hnsw')
            LOOP
                SELECT format_type(a.atttypid, a.atttypmod)
                  INTO col_type
                  FROM pg_attribute a
                  JOIN pg_class c ON c.oid = a.attrelid
                  JOIN pg_namespace n ON n.oid = c.relnamespace
                 WHERE n.nspname = 'public'
                   AND c.relname = tbl
                   AND a.attname = 'embedding'
                   AND a.attnum > 0
                   AND NOT a.attisdropped;

                IF col_type IS NULL THEN
                    RAISE NOTICE 'Skipping %: embedding column not found', tbl;
                    CONTINUE;
                END IF;

                dim_match := regexp_match(col_type, '^vector\\((\\d+)\\)$');
                IF dim_match IS NULL THEN
                    RAISE NOTICE 'Skipping %: unsupported embedding type %', tbl, col_type;
                    CONTINUE;
                END IF;

                vec_dim := dim_match[1]::INT;
                IF vec_dim <= 0 OR vec_dim > 64000 THEN
                    RAISE NOTICE 'Skipping %: vector dim % out of HNSW bit_hamming_ops range', tbl, vec_dim;
                    CONTINUE;
                END IF;

                EXECUTE format(
                    'CREATE INDEX IF NOT EXISTS %I ON public.%I '
                    'USING hnsw ((binary_quantize(embedding)::bit(%s)) bit_hamming_ops) '
                    'WITH (m = 16, ef_construction = 64) '
                    'WHERE embedding IS NOT NULL',
                    idx_name,
                    tbl,
                    vec_dim
                );
            END LOOP;
        END
        $$;
        """
    )


def downgrade() -> None:
    for _tbl, idx_name in reversed(INDEX_SPECS):
        op.execute(f"DROP INDEX IF EXISTS public.{idx_name}")
