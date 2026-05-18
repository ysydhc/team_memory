import kuzu
import sys

db = kuzu.Database('.kuzu_db')
conn = kuzu.Connection(db)

indexes = [
    ('Episodic', 'episode_content', ['content', 'source', 'source_description']),
    ('Entity', 'node_name_and_summary', ['name', 'summary']),
    ('Community', 'community_name', ['name']),
    ('RelatesToNode_', 'edge_name_and_fact2', ['name', 'fact'])
]

for table, index_name, cols in indexes:
    try:
        print(f"Dropping FTS index {index_name} on {table}...", flush=True)
        conn.execute(f"CALL DROP_FTS_INDEX('{table}', '{index_name}')")
        print(f"Dropped FTS index {index_name} on {table}", flush=True)
    except Exception as e:
        print(f"Exception dropping {index_name}: {e}", flush=True)
        
    try:
        print(f"Recreating FTS index {index_name} on {table}...", flush=True)
        cols_str = ", ".join([f"'{c}'" for c in cols])
        conn.execute(f"CALL CREATE_FTS_INDEX('{table}', '{index_name}', [{cols_str}])")
        print(f"Recreated FTS index {index_name} on {table}", flush=True)
    except Exception as e:
        print(f"Exception recreating {index_name}: {e}", flush=True)

