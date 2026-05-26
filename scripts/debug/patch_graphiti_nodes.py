import os

filepath = ".venv/lib/python3.13/site-packages/graphiti_core/models/nodes/node_db_queries.py"

with open(filepath, "r") as f:
    content = f.read()

# Episodic 1
old_ep1 = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
                RETURN n.uuid AS uuid
            \"\"\"
"""
new_ep1 = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Episodic {uuid: $uuid})
                ON CREATE SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
                ON MATCH SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
                RETURN n.uuid AS uuid
            \"\"\"
"""
content = content.replace(old_ep1, new_ep1)

# Episodic 2
old_ep2 = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
                RETURN n.uuid AS uuid
            \"\"\"
"""
# Note: old_ep1 and old_ep2 are identical, so replace will replace both if we don't limit it, or we just do it once.
# Let's just use replace on the string.

# Entity 1
old_ent1 = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                WITH n
                RETURN n.uuid AS uuid
            \"\"\"
"""
new_ent1 = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Entity {uuid: $uuid})
                ON CREATE SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                ON MATCH SET
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.attributes = $attributes
                WITH n
                RETURN n.uuid AS uuid
            \"\"\"
"""
content = content.replace(old_ent1, new_ent1)

# Entity 2
old_ent2 = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                RETURN n.uuid AS uuid
            \"\"\"
"""
new_ent2 = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Entity {uuid: $uuid})
                ON CREATE SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                ON MATCH SET
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.attributes = $attributes
                RETURN n.uuid AS uuid
            \"\"\"
"""
content = content.replace(old_ent2, new_ent2)

# Community
old_com = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Community {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary
                RETURN n.uuid AS uuid
            \"\"\"
"""
new_com = """
        case GraphProvider.KUZU:
            return \"\"\"
                MERGE (n:Community {uuid: $uuid})
                ON CREATE SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary
                ON MATCH SET
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary
                RETURN n.uuid AS uuid
            \"\"\"
"""
content = content.replace(old_com, new_com)

with open(filepath, "w") as f:
    f.write(content)
print("Patched node_db_queries.py")

