filepath = ".venv/lib/python3.13/site-packages/graphiti_core/graph_queries.py"

with open(filepath, "r") as f:
    content = f.read()

old_str = "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact2', ['name', 'fact']);"
new_str = "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);"

if old_str in content:
    content = content.replace(old_str, new_str)
    with open(filepath, "w") as f:
        f.write(content)
    print("Reverted graph_queries.py")
else:
    print("Already reverted or string not found")
