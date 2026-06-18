import os

files_to_patch = [
    ".venv/lib/python3.13/site-packages/graphiti_core/graph_queries.py",
    ".venv/lib/python3.13/site-packages/graphiti_core/driver/kuzu/operations/search_ops.py",
    ".venv/lib/python3.13/site-packages/graphiti_core/search/search_utils.py",
]

for filepath in files_to_patch:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()

        if "graph_queries.py" in filepath:
            old_str1 = "'edge_name_and_fact': 'RELATES_TO',"
            new_str1 = "'edge_name_and_fact2': 'RELATES_TO',"
            old_str2 = "'edge_name_and_fact': 'RelatesToNode_',"
            new_str2 = "'edge_name_and_fact2': 'RelatesToNode_',"
            if old_str1 in content:
                content = content.replace(old_str1, new_str1)
            if old_str2 in content:
                content = content.replace(old_str2, new_str2)
        elif "search_ops.py" in filepath:
            old_str = "get_relationships_query('edge_name_and_fact', limit=limit, provider=GraphProvider.KUZU)"
            new_str = "get_relationships_query('edge_name_and_fact2', limit=limit, provider=GraphProvider.KUZU)"
            if old_str in content:
                content = content.replace(old_str, new_str)
        elif "search_utils.py" in filepath:
            old_str = "get_relationships_query('edge_name_and_fact', limit=limit, provider=driver.provider)"
            new_str = "get_relationships_query('edge_name_and_fact2', limit=limit, provider=driver.provider)"
            if old_str in content:
                content = content.replace(old_str, new_str)

        with open(filepath, "w") as f:
            f.write(content)
        print(f"Patched {filepath}")
    else:
        print(f"File not found: {filepath}")
