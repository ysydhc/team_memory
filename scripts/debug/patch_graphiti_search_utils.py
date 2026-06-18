filepath = ".venv/lib/python3.13/site-packages/graphiti_core/search/search_utils.py"

with open(filepath, "r") as f:
    content = f.read()

old_str = "get_relationships_query('edge_name_and_fact', limit=limit, provider=driver.provider)"
new_str = "get_relationships_query('edge_name_and_fact2', limit=limit, provider=driver.provider)"

if old_str in content:
    content = content.replace(old_str, new_str)
    with open(filepath, "w") as f:
        f.write(content)
    print("Patched search_utils.py")
else:
    print("Already patched or string not found")
