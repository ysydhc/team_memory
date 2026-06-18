filepath = ".venv/lib/python3.13/site-packages/graphiti_core/graph_queries.py"

with open(filepath, "r") as f:
    content = f.read()

old_str1 = "'edge_name_and_fact': 'RELATES_TO',"
new_str1 = "'edge_name_and_fact2': 'RELATES_TO',"

old_str2 = "'edge_name_and_fact': 'RelatesToNode_',"
new_str2 = "'edge_name_and_fact2': 'RelatesToNode_',"

if old_str1 in content:
    content = content.replace(old_str1, new_str1)
if old_str2 in content:
    content = content.replace(old_str2, new_str2)

with open(filepath, "w") as f:
    f.write(content)
print("Patched graph_queries.py")
