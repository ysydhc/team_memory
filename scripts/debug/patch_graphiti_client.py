filepath = "src/team_memory/services/graph/graphiti_client.py"

with open(filepath, "r") as f:
    content = f.read()

old_str = """        edges_to_process = []
        if hasattr(results, "edges"):
            edges_to_process = results.edges
        elif isinstance(results, list):
            # In some versions, search returns a list of edges directly
            edges_to_process = results

        for edge in edges_to_process:
            if hasattr(edge, "source_node") and hasattr(edge, "target_node"):
                edges_found.append(
                    {
                        "source": edge.source_node.name,
                        "target": edge.target_node.name,
                        "relation": edge.name,
                        "fact": edge.fact,
                    }
                )
                summary_lines.append(
                    f"- {edge.source_node.name} {edge.name} {edge.target_node.name} ({edge.fact})"
                )"""

new_str = """        edges_to_process = []
        if hasattr(results, "edges"):
            edges_to_process = results.edges
        elif isinstance(results, list):
            # In some versions, search returns a list of edges directly
            edges_to_process = results

        for edge in edges_to_process:
            if hasattr(edge, "source_node") and hasattr(edge, "target_node"):
                edges_found.append(
                    {
                        "source": edge.source_node.name,
                        "target": edge.target_node.name,
                        "relation": edge.name,
                        "fact": edge.fact,
                    }
                )
                summary_lines.append(
                    f"- {edge.source_node.name} {edge.name} {edge.target_node.name} ({edge.fact})"
                )
            elif hasattr(edge, "source_node_uuid") and hasattr(edge, "target_node_uuid"):
                # We only have UUIDs, need to fetch nodes from Kuzu
                try:
                    driver = self._client.driver
                    query_str = \"\"\"
                    MATCH (n:Entity {uuid: $src_uuid})
                    MATCH (m:Entity {uuid: $dst_uuid})
                    RETURN n.name as src, m.name as dst
                    \"\"\"
                    async with self._db_lock:
                        records, _, _ = await driver.execute_query(
                            query_str, src_uuid=edge.source_node_uuid, dst_uuid=edge.target_node_uuid
                        )
                    if records:
                        src_name = records[0].get("src")
                        dst_name = records[0].get("dst")
                        edges_found.append(
                            {
                                "source": src_name,
                                "target": dst_name,
                                "relation": edge.name,
                                "fact": edge.fact,
                            }
                        )
                        summary_lines.append(
                            f"- {src_name} {edge.name} {dst_name} ({edge.fact})"
                        )
                except Exception as e:
                    logger.error(f"Failed to fetch nodes for edge {edge.uuid}: {e}")"""

if old_str in content:
    content = content.replace(old_str, new_str)
    with open(filepath, "w") as f:
        f.write(content)
    print("Patched graphiti_client.py")
else:
    print("Already patched or string not found")
