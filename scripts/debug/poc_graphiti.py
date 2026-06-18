#!/usr/bin/env python3
"""Offline PoC script for testing Graphiti ingestion and retrieval.

This script manually extracts a few existing Archive or Experience records
from the database, feeds them into the Graphiti skeleton, and attempts
a subgraph query.
"""

import asyncio
import logging
import os
import sys

# Ensure src/ is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from team_memory.services.graph.graphiti_client import get_graph_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("poc_graphiti")


async def main() -> None:
    """Run the Graphiti PoC."""
    logger.info("Starting Graphiti PoC...")

    # Force the client to use a persistent DB path for debugging
    # We want to see what the daemon has actually extracted
    import os

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".kuzu_db"))
    os.environ["KUZU_DB"] = db_path
    logger.info("Using Kuzu DB at: %s", db_path)

    client = get_graph_client()
    await client.initialize()

    # 1. Simulate feeding high-value records into the graph
    logger.info("--- Phase 1: Ingestion ---")
    mock_records = [
        {
            "id": "exp_chinese_1",
            "type": "experience",
            "content": "Web改动后必须做端到端验收\n修改前端后仅看代码未实测，导致问题反复\n建立验证闭环：重启服务、清缓存、打开页面操作关键路径、检查控制台报错\nreliability team_memory",
        },
    ]

    for record in mock_records:
        await client.ingest_to_graph(
            record_id=record["id"],
            content=record["content"],
            source_type=record["type"],
        )

    # 2. Simulate a dual-track retrieval query
    logger.info("--- Phase 2: Retrieval ---")

    # Try a few different queries to see what got extracted
    queries = ["IntentRouter", "Alice", "Bob", "search_orchestrator"]

    # Also let's print the entire graph to see if ANYTHING was extracted
    logger.info("--- Dumping all nodes in graph ---")
    try:
        # We can access Kuzu directly to see what's in there
        driver = client._client.driver

        # Check Node tables
        tables, _, _ = await driver.execute_query("CALL show_tables() RETURN *")
        logger.info("Tables in DB:")
        for t in tables:
            logger.info("  %s", t)

        nodes, _, _ = await driver.execute_query("MATCH (n) RETURN label(n) as label, n.name as name LIMIT 20")
        logger.info("Total nodes found: %d", len(nodes))
        for node in nodes:
            logger.info("Node: [%s] %s", node.get("label"), node.get("name"))

        edges, _, _ = await driver.execute_query(
            "MATCH (n)-[r]->(m) RETURN label(n) as src_label, n.name as src_name, label(r) as rel_type, label(m) as dst_label, m.name as dst_name LIMIT 20"
        )
        logger.info("Total edges found: %d", len(edges))
        for edge in edges:
            logger.info(
                "Edge: [%s] %s -> %s -> [%s] %s",
                edge.get("src_label"),
                edge.get("src_name"),
                edge.get("rel_type"),
                edge.get("dst_label"),
                edge.get("dst_name"),
            )
    except Exception as e:
        logger.error("Failed to dump graph: %s", e)

    for query in queries:
        logger.info("Query: %s", query)
        result = await client.query_subgraph(query, max_depth=2)
        logger.info("Result Summary: %s", result.get("summary"))
        logger.info("Nodes Found: %s", result.get("nodes_found"))
        logger.info("-" * 40)

    logger.info("PoC complete.")


if __name__ == "__main__":
    asyncio.run(main())
