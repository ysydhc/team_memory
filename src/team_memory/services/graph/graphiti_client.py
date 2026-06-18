"""Graphiti client for dual-track memory (vector + graph).

This module provides the integration layer with Graphiti for building
and querying the temporal knowledge graph.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from openai import AsyncOpenAI

from team_memory.client_config import get_client_config
from team_memory.config.settings import get_settings

logger = logging.getLogger("team_memory.services.graph")


class GraphitiClient:
    """Client for interacting with the Graphiti temporal graph.

    This is a skeleton implementation for the PoC. It encapsulates
    Graphiti operations so the rest of TeamMemory doesn't leak graph
    specifics.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the Graphiti client.

        Args:
            db_path: Optional path to Kuzu DB. Defaults to memory or KUZU_DB env var.
        """
        import asyncio

        self._db_lock = asyncio.Lock()
        self.db_path = db_path or os.environ.get(
            "KUZU_DB",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".kuzu_db")),
        )
        self._initialized = False
        self._client: Graphiti | None = None

    async def initialize(self) -> None:
        """Initialize the Graphiti connection and schema."""
        if self._initialized:
            return

        async with self._db_lock:
            if self._initialized:
                return

            logger.info("Initializing Graphiti client with Kuzu DB at %s", self.db_path)

            # Configure OpenAI client using TeamMemory's LLM config
            llm_config = get_client_config().llm

            # We set env vars as a fallback for Graphiti's internal defaults
            os.environ["OPENAI_API_KEY"] = llm_config.resolved_api_key or "dummy-key"
            if llm_config.base_url:
                os.environ["OPENAI_BASE_URL"] = llm_config.base_url

            openai_client = AsyncOpenAI(
                base_url=llm_config.base_url,
                api_key=llm_config.resolved_api_key or "dummy-key",
            )

            graphiti_llm_config = GraphitiLLMConfig(
                api_key=llm_config.resolved_api_key or "dummy-key",
                base_url=llm_config.base_url,
                model=llm_config.model,
                small_model=llm_config.model,
            )
            graphiti_llm = OpenAIGenericClient(config=graphiti_llm_config, client=openai_client)

            # Configure Embedder using TeamMemory's embedding config
            embedding_settings = get_settings().embedding

            # Determine base_url, api_key, and model based on provider
            if embedding_settings.provider == "openai":
                emb_base_url = embedding_settings.openai.base_url or "https://api.openai.com/v1"
                emb_api_key = embedding_settings.openai.resolved_api_key or "dummy-key"
                emb_model = embedding_settings.openai.model
            elif embedding_settings.provider == "ollama":
                emb_base_url = f"{embedding_settings.ollama.base_url.rstrip('/')}/v1"
                emb_api_key = "ollama"
                emb_model = embedding_settings.ollama.model
            elif embedding_settings.provider == "generic":
                emb_base_url = embedding_settings.generic.base_url
                emb_api_key = embedding_settings.generic.api_key or "dummy-key"
                emb_model = embedding_settings.generic.model
            else:
                # Fallback for local or unsupported
                emb_base_url = "http://localhost:11434/v1"
                emb_api_key = "dummy-key"
                emb_model = "qwen3-embedding-8b"

            emb_client = AsyncOpenAI(
                api_key=emb_api_key,
                base_url=emb_base_url,
            )
            logger.info(
                "Graphiti Embedder Config: base_url=%s, model=%s, api_key=%s***",
                emb_base_url,
                emb_model,
                emb_api_key[:5],
            )
            embedder_config = OpenAIEmbedderConfig(
                api_key=emb_api_key,
                base_url=emb_base_url,
                embedding_model=emb_model,
            )
            graphiti_embedder = OpenAIEmbedder(config=embedder_config, client=emb_client)

            # Initialize Kuzu driver and Graphiti
            driver = KuzuDriver(db=self.db_path)
            self._client = Graphiti(
                graph_driver=driver,
                llm_client=graphiti_llm,
                embedder=graphiti_embedder,
            )

            # Graphiti requires setup to create indexes/schema
            # Note: build_indices_and_constraints() seems to have issues in kuzu backend
            # We will try to use setup() if it exists, or just catch the error
            try:
                if hasattr(self._client, "setup"):
                    await self._client.setup()
                else:
                    await self._client.build_indices_and_constraints()
            except Exception as e:
                logger.warning("Error during Graphiti schema initialization: %s", e)

            # Workaround for Kuzu driver missing index issue
            # Kuzu syntax for index creation is different or not supported in the same way
            # Let's try standard neo4j syntax just in case, or ignore if it fails
            try:
                # Kuzu FTS index creation syntax
                # await driver.execute_query(
                #     "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact'])"
                # )
                pass
            except Exception as e:
                if "already exists" not in str(e):
                    logger.debug("Failed to create index manually: %s", e)

            self._initialized = True

    async def ingest_to_graph(self, record_id: str, content: str, source_type: str = "experience") -> None:
        """Ingest a high-value record into the graph.

        This method is designed to be called asynchronously via the task_runner.

        Args:
            record_id: The UUID of the Experience or Archive.
            content: The text content to extract entities and relations from.
            source_type: 'experience' or 'archive'.
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            "Graphiti ingest called for %s %s (len: %d)",
            source_type,
            record_id,
            len(content),
        )
        if self._client is None:
            raise RuntimeError("Graphiti client not initialized")

        from datetime import datetime, timezone

        try:
            # Call Graphiti's add_episode ingestion method
            async with self._db_lock:
                await self._client.add_episode(
                    name=f"{source_type}_{record_id}",
                    episode_body=content,
                    source_description=f"Extracted from {source_type}",
                    reference_time=datetime.now(timezone.utc),
                )
            logger.info("Graphiti ingest completed for %s %s", source_type, record_id)
        except Exception as e:
            logger.error("Graphiti ingest failed for %s %s: %s", source_type, record_id, e)
            raise

    async def query_subgraph(self, query: str, max_depth: int = 2, max_nodes: int = 20) -> dict[str, Any]:
        """Query the graph for a bounded subgraph based on the user's query.

        Args:
            query: The user's search query.
            max_depth: Maximum number of hops from the matched entities.
            max_nodes: Hard limit on returned nodes to prevent context explosion.

        Returns:
            A dictionary containing the structured subgraph or a natural language
            summary of the relationships found.
        """
        if not self._initialized:
            await self.initialize()

        logger.info("Graphiti query called for: '%s'", query)

        if self._client is None:
            raise RuntimeError("Graphiti client not initialized")

        # Call Graphiti's search method
        async with self._db_lock:
            results = await self._client.search(query)

        # Build a summary from the results
        edges_found = []
        summary_lines = []

        # Graphiti search returns a list of SearchResult objects or a SearchResponse
        # Depending on the version, it might be a list of edges directly or an object
        edges_to_process = []
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
                summary_lines.append(f"- {edge.source_node.name} {edge.name} {edge.target_node.name} ({edge.fact})")
            elif hasattr(edge, "source_node_uuid") and hasattr(edge, "target_node_uuid"):
                # We only have UUIDs, need to fetch nodes from Kuzu
                try:
                    driver = self._client.driver
                    query_str = """
                    MATCH (n:Entity {uuid: $src_uuid})
                    RETURN n.name as src
                    """
                    async with self._db_lock:
                        src_records, _, _ = await driver.execute_query(query_str, src_uuid=edge.source_node_uuid)
                    query_str = """
                    MATCH (m:Entity {uuid: $dst_uuid})
                    RETURN m.name as dst
                    """
                    async with self._db_lock:
                        dst_records, _, _ = await driver.execute_query(query_str, dst_uuid=edge.target_node_uuid)

                    src_name = src_records[0].get("src") if src_records else edge.source_node_uuid
                    dst_name = dst_records[0].get("dst") if dst_records else edge.target_node_uuid

                    edges_found.append(
                        {
                            "source": src_name,
                            "target": dst_name,
                            "relation": edge.name,
                            "fact": edge.fact,
                        }
                    )
                    summary_lines.append(f"- {src_name} {edge.name} {dst_name} ({edge.fact})")
                except Exception as e:
                    logger.error(f"Failed to fetch nodes for edge {edge.uuid}: {e}")

        # Fallback if Graphiti search fails to return edges due to Kuzu index issues
        # We can try to query Kuzu directly for the nodes found
        if not edges_found and hasattr(results, "nodes") and results.nodes:
            try:
                driver = self._client.driver
                node_names = [n.name for n in results.nodes]

                # Simple direct query to find relationships between these nodes
                # Graphiti stores relationships as intermediate nodes (RelatesToNode_)
                query_str = """
                MATCH (n:Entity)-[r1:RELATES_TO]->(rel:RelatesToNode_)-[r2:RELATES_TO]->(m:Entity)
                WHERE n.name IN $names OR m.name IN $names
                RETURN n.name as src, rel.name as rel_name, rel.uuid as rel_uuid, m.name as dst
                LIMIT $limit
                """
                async with self._db_lock:
                    records, _, _ = await driver.execute_query(query_str, names=node_names, limit=max_nodes)

                for record in records:
                    source = record.get("src")
                    target = record.get("dst")
                    rel_name = record.get("rel_name", "RELATES_TO")
                    rel_uuid = record.get("rel_uuid")

                    fact = ""
                    if rel_uuid:
                        try:
                            # Fetch fact separately to avoid Kuzu projection bug
                            fact_records, _, _ = await driver.execute_query(
                                "MATCH (rel:RelatesToNode_) WHERE rel.uuid = $uuid RETURN rel.fact as fact",
                                uuid=rel_uuid,
                            )
                            if fact_records:
                                fact = fact_records[0].get("fact", "")
                        except Exception as e:
                            logger.debug("Failed to fetch fact for rel %s: %s", rel_uuid, e)

                    edges_found.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": rel_name,
                            "fact": fact,
                        }
                    )
                    summary_lines.append(f"- {source} {rel_name} {target} ({fact})")
            except Exception as e:
                logger.debug("Fallback Kuzu query failed: %s", e)

        # If STILL no edges found, let's try a fuzzy search directly in Kuzu
        if not edges_found:
            try:
                driver = self._client.driver
                query_str = """
                MATCH (n:Entity)-[r1:RELATES_TO]->(rel:RelatesToNode_)-[r2:RELATES_TO]->(m:Entity)
                WHERE n.name CONTAINS $search_query OR m.name CONTAINS $search_query
                RETURN n.name as src, rel.name as rel_name, rel.uuid as rel_uuid, m.name as dst
                LIMIT $limit
                """
                async with self._db_lock:
                    records, _, _ = await driver.execute_query(query_str, search_query=query, limit=max_nodes)

                for record in records:
                    source = record.get("src")
                    target = record.get("dst")
                    rel_name = record.get("rel_name", "RELATES_TO")
                    rel_uuid = record.get("rel_uuid")

                    fact = ""
                    if rel_uuid:
                        try:
                            # Fetch fact separately to avoid Kuzu projection bug
                            fact_records, _, _ = await driver.execute_query(
                                "MATCH (rel:RelatesToNode_) WHERE rel.uuid = $uuid RETURN rel.fact as fact",
                                uuid=rel_uuid,
                            )
                            if fact_records:
                                fact = fact_records[0].get("fact", "")
                        except Exception as e:
                            logger.debug("Failed to fetch fact for rel %s: %s", rel_uuid, e)

                    edges_found.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": rel_name,
                            "fact": fact,
                        }
                    )
                    summary_lines.append(f"- {source} {rel_name} {target} ({fact})")
            except Exception as e:
                logger.debug("Fuzzy Kuzu query failed: %s", e)

        summary = "No relevant relationships found in the graph."
        if summary_lines:
            summary = "Graph relationships found:\n" + "\n".join(summary_lines)

        if hasattr(results, "nodes") or isinstance(results, list):
            nodes_set: set[str] = set()
            if hasattr(results, "nodes"):
                for node in results.nodes:
                    if hasattr(node, "name"):
                        nodes_set.add(node.name)
            elif isinstance(results, list):
                # If it's a list of edges, we can estimate nodes from unique sources/targets
                for edge in edges_to_process:
                    if hasattr(edge, "source_node"):
                        nodes_set.add(edge.source_node.name)
                    if hasattr(edge, "target_node"):
                        nodes_set.add(edge.target_node.name)

        nodes_found_list = list(set([e["source"] for e in edges_found] + [e["target"] for e in edges_found]))
        return {
            "query": query,
            "nodes_found": nodes_found_list,
            "summary": summary,
            "edges": edges_found,
        }


# Global singleton for the application
_graph_client: GraphitiClient | None = None


def get_graph_client() -> GraphitiClient:
    """Get or create the global Graphiti client instance."""
    global _graph_client
    if _graph_client is None:
        _graph_client = GraphitiClient()
    return _graph_client
