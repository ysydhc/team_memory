"""End-to-end test: MCP Client <-> MCP Server <-> PostgreSQL.

Tests the complete flow as it would work in Cursor:
1. Search experiences (empty DB -> no results)
2. Save a new experience
3. Search again (should find the saved experience)
4. Provide feedback
5. Update the experience

Requires: PostgreSQL with pgvector running and OPENAI_API_KEY set,
OR uses mock embedding via environment override.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def run_e2e():
    """Run the end-to-end test."""
    from fastmcp import Client
    from team_doc.server import mcp

    # Override service with mock embedding to avoid needing OpenAI API key
    import team_doc.server as srv
    from team_doc.auth.provider import NoAuth
    from team_doc.services.experience import ExperienceService

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from conftest import MockEmbeddingProvider

    mock_embed = MockEmbeddingProvider(dimension=1536)
    srv._service = ExperienceService(
        embedding_provider=mock_embed,
        auth_provider=NoAuth(),
    )

    print("=" * 60)
    print("team_doc MCP Server — End-to-End Test")
    print("=" * 60)

    async with Client(mcp) as client:
        # ---- Step 1: List available tools ----
        print("\n[1] Listing available tools...")
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"    Tools: {tool_names}")
        assert "search_experiences" in tool_names
        assert "save_experience" in tool_names
        assert "feedback_experience" in tool_names
        assert "update_experience" in tool_names
        print("    ✓ All 4 tools registered")

        # Helper to extract JSON from call_tool result
        def parse_result(result) -> dict:
            return json.loads(result.data)

        # ---- Step 2: Search (should be empty or have prior data) ----
        print("\n[2] Searching for 'FastAPI CORS configuration'...")
        result = await client.call_tool(
            "search_experiences",
            {"query": "FastAPI CORS configuration"},
        )
        search_data = parse_result(result)
        print(f"    Result: {search_data['message']}")

        # ---- Step 3: Save a new experience ----
        print("\n[3] Saving new experience...")
        result = await client.call_tool(
            "save_experience",
            {
                "title": "Fix FastAPI CORS for React Frontend",
                "problem": "React frontend gets CORS errors when calling FastAPI backend on a different port during development.",
                "solution": "Add CORSMiddleware to FastAPI app with allow_origins=['http://localhost:3000'], allow_methods=['*'], allow_headers=['*']. For production, restrict origins to actual domain.",
                "tags": ["fastapi", "cors", "react", "python"],
                "code_snippets": "from fastapi.middleware.cors import CORSMiddleware\napp.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000'])",
                "language": "python",
                "framework": "fastapi",
            },
        )
        save_data = parse_result(result)
        print(f"    Result: {save_data['message']}")
        exp_id = save_data["experience"]["id"]
        print(f"    Experience ID: {exp_id}")
        assert "saved successfully" in save_data["message"].lower()
        print("    ✓ Experience saved")

        # ---- Step 4: Search again (should find the saved experience) ----
        print("\n[4] Searching for 'CORS error React FastAPI'...")
        result = await client.call_tool(
            "search_experiences",
            {
                "query": "CORS error React FastAPI",
                "min_similarity": 0.0,  # Low threshold for mock embeddings
            },
        )
        search_data = parse_result(result)
        print(f"    Result: {search_data['message']}")
        assert len(search_data["results"]) >= 1
        found = any(
            "CORS" in r.get("title", "") for r in search_data["results"]
        )
        assert found, f"Expected to find CORS experience, got: {search_data['results']}"
        print(f"    ✓ Found {len(search_data['results'])} result(s), including the CORS fix")

        # ---- Step 5: Feedback ----
        print("\n[5] Submitting feedback (helpful=true)...")
        result = await client.call_tool(
            "feedback_experience",
            {
                "experience_id": exp_id,
                "helpful": True,
                "comment": "Exactly what I needed!",
            },
        )
        feedback_data = parse_result(result)
        print(f"    Result: {feedback_data['message']}")
        assert "recorded" in feedback_data["message"].lower()
        print("    ✓ Feedback recorded")

        # ---- Step 6: Update ----
        print("\n[6] Updating experience with additional solution...")
        result = await client.call_tool(
            "update_experience",
            {
                "experience_id": exp_id,
                "solution_addendum": "Alternative: Use nginx reverse proxy to avoid CORS entirely in production.",
                "tags": ["nginx", "deployment"],
            },
        )
        update_data = parse_result(result)
        print(f"    Result: {update_data['message']}")
        assert "updated successfully" in update_data["message"].lower()
        # Verify tags are merged
        updated_tags = update_data["experience"]["tags"]
        assert "fastapi" in updated_tags
        assert "nginx" in updated_tags
        print(f"    ✓ Updated. Tags: {updated_tags}")

    print("\n" + "=" * 60)
    print("ALL E2E TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_e2e())
