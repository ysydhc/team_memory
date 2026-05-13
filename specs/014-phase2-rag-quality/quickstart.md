# Quickstart: Phase 2 — RAG 质量提升

This phase focuses on internal refactoring and optimization. There are no new CLI commands or environment variables required to run the application after these changes.

## Verification

After implementation, you can verify the changes by:

1. **Checking Logs for Reranker**:
   Run the daemon with debug logging to observe the number of candidates passed to the reranker before and after the adaptive filter.

2. **Testing Grouped API**:
   Use `tm-cli recall --query "your query" --grouped` (if supported by CLI) or test the API endpoint directly to ensure results are grouped correctly.

3. **TS Query Safety**:
   Try searching with punctuation and special characters:
   ```bash
   tm-cli recall --query "foo & bar ! () :"
   ```
   This should return results normally without crashing with a PostgreSQL syntax error.

4. **No [mem:] Markers**:
   Perform searches and confirm the output format no longer prefixes titles with `[mem:xxxx]`.