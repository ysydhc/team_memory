# Research: Phase 2 — RAG 质量提升

## 1. TS Query Escaping (PostgreSQL tsquery)
- **Decision**: Use a regex replacement to strip `tsquery` meta-characters (`!`, `&`, `|`, `(`, `)`, `:`, `*`) from the input query before constructing the `&` or `|` joined FTS query string.
- **Rationale**: Manual construction of `tsquery` strings (e.g. `word1 & word2`) fails with a PostgreSQL syntax error if a word contains unescaped special characters. Stripping them is the most robust and simplest approach, as full text search ignores punctuation anyway.
- **Alternatives considered**: Using Postgres `plainto_tsquery` directly. However, the current `_build_hybrid_tsquery` specifically controls the AND/OR logic to provide a hybrid fallback, making direct string construction necessary.

## 2. Cross-Encoder Async Execution
- **Decision**: Wrap the CPU-bound ML prediction call with `asyncio.to_thread()`.
- **Rationale**: `SentenceTransformer.predict()` performs heavy matrix multiplication. In an async Python application, this blocks the event loop, causing all other concurrent requests to hang. `to_thread` offloads this to the default ThreadPoolExecutor. Since ML libraries like PyTorch/NumPy release the GIL during heavy computation, thread-based offloading is highly effective.
- **Alternatives considered**: `ProcessPoolExecutor`. Rejected because loading the Cross-Encoder model into multiple processes consumes too much memory, and IPC overhead is high for large input payloads.

## 3. Pipeline Stage Ordering (Adaptive Filter vs. Rerank)
- **Decision**: Pass more candidates (e.g., limit * 2 or limit * 3) into the Reranker, and move the Adaptive Filter to execute *after* the reranker.
- **Rationale**: The Reranker is the most accurate scorer. If the Adaptive Filter runs before it, it might prune highly relevant documents that the initial vector/FTS retrieval scored poorly. Moving the filter after the reranker ensures the threshold is applied to the most accurate scores.

## 4. Grouped and top_k_children Implementation
- **Decision**: Implement grouping at the end of the search pipeline (or in the orchestrator) using `itertools.groupby` (after sorting by the group key) or a dictionary.
- **Rationale**: The pipeline yields a flat list of `SearchResultItem`s. We will group by the entity's `parent_id` (or appropriate logical grouping key), slice the top `top_k_children` per group based on their rank, and return the flattened or nested result based on the API contract.
