# Data Model: Phase 2 — RAG 质量提升

This phase does not introduce new database tables or persistent schema migrations. It modifies the runtime data structures used in the search pipeline.

## Runtime Entities

### `SearchPipelineResult`

Represents the finalized output of a search execution.

**Changes**:
When `grouped=True` is provided to the pipeline, the structure of the returned items may change from a flat list to a hierarchical list, or remain a flat list that guarantees the grouping constraints.
*Decision*: To maintain backward compatibility with existing consumers (like the LLM hooks which expect a flat stringifiable list), the pipeline will return a **flat list of dictionaries**, but the list will be guaranteed to contain at most `top_k_children` items per distinct parent. The ordering will be determined by the best-scoring item of the group.

### `RetrievalConfig`

Configuration object for the search pipeline.

**Fields utilized**:
- `grouped` (bool): Whether to enforce parent-based grouping constraints.
- `top_k_children` (int): The maximum number of child items to return per unique parent group.
- `adaptive_filter` (bool): Whether to prune low-scoring results dynamically. (Behavior modified to run post-reranking).