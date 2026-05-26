# Design Spec: Agentic Multi-hop Retrieval with Faithfulness Evaluation

## 1. Overview
This spec outlines the implementation of an **Agentic Multi-hop Retrieval** system for TeamMemory, transitioning from a static single-pass retrieval to a dynamic, LLM-driven multi-hop loop. To prove the effectiveness of this new architecture, we will integrate the existing `FaithfulnessJudge` to provide quantitative A/B testing (Before vs. After) in our test suite, as well as continuous evaluation logging in the production runtime.

## 2. Architecture & Components

The implementation will be centered around a new orchestrator (or an extension of the existing one) in `services/search/adaptive_retriever.py`.

### 2.1 Core Components
1. **Intent & Complexity Evaluator**:
   - Uses a fast LLM to classify the incoming query.
   - If simple (e.g., "Where is the style guide?"), bypasses the loop and uses standard single-pass retrieval.
   - If complex, initializes the multi-hop loop.
2. **Multi-hop Loop Controller**:
   - Manages state: `collected_context` (deduplicated by ID) and `loop_count` (max 3).
   - Executes the underlying `search_pipeline` using the current query/queries.
3. **Retrieval Critic (Evaluator)**:
   - After each retrieval hop, a fast LLM evaluates if `collected_context` is sufficient to answer the original query.
   - If `is_sufficient == True` or `loop_count >= max_loops`, the loop terminates.
   - If `False`, it generates `missing_information` and `next_queries` for the next hop.
   - **Anti-Degradation Guard**: If the new context adds no value (high similarity/no new IDs), break early to save tokens.

### 2.2 Quality Evaluation Integration (Faithfulness Judge)
We will reuse `src/team_memory/daemon/quality/faithfulness_judge.py`.

- **Testing (A/B Test Script)**:
  - Create `tests/eval_adaptive_retrieval.py`.
  - Define 3-5 complex, multi-hop queries.
  - Run each query through:
    1. Legacy Single-pass Retrieval -> Generate Answer -> `FaithfulnessJudge` Score.
    2. New Adaptive Multi-hop Retrieval -> Generate Answer -> `FaithfulnessJudge` Score.
  - Output a comparative table proving the uplift in faithfulness.
- **Runtime Logging**:
  - When `memory_recall` is executed with `adaptive=True`, asynchronously trigger `FaithfulnessJudge` on the final generated response (if applicable) or log the multi-hop trace (number of hops, queries used, final context size) to the existing `SearchLog` infrastructure.

## 3. Data Flow
1. User calls `memory_recall(query="Complex question...")`.
2. `Intent Evaluator` -> `is_complex = True`.
3. Loop 1: `search_pipeline(query)` -> `Context A`.
4. `Critic(Context A, query)` -> `is_sufficient = False`, `next_queries = ["Sub-question X"]`.
5. Loop 2: `search_pipeline("Sub-question X")` -> `Context B`.
6. Merge & Deduplicate: `Context A + B`.
7. `Critic(Context A + B, query)` -> `is_sufficient = True`.
8. Return aggregated context.
9. (Async) `FaithfulnessJudge` evaluates the final LLM usage of this context and logs the score.

## 4. Constraints & Trade-offs
- **Latency**: Multi-hop inherently increases latency. We mitigate this by using fast, small models (e.g., `glm-4-flash` or equivalent configured fast model) for the Evaluator and Critic, and strictly capping `max_loops` to 3.
- **Token Limits**: Context merging must enforce a hard token limit, truncating older or lower-ranked chunks if necessary.
- **Backward Compatibility**: The existing `memory_recall` signature will remain intact, adding an optional `adaptive=True` flag (defaulting to False initially for safety).

## 5. Success Criteria
- Unit tests pass, verifying the loop logic and deduplication.
- The A/B evaluation script runs successfully and demonstrates a measurable increase in the `FaithfulnessJudge` score for complex queries compared to the baseline.
- Production logs capture the hop count and evaluation scores without blocking the main retrieval response.