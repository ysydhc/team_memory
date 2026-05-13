# Feature Specification: Phase 2 — RAG 质量提升

**Feature Branch**: `014-phase2-rag-quality`
**Created**: 2026-05-13
**Status**: Draft
**Input**: User description: "Phase 2 需求：RAG 质量提升。主要包含：1. 废弃并删除 [mem:] 标记；2. Cross-Encoder 推理异步化；3. 增加 TS 查询转义保障安全；4. 将自适应截断移至重排之后并扩大重排前候选；5. 落实 grouped 和 top_k_children 参数实现结果聚合。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 废弃并删除 `[mem:]` 标记 (Priority: P1)

移除现有的 `[mem:]` 标记机制。由于该标记机制在实际使用中存在体验问题，并会导致后续评估管线的偏置，因此需要全面清理该功能及其关联代码。

**Why this priority**: 这是一个会直接影响用户体验和当前系统准确度评估的冗余功能，清理成本极低但收益立竿见影。

**Independent Test**: Can be fully tested by submitting a query to the agent and verifying that the returned context/results no longer contain `[mem:xxxx]` tokens, and evaluation jobs complete successfully without requiring them.

**Acceptance Scenarios**:

1. **Given** A user query is processed by the search pipeline, **When** context is provided to the LLM agent, **Then** the context text must NOT contain any `[mem:xxx]` markers.
2. **Given** Evaluation service is running, **When** checking for used memory entries, **Then** the check must succeed or safely bypass without crashing due to missing `[mem:]` markers.

---

### User Story 2 - TS 查询转义保障安全 (Priority: P1)

在全文本搜索（FTS）构建查询条件时，必须对输入中可能存在的 PostgreSQL `tsquery` 元字符（如 `!`, `&`, `|`, `()`, `:` 等）进行转义或白名单过滤，防止搜索词引起数据库报错或潜在的注入风险。

**Why this priority**: 直接涉及到系统的运行时稳定性和安全性，任何异常的字符输入都可能导致查询失败。

**Independent Test**: Can be fully tested by searching for queries with special characters (e.g., "Error: 500 & timeout!") and verifying that the database executes the search successfully without throwing a syntax error.

**Acceptance Scenarios**:

1. **Given** A search query containing special characters like `!`, `&`, or `|`, **When** the query is sent to the search pipeline, **Then** it must not result in a PostgreSQL `tsquery` syntax error, and valid results should still be returned if matched.

---

### User Story 3 - Cross-Encoder 推理异步化 (Priority: P1)

将同步执行的 Cross-Encoder 推理过程隔离到异步线程中执行，防止重排阶段在处理大量并发请求时阻塞整个 asyncio 事件循环。

**Why this priority**: 直接影响并发性能。同步的 ML 推理在异步框架中是明显的反模式，会导致在高负载下所有请求挂起。

**Independent Test**: Can be fully tested by simulating multiple concurrent search requests and verifying that the event loop remains responsive (e.g., other endpoints still return instantly) while the reranking is processing.

**Acceptance Scenarios**:

1. **Given** The server is handling concurrent requests, **When** a search request reaches the reranking stage, **Then** it must execute `asyncio.to_thread` (or equivalent) to avoid blocking other concurrent requests.

---

### User Story 4 - 优化自适应截断与重排顺序 (Priority: P2)

重构检索管线的执行顺序：将原本在重排之前的“自适应截断（Adaptive Filter）”逻辑，移动到 Cross-Encoder 重排之后。同时，在初筛阶段扩大传递给重排器的候选集数量。

**Why this priority**: 这是 RAG 召回质量的关键优化。重排器如果只能看到已经被截断的候选集，其精准打分的价值就被大大压缩了。

**Independent Test**: Can be fully tested by observing the pipeline logs to verify that the reranker receives a larger pool of candidates, and the adaptive filter only applies its threshold logic on the final reranked scores.

**Acceptance Scenarios**:

1. **Given** A search pipeline execution, **When** candidate results are passed to the reranking stage, **Then** the candidate pool must be larger than the final target count, and adaptive filtering must occur AFTER the reranking stage is complete.

---

### User Story 5 - 落实 `grouped` 和 `top_k_children` 结果聚合 (Priority: P2)

在检索管线最终输出前，实现根据父级（如文档、会话、项目）进行结果聚合的逻辑。当 `grouped=True` 时，必须返回聚合后的结果结构，且每个分组内最多保留 `top_k_children` 个匹配项。

**Why this priority**: 这是修复已有 API 契约断裂的重要任务。目前参数存在但未生效，导致客户端请求的分组功能形同虚设。

**Independent Test**: Can be fully tested by calling the search API with `grouped=True` and `top_k_children=2`, and verifying that the returned payload correctly groups results by their parent ID, with no more than 2 children per group.

**Acceptance Scenarios**:

1. **Given** Multiple search results belong to the same parent document, **When** a search is executed with `grouped=True` and `top_k_children=2`, **Then** the results must be grouped by the parent, and that specific group must contain a maximum of 2 child matches.

### Edge Cases

- What happens when a search query consists ENTIRELY of `tsquery` special characters (e.g., `!!! &&&`)? The system should strip/escape them and fall back gracefully, either ignoring the terms or executing an empty search without crashing.
- What happens if the `grouped=True` logic receives results that do not have a defined parent ID? They should be treated as standalone top-level items.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST process concurrent searches without blocking the main event loop during cross-encoder prediction.
- **FR-002**: System MUST sanitize or escape all FTS inputs to prevent PostgreSQL `tsquery` syntax errors.
- **FR-003**: System MUST NOT inject or evaluate `[mem:]` markers in search results or prompts.
- **FR-004**: System MUST perform adaptive filtering only on the output scores of the reranker, not before reranking.
- **FR-005**: System MUST increase the intermediate candidate pool size fed into the reranker.
- **FR-006**: System MUST group output results by parent identity when `grouped=True`.
- **FR-007**: System MUST limit the number of child items per group to `top_k_children` when `grouped=True`.

### Key Entities

- **SearchPipelineResult**: Represents the output of a search execution. Will now reflect grouped structures if requested.
- **SearchRequest/RetrievalConfig**: The configuration object containing `grouped` and `top_k_children` parameters that dictate the pipeline's final formatting step.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Zero `tsquery` syntax errors logged in production when users search for non-alphanumeric text.
- **SC-002**: Concurrent search throughput (RPS) improves or P99 latency remains stable due to the removal of synchronous event loop blocking.
- **SC-003**: `[mem:]` markers are 100% absent from the codebase and from any LLM prompt payloads.
- **SC-004**: Reranker processing logs show an increased number of input candidates (e.g., from top N to top 2N or 3N) prior to adaptive filtering.
- **SC-005**: API responses for searches with `grouped=True` correctly return grouped data structures with enforced child limits.

## Assumptions

- We assume the existing Cross-Encoder model is thread-safe and can be executed via `asyncio.to_thread` without underlying C-extension thread contention issues.
- We assume removing `[mem:]` tracking won't break upstream analytics dashboards that relied purely on explicit feedback, or that explicit feedback is handled through a separate channel (e.g. `memory_feedback`).
- We assume that `top_k_children` filtering applies a simple ranking logic (e.g., retaining the top scoring children within each group) rather than complex diversity requirements.