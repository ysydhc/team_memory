# Research: Phase 0 — 代码组织重组

**Branch**: `013-code-organization-restructure` | **Date**: 2026-05-13

## R1: hooks/ 重复代码详细分析

### Decision
抽取 4 个 core 模块（daemon_http, stdin, mcp_callback, output），12 个平台 Hook 按 Claude/Cursor/Hermes 分入 platforms/，5 个跨平台工具放入 integrations/。

### Rationale
- `DAEMON_URL = "http://127.0.0.1:3901"` 在 12 个文件中重复定义
- MCP recall → mcp_callback 逻辑在 claude_post_tool_use / cursor_after_mcp / hermes_post_tool_call 中各有 ~65 行几乎相同的代码
- `common.py` 的 `parse_hook_input` 未被任何生产 Hook 使用（仅 `tests/test_hook_common.py` 引用）
- `shared.py` 定义了 `HookInput` / `PipelineConfig` 但全仓库零引用（dead API）
- `session_timeout.py` import `team_memory.daemon.draft_buffer`、`draft_refiner`，本质是 daemon 组件，不属于 hooks/

### 详细清单

| 文件 | 平台 | 行数 | 导入 common/shared | 模式 |
|------|------|------|-------------------|------|
| claude_session_start.py | claude | 71 | 无 | stdin→POST /hooks/session_start→stdout additionalContext |
| claude_user_prompt_submit.py | claude | 86 | 无 | stdin→POST /hooks/before_prompt→stdout additionalContext |
| claude_stop.py | claude | 67 | 无 | stdin→POST /hooks/after_response→stdout {} |
| claude_post_tool_use.py | claude | 106 | 无 | stdin→recall?→POST /hooks/mcp_callback→stdout {} |
| cursor_session_start.py | cursor | 29 | 无 | stdin→POST /hooks/session_start→resp.text |
| cursor_before_prompt.py | cursor | 39 | 无 | stdin→POST /hooks/before_prompt→action/context |
| cursor_after_response.py | cursor | 68 | 无 | stdin→POST /hooks/after_response→resp.text |
| cursor_after_mcp.py | cursor | 106 | 无 | stdin→recall?→POST /hooks/mcp_callback→{} |
| hermes_pipeline.py | hermes | 146 | 无 | async API（非 stdin）|
| hermes_stop.py | hermes | 37 | 无 | stdin→POST /hooks/after_response→{} |
| hermes_user_prompt_submit.py | hermes | 46 | 无 | stdin→POST /hooks/before_prompt→context/{} |
| hermes_post_tool_call.py | hermes | 111 | 无 | stdin→recall?→POST /hooks/mcp_callback→{} |
| common.py | infra | 125 | — | load_config, parse_hook_input, call_mcp_tool |
| shared.py | infra | 195 | — | TMClient, HookInput(dead), PipelineConfig(dead) |
| session_timeout.py | daemon | 139 | 无 | draft timeout manager |
| weekly_report.py | utility | 89 | TMClient | 周报生成 |
| git_watcher.py | utility | 275 | TMClient | Git→TM 索引 |
| initial_index.py | utility | 115 | TMClient | Obsidian 批量导入 |
| retrieval_trigger.py | utility | 24 | 无 | 纯逻辑（关键词触发） |

### .cursor/hooks.json 路径引用

| 事件 | 模块路径 |
|------|---------|
| afterAgentResponse | `team_memory.hooks.cursor_after_response` |
| sessionStart | `team_memory.hooks.cursor_session_start` |
| beforeSubmitPrompt | `team_memory.hooks.cursor_before_prompt` |
| afterMCPExecution | `team_memory.hooks.cursor_after_mcp` |

### Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| 仅移动文件不抽取 core | 重复代码仍在，只是换了目录 |
| 全部重写为 adapter 抽象类 | 过度设计，各平台 stdout 格式差异大，强制统一得不偿失 |
| 保持平铺但加 prefix 命名 | 不解决代码重复问题 |

---

## R2: daemon/ 子域分类验证

### Decision
8 个子包：runtime、buffers、refinement、backfill、quality、wiki、observe、integration。

### Rationale
- 36 个 Python 文件，依赖图为树形（app.py 为根），无循环
- 大多数模块是叶子节点（无 daemon 内部 import），天然安全分包
- `python -m team_memory.daemon.faithfulness_batch` 等 CLI 入口必须保持路径有效

### 验证后的分组

| 子包 | 文件 | 调整说明 |
|------|------|---------|
| runtime | __main__, app, config, reloader, health_check, structured_logger | structured_logger 仅被 refinement_worker 使用，但体量小 |
| buffers | conversation_buffer, draft_buffer, response_buffer, pending_queue, payload_capture | response_buffer 同时服务 quality，但物理上属于缓冲区 |
| refinement | pipeline, draft_refiner, refinement_worker | pipeline 是 hook 编排中枢 |
| backfill | embedding_backfill, entity_backfill, solution_backfill, entity_dedup | 全部 CLI 工具，无 daemon 内部依赖 |
| quality | faithfulness_batch, faithfulness_judge, search_eval, convergence_detector | — |
| wiki | llm_wiki, wiki_cli, wiki_compiler, markdown_indexer | — |
| observe | metrics_collector, search_log_writer, tm_stats, watcher | watcher→wiki/markdown_indexer 单向依赖 |
| integration | tm_sink, auto_dream, personal_memory_extractor, intent_classifier, test_llm | — |

### 跨包依赖（验证无环）

```
runtime/app → 所有包（composition root，可接受）
refinement/pipeline → integration/{tm_sink, personal_memory_extractor, intent_classifier}
refinement/refinement_worker → buffers/conversation_buffer, integration/tm_sink
quality/faithfulness_batch → quality/faithfulness_judge, buffers/response_buffer
observe/watcher → wiki/markdown_indexer, integration/tm_sink
observe/tm_stats → observe/search_log_writer, buffers/response_buffer
integration/personal_memory_extractor → integration/intent_classifier
integration/auto_dream → integration/tm_sink
```

### Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| 更细粒度分包（15+ 子包）| 目录深度过大，降低可发现性 |
| 仅分 3 个大包 | 颗粒度太粗，daemon 各子域差异大 |

---

## R3: embedding factory 迁出

### Decision
将 `bootstrap.py` L216-302 的 `_create_embedding_provider` 迁入 `embedding/factory.py` 作为 `create_embedding_provider`。

### Rationale
- 函数位于 L3 层（bootstrap.py），被 L2 层的 daemon 模块通过私有 API `_create_embedding_provider` 访问
- `reranker/factory.py` 已有对称的工厂模式（`create_reranker` 公共 API）
- 迁移后 embedding 包自给自足，不依赖 bootstrap

### 调用方更新

| 调用方 | 当前 | 迁移后 |
|--------|------|--------|
| `bootstrap.py` | 直接调用 `_create_embedding_provider(settings)` | `from team_memory.embedding.factory import create_embedding_provider` |
| `daemon/embedding_backfill.py` | `from team_memory.bootstrap import _create_embedding_provider` | `from team_memory.embedding.factory import create_embedding_provider` |
| `daemon/app.py` ~972 | `from team_memory.bootstrap import _create_embedding_provider` | 同上 |
| `daemon/pipeline.py` ~183 | `from team_memory.bootstrap import _create_embedding_provider` | 同上 |
| 测试 mock | `patch("team_memory.bootstrap._create_embedding_provider")` | `patch("team_memory.embedding.factory.create_embedding_provider")` |

---

## R4: services/ 领域分析

### Decision
8 个子包：search、memory、llm、entity、archive、maintenance、analysis、infra。

### Rationale
- `memory_operations.py` 1113 行，是明显的 God module，但本次仅做物理迁移，不拆分函数
- 子包间无循环依赖
- LAYER_MAP 无需更新（子包仍匹配 `src/team_memory/services/` 前缀）

### memory_operations.py 函数归类

| 域 | 函数 |
|----|------|
| search/recall | op_recall, op_context, write_search_log |
| experience/drafts | op_save, op_draft_save, op_draft_publish, op_experience_update, op_feedback |
| archive | op_get_archive, op_archive_upsert |

注：本次仅做文件迁移，memory_operations.py 不拆分。拆分留待后续 Phase。

### 跨子包依赖

```
search/search_pipeline → search/cache, llm/llm_client, entity/entity_search, analysis/topic_discovery
search/search_orchestrator → search/search_pipeline, analysis/evaluation, infra/intent_router
memory/experience → archive/archive, infra/event_bus
memory/memory_operations → archive/archive, llm/, memory/personal_memory
maintenance/janitor → maintenance/promotion_compiler
analysis/pattern_extractor → memory/personal_memory, llm/llm_provider
```

无循环。
