# Implementation Plan: Phase 0 — 代码组织重组

**Branch**: `013-code-organization-restructure` | **Date**: 2026-05-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/013-code-organization-restructure/spec.md`

## Summary

将 TeamMemory 项目中 4 个核心包（hooks/、daemon/、embedding/、services/）从平铺文件结构重组为按平台/功能域分层的子包结构。不修改任何业务逻辑，仅变更文件物理位置和 import 路径，通过 re-export shim 保证向后兼容。

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: FastMCP, FastAPI, asyncpg, SQLAlchemy (async), httpx, aiosqlite
**Storage**: PostgreSQL + pgvector, SQLite (daemon buffers)
**Testing**: pytest (async), `make verify` = lint + 全量测试, `make harness-check` = import 门禁
**Target Platform**: Linux/macOS (MCP stdio + HTTP)
**Project Type**: MCP server + Web service + CLI + Daemon
**Constraints**: 严格分层 Server/Web→Services→Storage→Models；零行为变更
**Scale/Scope**: hooks/ 22 文件, daemon/ 36 文件, services/ 23 文件, embedding/ 6 文件

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| I. 分层架构 | ✅ PASS | 不引入新的层级违规；US3 消除 bootstrap→embedding 的 L3→L2 隐式依赖 |
| II. 异步优先 | ✅ PASS | 不涉及 IO 路径变更 |
| III. 类型安全 | ✅ PASS | 不变更函数签名 |
| IV. 项目隔离 | ✅ PASS | 不涉及查询逻辑 |
| V. 双语 | ✅ PASS | 新增 docstring 遵循中英双语 |
| VI. 可观测性 | ✅ PASS | 不变更日志 |
| Non-Negotiable #2 — 禁止向上 import | ✅ IMPROVE | embedding factory 迁出后消除一处 L3→L2 访问 |

**Gate Result**: 全部 PASS。纯重构，无新增违规。

## Project Structure

### Documentation (this feature)

```text
specs/013-code-organization-restructure/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: 研究结论
├── data-model.md        # Phase 1: 无 schema 变更说明
└── tasks.md             # Phase 2 output (by /speckit-tasks)
```

### Source Code (target structure after restructure)

```text
src/team_memory/
├── hooks/
│   ├── __init__.py              # re-export shim 保持旧路径兼容
│   ├── core/
│   │   ├── __init__.py
│   │   ├── daemon_http.py       # 统一 DAEMON_URL + sync httpx POST
│   │   ├── stdin.py             # parse_hook_input (from common.py)
│   │   ├── mcp_callback.py      # 共享的 recall → mcp_callback 逻辑
│   │   └── output.py            # 平台特定 JSON 输出适配器
│   ├── platforms/
│   │   ├── __init__.py
│   │   ├── claude/
│   │   │   ├── __init__.py
│   │   │   ├── session_start.py
│   │   │   ├── user_prompt_submit.py
│   │   │   ├── stop.py
│   │   │   └── post_tool_use.py
│   │   ├── cursor/
│   │   │   ├── __init__.py
│   │   │   ├── session_start.py
│   │   │   ├── before_prompt.py
│   │   │   ├── after_response.py
│   │   │   └── after_mcp.py
│   │   └── hermes/
│   │       ├── __init__.py
│   │       ├── pipeline.py
│   │       ├── stop.py
│   │       ├── user_prompt_submit.py
│   │       └── post_tool_call.py
│   ├── integrations/             # 跨平台工具（非 stdin hook）
│   │   ├── __init__.py
│   │   ├── weekly_report.py
│   │   ├── git_watcher.py
│   │   ├── initial_index.py
│   │   └── retrieval_trigger.py
│   ├── shared.py                 # TMClient (被 integrations/ 使用)
│   └── config.yaml
│
├── daemon/
│   ├── __init__.py               # re-export shim
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── app.py
│   │   ├── config.py
│   │   ├── reloader.py
│   │   ├── health_check.py
│   │   └── structured_logger.py
│   ├── buffers/
│   │   ├── __init__.py
│   │   ├── conversation_buffer.py
│   │   ├── draft_buffer.py
│   │   ├── response_buffer.py
│   │   ├── pending_queue.py
│   │   └── payload_capture.py
│   ├── refinement/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── draft_refiner.py
│   │   └── refinement_worker.py
│   ├── backfill/
│   │   ├── __init__.py
│   │   ├── embedding_backfill.py
│   │   ├── entity_backfill.py
│   │   ├── solution_backfill.py
│   │   └── entity_dedup.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── faithfulness_batch.py
│   │   ├── faithfulness_judge.py
│   │   ├── search_eval.py
│   │   └── convergence_detector.py
│   ├── wiki/
│   │   ├── __init__.py
│   │   ├── llm_wiki.py
│   │   ├── wiki_cli.py
│   │   ├── wiki_compiler.py
│   │   └── markdown_indexer.py
│   ├── observe/
│   │   ├── __init__.py
│   │   ├── metrics_collector.py
│   │   ├── search_log_writer.py
│   │   ├── tm_stats.py
│   │   └── watcher.py
│   └── integration/
│       ├── __init__.py
│       ├── tm_sink.py
│       ├── auto_dream.py
│       ├── personal_memory_extractor.py
│       ├── intent_classifier.py
│       └── test_llm.py
│
├── embedding/
│   ├── __init__.py               # 新增 re-export
│   ├── base.py
│   ├── factory.py                # NEW: 从 bootstrap.py 迁入
│   ├── ollama_provider.py
│   ├── openai_provider.py
│   ├── generic_provider.py
│   └── local_provider.py
│
├── services/
│   ├── __init__.py               # re-export shim
│   ├── search/
│   │   ├── __init__.py
│   │   ├── search_pipeline.py
│   │   ├── search_orchestrator.py
│   │   └── cache.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── memory_operations.py
│   │   ├── experience.py
│   │   └── personal_memory.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── llm_parser.py
│   │   └── llm_provider.py
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── entity_extractor.py
│   │   └── entity_search.py
│   ├── archive/
│   │   ├── __init__.py
│   │   └── archive.py
│   ├── maintenance/
│   │   ├── __init__.py
│   │   ├── janitor.py
│   │   ├── janitor_scheduler.py
│   │   └── promotion_compiler.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   ├── contradiction_detector.py
│   │   ├── pattern_extractor.py
│   │   └── topic_discovery.py
│   └── infra/
│       ├── __init__.py
│       ├── event_bus.py
│       ├── task_runner.py
│       ├── tokenizer.py
│       └── intent_router.py
│
└── bootstrap.py                  # 删除 _create_embedding_provider，调用 embedding.factory
```

**Structure Decision**: 四个目录全部采用子包分层，顶层 `__init__.py` 提供 re-export shim 保持向后兼容。

---

## Phase 0: Research

### R1: hooks/ 重复代码分析

**Decision**: 抽取 4 个 core 模块（daemon_http, stdin, mcp_callback, output），将 12 个平台 Hook 按 Claude/Cursor/Hermes 分入 platforms/ 子包。

**Key Findings**:
- `DAEMON_URL = "http://127.0.0.1:3901"` 重复 12 次
- MCP recall → mcp_callback 逻辑在 3 个文件中重复 ~65 行
- `common.py` 的 `parse_hook_input` 未被任何生产 Hook 使用（仅测试引用）
- `shared.py` 的 `HookInput`/`PipelineConfig` 是 dead API（无人引用）
- `session_timeout.py` 属于 daemon 概念，不属于 hooks/

**Rationale**: 按平台分目录是用户明确要求的方案，core/ 抽取消除最大的重复代码块

### R2: daemon/ 子域分类验证

**Decision**: 8 个子包（runtime, buffers, refinement, backfill, quality, wiki, observe, integration）。

**Key Findings**:
- 36 个 Python 文件，无循环依赖
- `app.py` 是 composition root，import 几乎所有其他模块
- 大部分模块是叶子节点（无 daemon 内部依赖），安全分包
- `python -m team_memory.daemon.faithfulness_batch` 等 CLI 入口必须保持有效

**Rationale**: 子域分类与架构文档一致，import 图为树形（app 为根），无循环风险

### R3: embedding factory 迁出

**Decision**: `bootstrap._create_embedding_provider` (~90 行) 迁入 `embedding/factory.py`。

**Key Findings**:
- 函数位于 `bootstrap.py` L216-302，属 L3 层
- `daemon/embedding_backfill.py`、`daemon/app.py`、`daemon/pipeline.py` 通过私有 API `_create_embedding_provider` 访问
- `reranker/factory.py` 已有对称的工厂模式

**Rationale**: 消除 L2 模块访问 L3 私有函数的层级违规，与 reranker 模式对称

### R4: services/ 领域分析

**Decision**: 8 个子包（search, memory, llm, entity, archive, maintenance, analysis, infra）。

**Key Findings**:
- `memory_operations.py` 1113 行，包含 9 个 `op_*` 函数
- 子包间无循环依赖（search → llm/entity/analysis 单向）
- `LAYER_MAP` 无需更新——子包仍匹配 `src/team_memory/services/` 前缀

**Rationale**: 子包颗粒度与实际领域对齐，import 图无环

---

## Phase 1: Design

### Migration Strategy

每个 User Story 遵循相同的迁移模式：

1. **创建子包目录** + `__init__.py`
2. **移动文件**到子包（git mv）
3. **更新文件内部 import**（相对或绝对路径）
4. **在旧位置放置 re-export shim**（`from team_memory.xx.subpkg.mod import *`）
5. **更新测试中的 mock/import 路径**
6. **更新配置引用**（如 hooks.json）
7. **验证**: `make verify` + `make harness-check`

### Re-export Shim 规范

```python
# 旧路径 shim 示例: src/team_memory/hooks/claude_session_start.py
"""Backward-compatibility shim. 向后兼容 shim。

Use team_memory.hooks.platforms.claude.session_start instead.
"""
from team_memory.hooks.platforms.claude.session_start import *  # noqa: F401,F403
from team_memory.hooks.platforms.claude.session_start import main  # noqa: F401

if __name__ == "__main__":
    main()
```

### hooks/ core 模块设计

| Module | 职责 | 消除的重复 |
|--------|------|-----------|
| `core/daemon_http.py` | `DAEMON_URL` 解析（env > config > 默认）+ `post_hook(name, payload)` | 12 处 DAEMON_URL 常量 |
| `core/stdin.py` | `parse_stdin_json()` 统一 stdin 解析 | 12 处 json.load/loads 变体 |
| `core/mcp_callback.py` | `is_recall_tool(patterns)` + `handle_recall_callback(payload)` | 3×65 行 MCP 回调逻辑 |
| `core/output.py` | `claude_context(text)`, `cursor_action(data)`, `hermes_context(text)` | 散落各处的 stdout 格式化 |

### daemon/ 依赖方向（验证无环）

```
runtime/app → 所有其他子包（composition root）
refinement/pipeline → integration/tm_sink, integration/personal_memory_extractor
refinement/refinement_worker → buffers/conversation_buffer, integration/tm_sink
quality/faithfulness_batch → quality/faithfulness_judge, buffers/response_buffer
observe/watcher → wiki/markdown_indexer, integration/tm_sink
observe/tm_stats → observe/search_log_writer, buffers/response_buffer
integration/personal_memory_extractor → integration/intent_classifier
```

所有边均单向，无环。

### LAYER_MAP 影响

**无需更新**。`harness_import_check.py` 使用路径前缀匹配：
- `services/search/search_pipeline.py` 仍匹配 `src/team_memory/services/` → L2
- `embedding/factory.py` 仍匹配 `src/team_memory/embedding/` → L2
- daemon 和 hooks 不在 LAYER_MAP 中（仅客户端组件）

### Implementation Sequence

```
US1: hooks/ 适配器重组 (P1)     — 独立，最高优先级
  1a. 创建 core/ 模块（daemon_http, stdin, mcp_callback, output）
  1b. 创建 platforms/ 子包，迁移 12 个 Hook 脚本
  1c. 迁移 integrations/ (weekly_report, git_watcher 等)
  1d. 迁移 session_timeout.py → daemon/
  1e. 旧路径 re-export shim
  1f. 更新 .cursor/hooks.json
  1g. 清理 common.py 未使用代码，删除 shared.py 的 dead API
  验证: make verify + hooks 调用测试

US3: embedding/ 工厂迁出 (P2)   — 独立，改动最小
  3a. 创建 embedding/factory.py
  3b. 更新 bootstrap.py 调用
  3c. 更新 daemon/ 的 import
  3d. 更新测试 mock 路径
  验证: make verify

US2: daemon/ 子域分包 (P2)      — 独立
  2a. 创建 8 个子包目录
  2b. git mv 文件到对应子包
  2c. 更新内部 import
  2d. 旧路径 re-export shim（尤其 faithfulness_batch CLI 入口）
  2e. 更新测试 import
  验证: make verify + tm-daemon 启动测试

US4: services/ 领域子包 (P3)    — 依赖 US1-3 稳定后执行
  4a. 创建 8 个子包目录
  4b. git mv 文件到对应子包
  4c. 更新内部 import
  4d. 旧路径 re-export shim
  4e. 更新测试 import
  验证: make verify + make harness-check + MCP 工具测试
```
