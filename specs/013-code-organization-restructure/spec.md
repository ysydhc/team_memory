# Feature Specification: Phase 0 — 代码组织重组

**Feature Branch**: `013-code-organization-restructure`
**Created**: 2026-05-13
**Status**: Draft
**Input**: 三位专家（架构师/后台开发/RAG工程师）代码组织审查结论 + 用户确认的重组方案

## Background

三位专家对 TeamMemory 全项目的代码存放位置进行了审查，发现多个目录存在「平铺文件过多」「跨平台代码重复」「逻辑子域未分包」等问题。本次重组聚焦**代码文件的物理位置和模块组织**，不修改任何业务逻辑，不改变功能行为。

### 现状问题

| 目录 | 文件数 | 问题 |
|------|--------|------|
| `hooks/` | 22 | 3 个平台（Claude/Cursor/Hermes）的 Hook 脚本平铺，大量重复的 stdin 解析、HTTP POST、输出格式化逻辑 |
| `daemon/` | 38 | 30+ 文件平铺，概念子域（缓冲区、回填、质量、Wiki、观测）未分包 |
| `embedding/` | 6 | 缺少 `factory.py`，工厂函数藏在 `bootstrap.py`（L3 层），与 `reranker/` 模式不对称 |
| `services/` | 25 | `memory_operations.py` 为 God module，领域子域未拆分 |

## User Scenarios & Testing

### User Story 1 — hooks/ 适配器重组 (Priority: P1)

开发者打开 `hooks/` 目录时，能按平台（Claude/Cursor/Hermes）快速定位对应 Hook 脚本，共享的 stdin 解析、HTTP POST、输出格式化逻辑集中在 `core/` 下，新增平台只需实现适配器接口。

**Why this priority**: hooks/ 的代码重复率最高（3 个平台各自重复 stdin 解析 + HTTP POST + 格式化），是重组收益最大的模块

**Independent Test**: 执行 `make test` 全部 hook 相关测试通过；`hooks.json` 中的路径仍可正确调用各 Hook 脚本；`make harness-check` import 检查通过

**Acceptance Scenarios**:

1. **Given** 重组后的 `hooks/` 目录结构, **When** 查看目录, **Then** 存在 `hooks/platforms/claude/`、`hooks/platforms/cursor/`、`hooks/platforms/hermes/` 子目录，各自只包含该平台的 Hook 脚本
2. **Given** `hooks/core/` 目录, **When** 查看内容, **Then** 包含共享的基础设施代码（stdin 解析、HTTP 客户端、输出格式化）
3. **Given** `.cursor/hooks.json` 配置, **When** 检查所有 Hook 路径, **Then** 所有路径指向重组后的正确位置且可执行
4. **Given** 任意 Claude/Cursor/Hermes Hook, **When** 通过模拟 stdin 调用, **Then** 行为与重组前完全一致
5. **Given** `hooks/` 旧路径, **When** 有外部引用, **Then** 提供 re-export shim 或文档说明迁移路径

---

### User Story 2 — daemon/ 子域分包 (Priority: P2)

开发者浏览 `daemon/` 目录时，能按功能域（runtime、buffers、refinement、backfill、quality、wiki、observe）快速定位文件，而非在 38 个平铺文件中搜索。

**Why this priority**: daemon/ 文件数最多但重复较少，主要收益是可发现性和可维护性

**Independent Test**: `make test` 全部 daemon 相关测试通过；`tm-daemon` 入口正常启动；`make harness-check` import 检查通过

**Acceptance Scenarios**:

1. **Given** 重组后的 `daemon/` 目录, **When** 查看目录, **Then** 存在按功能域命名的子包（如 `runtime/`、`buffers/`、`refinement/` 等）
2. **Given** `daemon/__init__.py`, **When** 外部代码 import daemon 模块, **Then** 通过顶层 re-export 保持向后兼容
3. **Given** `tm-daemon` 命令, **When** 启动 daemon, **Then** 正常运行，所有后台任务正确加载
4. **Given** 任意 daemon 模块的 import 路径, **When** 使用旧路径 import, **Then** 通过 re-export shim 仍可工作（过渡期）

---

### User Story 3 — embedding/ 工厂迁出 (Priority: P2)

`embedding/` 包拥有自己的 `factory.py`，与 `reranker/factory.py` 模式对称，`bootstrap.py` 不再持有 embedding provider 创建逻辑。

**Why this priority**: 改动面小但消除了 L3→L1 的隐式依赖，且与 reranker 模式统一

**Independent Test**: `make test` embedding 相关测试通过；`bootstrap.py` 中不再包含 `_create_embedding_provider` 函数

**Acceptance Scenarios**:

1. **Given** `embedding/factory.py`, **When** 查看内容, **Then** 包含从 `bootstrap.py` 迁入的 embedding provider 创建逻辑
2. **Given** `bootstrap.py`, **When** 搜索 embedding 创建逻辑, **Then** 改为调用 `embedding.factory.create_embedding_provider()`
3. **Given** 现有的 embedding 配置, **When** 启动系统, **Then** embedding provider 正常初始化

---

### User Story 4 — services/ 领域子包 (Priority: P3)

`services/` 目录按领域拆分为子包（如 `search/`、`memory/`、`llm/`、`entity/`、`maintenance/`），`memory_operations.py` 的 God module 问题得到缓解。

**Why this priority**: services/ 改动面最大、风险最高，需在前 3 个 Story 稳定后进行

**Independent Test**: `make verify` 全绿；`make harness-check` import 检查通过；所有 MCP 工具正常调用

**Acceptance Scenarios**:

1. **Given** 重组后的 `services/` 目录, **When** 查看目录, **Then** 按领域存在子包
2. **Given** `services/__init__.py`, **When** 外部代码 import services 模块, **Then** 通过顶层 re-export 保持向后兼容
3. **Given** `make verify`, **When** 执行, **Then** 全部测试通过
4. **Given** MCP 工具调用, **When** 使用任意 memory_* 工具, **Then** 功能行为与重组前一致

---

### Edge Cases

- Hook 脚本被 `.cursor/hooks.json` 按绝对/相对路径引用，路径变更需同步更新
- 部分 daemon 模块被外部 cron 或 systemd 直接调用，需确认入口点兼容
- `harness_import_check.py` 的 `LAYER_MAP` 需要更新以识别新子包路径
- 现有测试中 mock 的 import 路径需要批量更新
- re-export shim 需设定清理时间线（建议下一个 minor version 移除）

## Requirements

### Functional Requirements

- **FR-001**: 重组 MUST NOT 改变任何模块的对外行为（输入/输出/副作用完全一致）
- **FR-002**: 重组后 `make verify`（lint + 全量测试）MUST 全绿
- **FR-003**: 重组后 `make harness-check`（import 门禁）MUST 零违规
- **FR-004**: `hooks/` MUST 按平台分为 `platforms/claude/`、`platforms/cursor/`、`platforms/hermes/` 子目录，共享逻辑集中在 `core/`
- **FR-005**: `daemon/` MUST 按功能域拆分为子包，顶层 `__init__.py` 提供 re-export
- **FR-006**: `embedding/` MUST 拥有 `factory.py`，`bootstrap.py` 中的工厂函数迁入
- **FR-007**: `services/` MUST 按领域拆分为子包，顶层 `__init__.py` 提供 re-export
- **FR-008**: `.cursor/hooks.json` 中所有 Hook 路径 MUST 更新为重组后的路径
- **FR-009**: `harness_import_check.py` 的 `LAYER_MAP` MUST 更新以覆盖新子包
- **FR-010**: 每个 Story 完成后 MUST 提供 re-export shim，保证旧 import 路径在过渡期可用

### Key Entities

- **Hook 脚本**: 平台特定的事件处理脚本，读取 stdin JSON → 调用 Daemon HTTP API → 格式化输出
- **Daemon 模块**: 后台任务处理器（缓冲区、回填、精炼、质量评估、Wiki 编译、观测等）
- **Embedding Provider**: 向量化模型的抽象工厂（Ollama/OpenAI/Generic）
- **Service 模块**: 业务逻辑层（搜索、记忆、LLM、实体、归档、维护等领域）

## Success Criteria

### Measurable Outcomes

- **SC-001**: `make verify` 全绿——零测试失败、零 lint 错误
- **SC-002**: `make harness-check` 零 import 违规
- **SC-003**: `hooks/` 目录从 22 个平铺文件 → 3 个平台子目录 + 1 个 core 目录
- **SC-004**: `daemon/` 目录从 38 个平铺文件 → 若干功能域子包，每个子包 ≤8 个文件
- **SC-005**: `embedding/` 拥有 `factory.py`，`bootstrap.py` 不再包含 `_create_embedding_provider`
- **SC-006**: 所有 Hook 脚本通过 `hooks.json` 配置可正常调用
- **SC-007**: 重组前后功能行为完全一致（无回归）

## Assumptions

- 现有测试套件覆盖了受影响模块的核心路径，可作为回归保护
- re-export shim 在过渡期（1-2 个版本）内保留，之后可安全移除
- `hooks.json` 路径更新是原子操作，不存在中间状态
- 外部消费者（如用户自定义 cron）的 import 路径变更可通过 CHANGELOG 通知
- `services/memory_operations.py` 的拆分（US4）可能需要更细致的设计，本 spec 只要求物理文件迁移，不重构函数签名
