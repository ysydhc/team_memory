# Feature Specification: Phase 3 — 架构治理

**Feature Branch**: `015-phase3-arch-governance`  
**Created**: 2026-05-13  
**Status**: Draft  
**Input**: Phase 3: 架构治理。包含引入依赖注入收敛 memory_operations 对 bootstrap 的懒加载依赖，以及解耦 Event Bus。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 依赖注入收敛 (Priority: P1)

在 `memory_operations.py` 中彻底移除对 `bootstrap.py`（如 `_get_service()`, `_get_settings()` 等懒加载函数）的依赖，改由上层（如 MCP server 或 Web API 路由）在调用这些 operations 时显式传入依赖。

**Why this priority**: 解决循环依赖和分层被打破的问题（目前是向下层的 operations 反向调用顶层 bootstrap）。这是保障架构整洁度和可测试性的基石。

**Independent Test**: Can be fully tested by running unit tests without mocking global bootstrap state, ensuring the system boots and routes requests successfully.

**Acceptance Scenarios**:

1. **Given** MCP server 收到调用请求，**When** 调用 `op_recall` 等操作时，**Then** server 负责组装并显式传递所需 service 实例给 operations。
2. **Given** 单元测试环境，**When** 测试 `memory_operations`，**Then** 可以直接传入 Mock 的 service 实例而无需 patch 全局的 `get_context()`。

---

### User Story 2 - Event Bus 解耦 (Priority: P1)

移除 `ExperienceService` 对 `event_bus.py` 的全局单例强耦合订阅。改用构造函数注入 EventBus 或者采用更松散的回调/观察者模式，防止隐藏的副作用和“事件风暴”。

**Why this priority**: 强耦合的全局事件总线导致服务状态极难追踪，测试脆弱，违背了单一职责原则和依赖倒置原则。

**Independent Test**: Can be fully tested by verifying that `ExperienceService` can be instantiated and tested in isolation without triggering global event handlers.

**Acceptance Scenarios**:

1. **Given** 初始化 `ExperienceService`，**When** 提供可选的 Event Bus 实例，**Then** 服务按预期发布事件。
2. **Given** `ExperienceService` 在未提供 Event Bus 时执行操作，**When** 保存或更新 experience，**Then** 操作成功完成，不报错且不触发未预期的全局副作用。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `memory_operations.py` 内部 MUST 不包含任何对 `team_memory.bootstrap` 的 import。
- **FR-002**: `op_save`, `op_recall`, `op_context` 等方法 MUST 接受显式的依赖参数（如 `ExperienceService`, `SearchOrchestrator`, `Settings` 等）。
- **FR-003**: `ExperienceService` MUST 不在模块级别或内部硬编码导入全局 `EventBus` 实例。
- **FR-004**: `ExperienceService` 的构造函数 MUST 允许传入可选的 `EventBus` 实例（或相应的 dispatcher 回调）。
- **FR-005**: 系统的 MCP 入口 `server.py` 和 Web 路由 MUST 负责从 Context 中提取依赖并注入给 operations 或 services。

### Key Entities 

- **MemoryOperations**: 核心知识库读写统筹入口。
- **ExperienceService**: 经验生命周期管理与数据服务。
- **EventBus**: 异步事件分发器。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `tests/` 下与 `memory_operations` 及 `ExperienceService` 相关的测试中，移除全部关于 `bootstrap.get_context` 的全局 patch。
- **SC-002**: 架构分层检查 (`make harness-check`) 完全通过，无任何跨层或逆向依赖报警。
- **SC-003**: `ExperienceService` 单测隔离度提升，测试运行时间或稳定性有所改善（无隐式事件副产物）。

## Assumptions

- 现有 MCP server.py 和 Web app 已经能够正常获取到 AppContext。
- 不引入新的第三方 IoC 框架，而是采用简单的构造函数注入（Constructor Injection）和方法参数传递。