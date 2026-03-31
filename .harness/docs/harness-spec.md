# Harness Framework Spec

> 可移植的 Agent Harness 框架规范。项目无关，适用于任何使用 AI 编码 Agent 的项目。

## 核心原则

1. **Human steers, Agent executes** — 人决定做什么和为什么，Agent 决定怎么做
2. **Enforce over advise** — 能用 Hook 强制的不用文字建议（Mechanical Enforcement）
3. **Plan before execute** — 非平凡任务先签契约再动手
4. **File-backed state** — 状态写文件（progress.md），不依赖会话记忆
5. **Context is precious** — 主动管理 context window，防止 context rot
6. **Interface IS the harness** — 工具接口设计 > 提示词工程（ACI 原则）

## 三层架构

```
┌─────────────────────────┐
│   Project Extension     │  harness-config.yaml（项目特有配置）
├─────────────────────────┤
│   Runtime Adapter       │  .claude/ · .cursor/（平台翻译层）
├─────────────────────────┤
│   Harness Core          │  .harness/（跨平台、可移植）
└─────────────────────────┘
```

- **Harness Core** (`.harness/`)：编排流程、契约模板、失败分类、Hook 脚本、计划管理
- **Runtime Adapter** (`.claude/` / `.cursor/`)：将 Core 翻译为平台能理解的格式
- **Project Extension** (`harness-config.yaml`)：质量门禁、分层约束、超时策略等

## 文件导航

| 文件 | 用途 |
|------|------|
| `harness-config.yaml` | 项目配置（质量门禁、超时、context、安全） |
| `orchestration/task-flow.md` | Phase 0-4 编排流程 |
| `orchestration/context-management.md` | Context rot 管理（4 层策略） |
| `orchestration/contracts/plan-contract.md` | Plan 启动契约模板 |
| `orchestration/contracts/task-contract.md` | Task 执行契约模板 |
| `orchestration/extensions/` | 编排扩展点 |
| `failure/failure-taxonomy.md` | 三级恢复 + 行为异常 + 超时 + 安全分级 |
| `hooks/` | 共享 Hook 脚本（两个平台引用同一份） |
| `plans/progress.md` | 当前活跃计划状态 |
| `plans/completed/` | 归档目录 |
| `docs/migration-guide.md` | 迁移到新项目指南 |

## 编排流程概要

```
Phase 0: Context Loading → Phase 1: Planning → Phase 2: Execution → Phase 3: Verification → Phase 4: Closure
```

每个 Phase 详见 `orchestration/task-flow.md`。

## 失败恢复概要

```
Level 1: 自动恢复（Hook）  → lint、format、编译重试
Level 2: Agent 判断恢复    → 测试失败、类型错误、依赖缺失
Level 3: 人工介入          → 架构冲突、外部服务、scope 超出
```

详见 `failure/failure-taxonomy.md`。

## 迁移到新项目

1. 复制 `.harness/` 到新项目
2. 修改 `harness-config.yaml`
3. 创建平台入口文件（CLAUDE.md / AGENTS.md）

详见 `docs/migration-guide.md`。
