# Agent 共享记忆系统 — 总计划

## 项目背景

在 Hermes / Claude Code / Cursor 三个 Agent 之间建立共享记忆系统。

核心设计（v9）：
- **两类记忆**：痕迹（Agent 用，人不看）→ TM Experience；知识（人要学要看）→ Markdown + Obsidian
- **Hooks 驱动**：通过 Cursor/Claude Code 的 Hooks API 自动触发捕获和检索，不依赖 Agent "想起来"
- **草稿缓冲**：对话中的发现先进草稿（TM draft Experience），等收敛信号后提炼写入
- **自动提升**：Janitor 后台任务把被反复使用的 L2 痕迹自动提升为 L3 知识
- **Git 索引**：Obsidian 仓库通过 Git staged/committed 驱动 TM 索引
- **评估闭环**：检索结果注入 [mem:xxx] 标记，系统自动观察 Agent 是否使用

详细设计见对话历史中的系统描述 v9。

## 阶段总览

| 阶段 | 名称 | 核心目标 | 依赖 | 预计子任务数 |
|------|------|---------|------|------------|
| 0 | 基础准备 | 搭建 Hook 脚本框架，不改 TM | 无 | 5 |
| 1 | TM 内核扩展 | TM 支持管线模式写入和检索 | 无 | 8 |
| 2 | 管线实现 | Hook 跑通草稿→写入全流程 | 阶段 1 | 6 |
| 3 | Obsidian + Git 索引 | 文件通过 Git 驱动进入 TM | 阶段 1 | 5 |
| 4 | 提升 + 评估闭环 | 痕迹自动提升 + 评估生效 | 阶段 1+2+3 | 6 |

阶段 0 和阶段 1 可以并行执行。

## 工作流程

```
主 Agent（Hermes）
  ├── 拆分阶段为子任务
  ├── 通过 delegate_task 派发给 subagent
  ├── subagent 按 TDD 执行：写测试 → 实现 → 验证
  ├── subagent 返回验收结果
  ├── 主 Agent 评估结果
  │   ├── 满足 → 标记完成
  │   └── 不满足 → 派发修改任务
  └── 阶段全部完成后 → 输出人工验收条目
```

## 文件索引

- `plan-phase0.md` — 基础准备
- `plan-phase1.md` — TM 内核扩展
- `plan-phase2.md` — 管线实现
- `plan-phase3.md` — Obsidian + Git 索引
- `plan-phase4.md` — 提升 + 评估闭环
