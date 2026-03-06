# 经验提取与检索触发 — 设计文档

> 日期：2025-03-05  
> 状态：已落地

## 一、目标

通过 prompt/rule 引导 Agent：
1. **何时**从对话、计划、任务复盘中提取经验或记忆并写入 tm
2. **何时**从 tm 检索相关经验以辅助执行

## 二、概念区分

| 概念 | 含义 | 存储 |
|------|------|------|
| **经验** | 项目相关、结构化沉淀 | tm_save / tm_learn / tm_save_group |
| **记忆** | 个人偏好、表达习惯 | tm_save（tags 含 user_preference） |

## 三、调研结论摘要

基于 3 个 agent 并行调研（GitHub、AI 论坛、2024-2025 记忆领域）：

- **保存触发**：事件 + 时间 + 计数混合；Agent 工具化决策；写比读更谨慎
- **检索触发**：新会话预加载；任务相关时检索；按需 vs Always-On 按场景选择
- **原则**：选择性整合、分层存储、可解释可回滚

## 四、落地产物

| 产物 | 路径 |
|------|------|
| 规则 | `.cursor/rules/tm-extraction-retrieval.mdc` |
| 快速参考 prompt | `.cursor/prompts/experience-extraction-trigger.md` |
| 规则索引 | `.cursor/rules/team_memory.mdc`（已追加 tm-extraction-retrieval） |
| 调研报告 | `.debug/docs/analysis/ai-agent-memory-trigger-research.md` 等 |

## 五、与现有工作流关系

- 本设计**不依赖** tm_task 工作流，仅配合当前经验/记忆系统
- 与 tm_preflight、tm-core 推荐工作流、task-sediment-extract 衔接
- 工作流相关触发由后续单独调整
