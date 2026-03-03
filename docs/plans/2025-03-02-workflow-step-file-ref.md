# Workflow Step File Reference 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 支持 workflow step 独立成文件、主 YAML 通过 $ref 引用组合，后端在加载时解析 $ref 返回完整 step，无需 Agent 或 Cursor 做任何改动。

**Architecture:** 在 workflow_oracle 的 `_load_workflow` 中增加 `_resolve_refs` 逻辑；step 的 $ref 以相对主 workflow 文件路径解析；保持 `get_next_step` 返回结构与现有一致（action 为完整文本）。

**Tech Stack:** Python, PyYAML, Path (pathlib)

---

## 已完成任务

- [x] 前置：更新执行计划阶段六
- [x] Task 1: $ref 解析函数（_resolve_step_ref + _load_workflow）
- [x] Task 2: 兼容无 $ref 的现有 workflow
- [x] Task 3: 拆分 task-execution-workflow 的 steps
- [x] Task 4: 集成测试与文档

---

## 目录结构

```
.cursor/plans/workflows/
├── task-execution-workflow.yaml   # 主 workflow，steps 用 $ref 引用
└── steps/task-execution/
    ├── step-coldstart.yaml
    ├── step-claim.yaml
    ├── step-execute.yaml
    ├── step-complete.yaml
    ├── step-verify.yaml
    └── step-retro.yaml
```

---

## 不在此计划范围内的内容

- 原子动作库（actions/）及 step 内 `actions: [$ref, inline]`：阶段六 6.3，后续排期
- workflow-optimization-workflow 的 step 拆分：可复用同机制，单独任务
- 原阶段六 6.4～6.6（可选元数据、任务验收字段）：保持为独立子项
