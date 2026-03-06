# 反馈回路机制

> Agent 出错时记录根因、修复与规则更新，形成闭环，避免同类问题重复发生。

---

## 一、机制概述

当 Agent 执行任务过程中遇到错误、修复或可复用的排查路径时，应通过 **反馈回路** 将经验沉淀到团队经验库（team_memory），便于后续检索与复用。

**核心流程**：出错 → 定位根因 → 修复 → 沉淀经验（建议 `tm_save`）→ 必要时更新规则。

---

## 二、沉淀方式

| 场景 | 推荐工具 | experience_type |
|------|----------|-----------------|
| Bug 修复、单点经验 | `tm_save` | bugfix / best_practice |
| 长对话、多步骤复盘 | `tm_learn` | 按内容 |
| 组级任务复盘 | `tm_save_group` | tech_design |

**建议**：单次出错与修复优先用 `tm_save`，结构清晰、检索高效。

---

## 三、沉淀内容要求

1. **根因**：问题产生的直接原因或触发条件。
2. **修复**：具体修复步骤、代码/配置变更、验证方式。
3. **规则更新**（若适用）：若需固化到 `.cursor/rules/` 或文档，说明更新点与理由。
4. **自包含**：若改动涉及未被 git 追踪的路径，须将关键内容写入 `solution` 或 `code_snippets`，不得仅写文件路径。

---

## 四、示例

### 示例 1：Web 静态缓存导致旧 JS 生效

**问题**：修改前端 JS 后，页面仍加载旧逻辑。

**根因**：浏览器或 CDN 缓存了旧版静态资源，未强制刷新。

**修复**：部署/开发时禁用静态资源缓存，或在构建产物中加入版本号/hash；本地调试时使用硬刷新（Ctrl+Shift+R）或禁用缓存。

**沉淀**（`tm_save`）：

```
title: Web 前端修改后需禁用静态缓存避免旧 JS 生效
problem: 修改前端 JS 后页面仍执行旧逻辑
solution: 部署时禁用静态缓存或为资源加版本号；本地调试用硬刷新或禁用缓存
experience_type: best_practice
tags: ["web", "cache", "frontend"]
```

---

### 示例 2：ruff 检查未通过导致提交失败

**问题**：本地 `git commit` 后 CI 报 ruff 检查失败。

**根因**：提交前未执行 `ruff check src/`，本地与 CI 环境不一致。

**修复**：提交前必须执行 `ruff check src/`（及 `tests/` 若涉及测试），通过后再提交。

**沉淀**（`tm_save`）：

```
title: Python 代码改动后必须通过 ruff check 再提交
problem: 未在提交前执行 ruff 检查，CI 报错
solution: 提交前执行 ruff check src/ 及 tests/（若涉及），通过后再提交
experience_type: best_practice
tags: ["ruff", "ci", "quality"]
```

---

## 五、与现有规则衔接

| 规则 | 衔接点 |
|------|--------|
| tm-extraction-retrieval | Bug 修复、故障排查等场景触发经验提取 |
| harness-engineering | 反馈回路为「出错时沉淀」的细化设计 |
| tm-core | 类型勿用 general；自包含写入 solution/code_snippets |
