# Feature Specification: Instant Feedback

**Created**: 2026-05-13
**Status**: Draft
**Input**: before_prompt 返回时包含 recall 结果摘要，让用户第一时间看到 TM 找到了什么。

## Current State

before_prompt 已经调了 recall，返回 `{"results": [...], "persona_context": "..."}`。
但是:
  - Hook 脚本不消费 persona_context
  - 用户看不到 "Found 3 experiences"
  - process_before_prompt 返回的 results 只是 list，没有格式化摘要

## User Scenarios

### P1: before_prompt response includes formatted recall summary
- As an AI developer
- I want my hook to receive a `context` field with formatted recall results
- So my AI client immediately sees relevant experiences
- Acceptance Criteria:
  1. process_before_prompt 返回新增 `context` 字段 (formatted markdown string)
  2. `context` 包含 recall results + persona_context 合并
  3. Hook 脚本 (cursor_before_prompt, hermes_user_prompt_submit) 将 context 注入 AI 上下文中
  4. 用户对话中立刻可以看到 TM 记忆

### P2: CLI shows instant feedback
- As a user
- I want `tm-cli` or `make e2e-verify` to show recall results upfront
- Acceptance Criteria:
  1. make diagnose-pipeline 显示 "Last recall: N results"
  2. Hook debug logs show recall count at INFO level

## Technical Requirements

### Data Flow
```
before_prompt
  → sink.recall(query, project)          → results list
  → personal_memory.build_profile()      → persona_context
  → format_context(results, persona)     → context string (NEW)
  → return {"results": ..., "context": ..., "persona_context": ...}

hook script (cursor/hermes)
  → reads response["context"]
  → injects into AI prompt or prepends
  → AI sees the knowledge immediately
```

### Changes
1. process_before_prompt: 加 `context` 字段 (合并 recall + persona)
2. Hook scripts: 读取并注入 context (cursor_before_prompt.py, hermes_user_prompt_submit.py)

## Edge Cases
- No recall results → context = persona_context only
- context 太长 → 截断到 2000 chars
- Hook script 不支持 context → 向后兼容 (不报错)
