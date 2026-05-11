# Implementation Plan: Instant Feedback

**Spec**: spec.md
**Status**: Ready
**Steps**: 3

### Step 1: process_before_prompt returns context field
- File: src/team_memory/daemon/pipeline.py
- Action:
  - 在 return 前格式化 context:
    ```
    parts = []
    if results: parts = [f"Found {len(results)} relevant:", *titles]
    if persona_context: parts.append(persona_context)
    context = "\n".join(parts)[:2000]
    ```
  - 返回新增 "context" 字段
- Test: test_process_before_prompt_returns_context

### Step 2: Hook scripts inject context
- Files: hooks/cursor_before_prompt.py, hooks/hermes_user_prompt_submit.py
- Action: 读 response["context"] → 注入到输出中
- Test: 手动 (真实 hook 调用)

### Step 3: Verify
- make e2e-verify → 检查 daemon 日志显示 context
