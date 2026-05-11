# Tasks: Hook One-Click Setup

**Plan**: plan.md
**Total tasks**: 9 (was 7, split T5 into 3 sub-tasks)

## Task 1: TDD — _setup_cursor test (FAIL first)
- File: tests/test_cli_config_setup.py (append to existing)
- Action: Write `test_setup_cursor_writes_all_four_hooks`
  1. Mock file I/O (open, json.dump) 
  2. Call _setup_cursor with mock cfg + venv_python
  3. Assert hooks.json has: sessionStart, beforeSubmitPrompt, afterAgentResponse, afterMCPExecution
- Result: Test FAILS — only afterMCPExecution written ❌
- Status: [x] done

## Task 2: Fix _setup_cursor — add 3 missing hooks
- File: src/team_memory/cli.py, function _setup_cursor (line ~822)
- Action:
  1. Keep existing afterMCPExecution logic
  2. Add sessionStart → cursor_session_start.py
  3. Add beforeSubmitPrompt → cursor_before_prompt.py
  4. Add afterAgentResponse → cursor_after_response.py
  5. Each: hooks_section[event] = [{"matcher": "", "hooks": [{"type": "command", "command": venv_python -m ...}]}]
  6. Existing entry dedup: remove old entries with "team_memory" matcher before adding
- Result: T1 test passes ✅
- Status: [x] done

## Task 3: TDD — _setup_hermes test (FAIL first)
- File: tests/test_cli_config_setup.py
- Action: Write `test_setup_hermes_writes_yaml`
  1. Mock file I/O for YAML read/write
  2. Call _setup_hermes with mock cfg + venv_python
  3. Assert config.yaml hooks section has: UserPromptSubmit, Stop, PostToolUse
  4. Assert non-hooks config preserved
- Result: Test FAILS — only prints, no file written ❌
- Status: [x] done

## Task 4: Fix _setup_hermes — auto-write config.yaml
- File: src/team_memory/cli.py, function _setup_hermes (line ~904)
- Action:
  1. yaml.safe_load existing ~/.hermes/config.yaml (or empty dict)
  2. Build hooks dict:
     - UserPromptSubmit → hermes_user_prompt_submit.py
     - Stop → hermes_stop.py
     - PostToolUse → hermes_post_tool_call.py (matcher: "recall|memory_recall")
  3. Merge into config (hooks key replaces existing TM hooks)
  4. yaml.dump back with allow_unicode=True
  5. Print: "[hermes] Written 3 hooks to config.yaml"
- Result: T3 test passes ✅
- Status: [x] done

## Task 5: Fix _check_platform("claude-code")
- File: src/team_memory/cli.py, _check_platform branch at line ~756
- Action: Replace "only check PostToolUse" with full check:
  1. Check SessionStart entry + claude_session_start.py exists
  2. Check UserPromptSubmit entry + claude_user_prompt_submit.py exists
  3. Check Stop entry + claude_stop.py exists
  4. Check PostToolUse entry + claude_post_tool.py exists (keep existing check)
  5. Print per-hook: green_check / red_x
- Tests: test_check_claude_all_four_hooks
- Status: [x] done

## Task 6: Fix _check_platform("cursor")
- File: src/team_memory/cli.py, _check_platform branch at line ~758
- Action: Replace "only check afterMCPExecution" with:
  1. Check sessionStart entry + cursor_session_start.py exists
  2. Check beforeSubmitPrompt entry + cursor_before_prompt.py exists
  3. Check afterAgentResponse entry + cursor_after_response.py exists
  4. Check afterMCPExecution entry + cursor_after_mcp.py exists (keep existing)
  5. Print per-hook: green_check / red_x
- Tests: test_check_cursor_all_four_hooks
- Status: [x] done

## Task 7: Fix _check_platform("hermes")
- File: src/team_memory/cli.py, _check_platform branch at line ~760
- Action: Replace "only check file exists" with:
  1. Check UserPromptSubmit entry + hermes_user_prompt_submit.py exists
  2. Check Stop entry + hermes_stop.py exists
  3. Check PostToolUse entry + hermes_post_tool_call.py exists
  4. Print per-hook: green_check / red_x
- Tests: test_check_hermes_all_three_hooks
- Status: [x] done

## Task 8: Run full verification
- Action:
  1. ruff check src/team_memory/cli.py tests/test_cli_config_setup.py
  2. pytest tests/test_cli_config_setup.py -v
  3. Full test suite: pytest -m "not integration" -q
  4. tm-cli setup --check (should show all green)
  5. make e2e-verify (should pass)
- Status: [x] done

## Task 9: Commit + update roadmap
- Action:
  1. git add -A && git commit
  2. Mark P0-1 [x] done in roadmap.md
- Status: [x] done
