# Implementation Plan: Hook One-Click Setup

**Spec**: spec.md
**Status**: Ready for review

## Architecture

### Platforms and Hook Systems
| Platform | Hook system | Setup method |
|----------|------------|--------------|
| Claude | Shell hooks (settings.json) | JSON merge |
| Cursor | Shell hooks (hooks.json) | JSON merge |
| Hermes | Shell hooks (config.yaml) | YAML merge |
| OpenClaw | Gateway Event Hooks (HOOK.yaml + handler.py) | P2 — separate architecture |

### Modules Involved
| Module | File | Role |
|--------|------|------|
| CLI setup | src/team_memory/cli.py | cmd_setup, _setup_*, _check_platform |
| Hook scripts | src/team_memory/hooks/ | Scripts that hooks execute |

### Data Flow
```
tm-cli setup --platform all
  |
  +-> _setup_claude_code(venv_python)     ✅ already complete
  +-> _setup_cursor(venv_python)          ❌ FIX: add 3 missing hooks
  +-> _setup_hermes(venv_python)          ❌ FIX: auto-write YAML

tm-cli setup --check
  |
  +-> _check_platform("claude-code")      ❌ FIX: check all 4
  +-> _check_platform("cursor")           ❌ FIX: check all 4
  +-> _check_platform("hermes")           ❌ FIX: check all 3
```

## Implementation Steps

### Step 1: Fix _setup_cursor — add 3 missing hooks
- File: src/team_memory/cli.py
- Change: Add sessionStart, beforeSubmitPrompt, afterAgentResponse entries
- Pattern: Same as _setup_claude_code — hooks[eventName] = [{"matcher": "", "hooks": [{"type": "command", "command": "..."}]}]
- Tests: test_setup_cursor_writes_all_four, test_setup_cursor_no_duplicates

### Step 2: Fix _setup_hermes — auto-write config.yaml
- File: src/team_memory/cli.py
- Change: Replace print() with yaml.safe_load → merge hooks → yaml.dump
- Tests: test_setup_hermes_writes_config, test_setup_hermes_preserves_existing

### Step 3: Fix _check_platform("claude-code")
- Change: Check SessionStart + UserPromptSubmit + Stop + PostToolUse (not just PostToolUse)
- Also check hook scripts exist on disk

### Step 4: Fix _check_platform("cursor")  
- Change: Check sessionStart + beforeSubmitPrompt + afterAgentResponse + afterMCPExecution
- Also check hook scripts exist on disk

### Step 5: Fix _check_platform("hermes")
- Change: Check UserPromptSubmit + Stop + PostToolUse in config.yaml
- Also check hook scripts exist on disk

### Step 6: Verify end-to-end
- Run make e2e-verify after setup

## Test Strategy
- Unit: Mock file I/O, verify correct structure written
- Integration: Actually run setup in temp dir, read back files
- Edge cases: missing file → create, corrupted → error, no scripts → warning, duplicate → dedup

## Risks
- YAML dump formatting → test with real config.yaml
- Running setup twice creates duplicates → remove old entries first

## OpenClaw — P2 Future Scope
- Requires creating ~/.openclaw/hooks/tm-pipeline/
  - HOOK.yaml: subscribe to agent:start + agent:end
  - handler.py: forward to daemon /hooks/before_prompt and /hooks/after_response
- Separate architecture from shell hooks — not in this iteration
