# Feature Specification: Hook One-Click Setup

**Created**: 2026-05-11
**Updated**: 2026-05-11 (added OpenClaw assessment)
**Status**: Draft
**Input**: tm-cli setup --platform all — one-click configure all AI client hooks

## Current State (verified)

### Claude Code
| Item | Status |
|------|--------|
| hooks.json / settings.json | all 4 hooks (hand-configured) |
| _setup_claude_code | writes all 4 ✅ |
| _check_platform | only checks PostToolUse ❌ |

### Cursor
| Item | Status |
|------|--------|
| hooks.json | all 4 hooks (hand-configured) |
| _setup_cursor | only writes afterMCPExecution ❌ |
| _check_platform | only checks afterMCPExecution ❌ |

### Hermes
| Item | Status |
|------|--------|
| config.yaml | 3 hooks (hand-configured) |
| _setup_hermes | only prints instructions ❌ |
| _check_platform | only checks file exists ❌ |

### OpenClaw
| Item | Status |
|------|--------|
| Hook system | Gateway Event Hooks (HOOK.yaml + handler.py) |
| TM hooks | none ❌ |
| _setup_openclaw | doesn't exist ❌ |
| _check_platform | doesn't exist ❌ |

Key insight: OpenClaw uses a DIFFERENT hook system than Claude/Cursor/Hermes.
It uses Gateway Event Hooks (Python async handlers in ~/.openclaw/hooks/)
with events: agent:start (user prompt) and agent:end (assistant response).
This requires creating a handler.py, not a shell script.

OpenClaw is P2 scope — requires different architecture.
P0 scope is Claude + Cursor + Hermes (shell hooks).

## User Scenarios

### P1: New user runs tm-cli setup --platform all
- As a new TeamMemory user
- I want to run one command to configure all hooks
- Acceptance Criteria:
  1. _setup_cursor writes all 4 Cursor hooks
  2. _setup_hermes auto-writes config.yaml
  3. _setup_claude_code already correct
  4. _check_platform shows all-green for claude+cursor+hermes

### P2: Existing user re-runs setup — no breakage
- Acceptance Criteria:
  1. Old entries replaced, not duplicated
  2. Complete config stays complete

### P3: tm-cli setup --check shows real status
- Acceptance Criteria:
  1. Claude: checks all 4 hook types
  2. Cursor: checks all 4 hook types
  3. Hermes: checks all 3 hook types
  4. Also checks hook scripts exist on disk

## Technical Requirements

### Data Flow
tm-cli setup --platform all
  -> Claude: _setup_claude_code (no change needed)
  -> Cursor: _setup_cursor (fix: add 3 missing hooks)
  -> Hermes: _setup_hermes (fix: auto-write YAML)

tm-cli setup --check
  -> _check_platform("claude-code") (fix: check 4 hooks)  
  -> _check_platform("cursor") (fix: check 4 hooks)
  -> _check_platform("hermes") (fix: check 3 hooks)

### Hook Type Reference
| Platform | Hook event | Script / handler |
|----------|-----------|-----------------|
| Claude | SessionStart | hooks/claude_session_start.py |
| Claude | UserPromptSubmit | hooks/claude_user_prompt_submit.py |
| Claude | Stop | hooks/claude_stop.py |
| Claude | PostToolUse | hooks/claude_post_tool.py |
| Cursor | sessionStart | hooks/cursor_session_start.py |
| Cursor | beforeSubmitPrompt | hooks/cursor_before_prompt.py |
| Cursor | afterAgentResponse | hooks/cursor_after_response.py |
| Cursor | afterMCPExecution | hooks/cursor_after_mcp.py |
| Hermes | UserPromptSubmit | hooks/hermes_user_prompt_submit.py |
| Hermes | Stop | hooks/hermes_stop.py |
| Hermes | PostToolUse | hooks/hermes_post_tool_call.py |

### OpenClaw (P2 — future)
| Platform | Hook event | Handler |
|----------|-----------|---------|
| OpenClaw | agent:start | ~/.openclaw/hooks/tm-pipeline/handler.py |
| OpenClaw | agent:end | ~/.openclaw/hooks/tm-pipeline/handler.py |

## Edge Cases
1. Config file doesn't exist → create from scratch
2. Config file corrupted → show error, suggest backup
3. Hook scripts not found on disk → show warning
4. Duplicate entries → dedup before writing

## Out of Scope (this iteration)
- OpenClaw hook integration (requires Gateway Event Hooks, different system)
- Installing Python dependencies
- Setting up the daemon itself
- Webhook integration
