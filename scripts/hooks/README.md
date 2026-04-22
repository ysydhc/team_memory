# Agent Memory Pipeline — Hook Scripts

This directory contains hook scripts that integrate Cursor and Claude Code with
the TeamMemory (TM) pipeline. Hooks fire at key points in the agent lifecycle,
enabling automatic context retrieval, keyword-triggered recall, and draft
capture for later publishing.

## Hook Event Mapping

| Lifecycle Event        | Cursor Hook Name      | Claude Code Hook Name | Script                              |
|------------------------|-----------------------|-----------------------|--------------------------------------|
| Session starts         | `sessionStart`        | `SessionStart`        | `cursor_session_start.py` / `claude_session_start.py` |
| User submits a prompt  | `beforeSubmitPrompt`  | `PreToolUse`          | `cursor_before_prompt.py`            |
| Agent finishes reply   | `afterAgentResponse`  | `PostToolUse`         | `cursor_after_response.py`           |

## What Each Hook Does

### sessionStart / SessionStart

- **Script**: `cursor_session_start.py` (Cursor), `claude_session_start.py` (Claude Code)
- **Input**: `{"workspace_roots": [...], "conversation_id": "..."}`
- **Behavior**: When a new agent session starts, the hook resolves the project
  from the workspace path using `config.yaml` patterns, then calls the
  `memory_context` MCP tool to retrieve relevant past experiences. The result
  is returned as `additionalContext` so the agent starts with prior knowledge.
- **Output**: `{"additionalContext": "<retrieved context>"}`

### beforeSubmitPrompt / PreToolUse

- **Script**: `cursor_before_prompt.py`
- **Input**: `{"prompt": "user text", "workspace_roots": [...], "conversation_id": "..."}`
- **Behavior**: Scans the user's prompt for retrieval trigger keywords (defined
  in `config.yaml` under `retrieval.keyword_triggers`, e.g. "之前", "上次",
  "经验"). If triggered, queries the `memory_recall` MCP tool and injects
  results as `additionalContext`. Silently fails so the user is never blocked.
- **Output**: `{"additionalContext": "<retrieved context>"}` if triggered, `{}` otherwise

### afterAgentResponse / PostToolUse

- **Script**: `cursor_after_response.py`
- **Input**: `{"conversation_id": "...", "prompt": "...", "workspace_roots": [...]}`
- **Behavior**: After the agent replies, the hook runs convergence detection on
  the conversation. It also creates or updates a draft in the local DraftBuffer
  (SQLite). If convergence is detected, the draft is marked for publishing.
- **Output**: `{"status": "ok", "convergence": bool, "draft_id": "..."}`

## Supporting Modules

| File                     | Purpose                                                  |
|--------------------------|----------------------------------------------------------|
| `common.py`              | Shared utilities: stdin parsing, config loading, MCP calls |
| `config.yaml`            | Pipeline configuration (TM base URL, project patterns, triggers) |
| `retrieval_trigger.py`   | Keyword-based trigger detection for beforeSubmitPrompt    |
| `convergence_detector.py`| Detects when a conversation has converged on a solution   |
| `draft_buffer.py`        | SQLite-backed draft storage with async interface          |

## Installation

### Cursor

Copy the hooks configuration to the project root:

```bash
# The template is already in the repo at .cursor/hooks.json
# If you need to set it up for a different project, copy and edit:
cp /Users/yeshouyou/Work/agent/team_doc/.cursor/hooks.json /path/to/your/project/.cursor/hooks.json
```

Make sure the `command` paths inside `.cursor/hooks.json` point to the correct
location of the hook scripts on your machine.

### Claude Code

Copy the Claude Code hooks template to the user-level hooks config:

```bash
# Claude Code reads hooks from ~/.claude/hooks.json
mkdir -p ~/.claude
cp /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/claude_hooks_template.json ~/.claude/hooks.json
```

Edit `~/.claude/hooks.json` if you need to adjust script paths for your setup.

## Testing Hooks Manually

Each hook script reads JSON from stdin and writes JSON to stdout. You can test
them from the command line:

```bash
# Test sessionStart hook (Cursor)
echo '{"workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"], "conversation_id": "test-123"}' \
  | python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_session_start.py

# Test beforeSubmitPrompt hook — with a trigger keyword
echo '{"prompt": "之前遇到过类似的问题吗？", "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"]}' \
  | python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_before_prompt.py

# Test beforeSubmitPrompt hook — without a trigger keyword (should return {})
echo '{"prompt": "Hello world", "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"]}' \
  | python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_before_prompt.py

# Test afterAgentResponse hook
echo '{"conversation_id": "test-123", "prompt": "Here is the solution...", "workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"]}' \
  | python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/cursor_after_response.py

# Test Claude Code sessionStart hook
echo '{"workspace_roots": ["/Users/yeshouyou/Work/agent/team_doc"], "conversation_id": "test-456"}' \
  | python3 /Users/yeshouyou/Work/agent/team_doc/scripts/hooks/claude_session_start.py
```

**Note**: The MCP-dependent hooks (`sessionStart`, `beforeSubmitPrompt`) require
TeamMemory to be running at the URL specified in `config.yaml`
(default: `http://localhost:3900`). If TM is not running, these hooks will
return an error status but will not block the agent.

## Current Status: Phase 0

These hooks are in **Phase 0** — they log data and exercise the pipeline
locally but do **not** yet write to TeamMemory in production. Specifically:

- `sessionStart` and `beforeSubmitPrompt` call MCP tools (`memory_context`,
  `memory_recall`) which return real results when TM is running.
- `afterAgentResponse` writes to a local SQLite DraftBuffer but does not
  publish drafts to TM yet.
- All hooks handle errors gracefully and will not block or crash the agent.

Next phases will add:
- **Phase 1**: Automatic draft publishing when convergence is detected
- **Phase 2**: Full bidirectional sync (read + write) with TeamMemory
