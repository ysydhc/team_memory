# Implementation Plan: Structured Logging

**Spec**: spec.md
**Status**: Ready
**Steps**: 4

## Architecture

### New Modules
| Module | File | Role |
|--------|------|------|
| StructuredLogger | daemon/structured_logger.py | 包装 logger, 自动注入 session_id + source, JSON 输出 |
| filter-logs | scripts/filter-logs.py | 读 stdin JSON/文本行, 按条件过滤, 对齐输出 |

### Modified Modules
| Module | Change |
|--------|--------|
| pipeline.py | before_prompt/after_response 改用 StructuredLogger |
| refinement_worker.py | refine/publish 日志改用 StructuredLogger |
| app.py | 创建 StructuredLogger, 注入到各模块; 替换裸 logger |
| Makefile | 加 make logs target |

### Data Flow
```
app.py 启动 → StructuredLogger(logger, source="daemon")
  ↓
pipeline: log.info("before_prompt", session=..., source=detected, ...)
  ↓
/tmp/tm-daemon.log: {"event":"before_prompt","session":"abc","source":"cursor",...}

make logs SESSION=abc:
  → scripts/filter-logs.py --session abc < /tmp/tm-daemon.log
  → 输出对齐的行
```

## Implementation Steps

### Step 1: Create StructuredLogger
- File: src/team_memory/daemon/structured_logger.py (new)
- API:
  - info(event, **kwargs) → json.dumps + source注入
  - warning/error/debug 同理
  - 自动跳过 source=None 的场景
- Test: test_structured_logger_output_is_json

### Step 2: Apply StructuredLogger to key log points
- Files: pipeline.py, refinement_worker.py
- 替换 before_prompt/after_response/convergence/refine/publish 的 logger 调用
- 不改 debug 日志和异常堆栈

### Step 3: Create filter-logs.py + make logs
- File: scripts/filter-logs.py
- 功能: --session / --source / --level / --tail
- Makefile: make logs target

### Step 4: Verify
- ruff + pytest + make e2e-verify + make logs
