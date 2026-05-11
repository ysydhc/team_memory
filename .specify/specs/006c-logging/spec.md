# Feature Specification: Structured Logging

**Created**: 2026-05-14
**Status**: Draft
**Input**: 日志改为结构化 JSON, 关键节点带 session_id + source, make logs 过滤查看

## Clarified
- 保持 /tmp/tm-daemon.log
- JSON 格式: {"ts":"...","level":"INFO","event":"...","session":"...","source":"..."}
- make logs SESSION=xxx SOURCE=cursor LEVEL=ERROR 过滤
- Web 日志不动

## User Scenarios

### P1: 按 session_id 追踪全链路
- As 开发者排查问题
- I want to make logs SESSION=abc123
- So that 看到该 session 从 hook → refine → publish 的完整日志
- Acceptance Criteria:
  1. before_prompt / after_response / convergence / refinement / publish 日志都带 session_id
  2. make logs SESSION=xxx 只输出该 session 的日志行
  3. 能看到每一步的 timestamp 和耗时

### P2: 按 source + level 过滤
- As 开发者
- I want to make logs SOURCE=cursor LEVEL=ERROR
- So that 只看 Cursor 的错误
- Acceptance Criteria:
  1. 每条日志带 source 字段 (claude/cursor/hermes/unknown)
  2. make logs SOURCE=xxx 只输出该 source 的日志
  3. make logs LEVEL=ERROR 只输出 ERROR 及以上

### P3: 日志格式向后兼容
- As 用户
- I want to 旧的纯文本日志还能看
- So that 不破坏现有工具链
- Acceptance Criteria:
  1. 改造后 tail /tmp/tm-daemon.log 仍然可读 (非 JSON 行保持原样)
  2. JSON 行能被 make logs 正确解析和过滤

## Technical Requirements

### 日志改造范围 (只改关键节点)
| 事件 | 新增字段 |
|------|---------|
| before_prompt | session, source, project, query |
| after_response | session, source, project, convergence |
| convergence | session, source, msg_count |
| refine_start/done/fail | session, project, latency |
| publish | session, experience_id |
| memory_extract | session, persona, count |
| startup/shutdown | — (保持原样) |

### StructuredLogger 类
```python
class StructuredLogger:
    def __init__(self, logger, source=None):
        self._logger = logger
        self._source = source  # 默认 source

    def info(self, event, **kwargs):
        msg = {"event": event, **kwargs}
        if self._source: msg["source"] = self._source
        self._logger.info(json.dumps(msg, ensure_ascii=False))
```

### make logs 实现
```bash
make logs                      # tail -50 + 彩色输出 (JSON → 对齐格式)
make logs SESSION=abc123       # python -c 过滤 json
make logs SOURCE=cursor
make logs LEVEL=ERROR
make logs --tail 200
```
实现: scripts/filter-logs.py (读 stdin, 解析 JSON/文本行, 按条件输出)

## Edge Cases
- session_id 缺失 → "unknown"
- source 无法识别 → "unknown"
- 旧日志行非 JSON → 原样输出, make logs 跳过

## Out of Scope
- 日志压缩归档
- Web 日志改造
- 日志发送到外部系统 (ELK/Splunk)
