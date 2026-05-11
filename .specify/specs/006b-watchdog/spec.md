# Feature Specification: Watchdog + Alerting

**Created**: 2026-05-14
**Status**: Draft
**Input**: 每次 tick 检查系统健康, 异常飞书告警 (首次→10min→20min→40min backoff, 恢复通知)

## Clarified
- 每次 tick (30s) 附带检查
- 告警: 首次立即 → 10min → 20min → 40min backoff
- 恢复: 飞书通知 "已恢复"

## User Scenarios

### P1: 异常自动发现
- As 系统管理员
- I want to 系统自动检测 PG/SQLite/worker 异常
- So that 我不用手动盯着 /tmp/tm-daemon.log
- Acceptance Criteria:
  1. worker task 意外退出 → 飞书告警
  2. SQLite 文件 > 500MB → 飞书告警
  3. PG ping 超时 > 3s → 飞书告警
  4. 异常持续 → backoff 告警 (10m→20m→40m)

### P2: 恢复通知
- As 系统管理员
- I want to 异常恢复时收到通知
- So that 我知道问题已解决
- Acceptance Criteria:
  1. PG 恢复 → 飞书 "✅ PG 已恢复 (中断 5 分钟)"
  2. worker task 恢复 → 飞书 "✅ worker 已重启"

## Technical Requirements

### Watchdog 类
```python
class Watchdog:
    def __init__(self, db_path, pg_pool, worker_task):
        self._checks = {...}
        self._alert_state = {}  # name → {last_alert, backoff_level}

    async def check(self) -> list[str]:  # 返回异常列表
    def _should_alert(self, name) -> bool:  # backoff 计算
    async def _send_alert(self, name, detail):  # 飞书推送
    async def _send_recovery(self, name, duration):  # 恢复通知
```

### 检查项
| 检查 | 异常条件 | 告警消息格式 |
|------|---------|------------|
| worker_task | task.done() | "⚠️ RefinementWorker 已停止" |
| sqlite_size | > 500MB | "⚠️ SQLite {size:.0f}MB > 500MB limit" |
| pg_connect | ping > 3s | "⚠️ PG 不可达 (timeout={t}s)" |

### 飞书集成
- 复用 config/monitoring.yaml 的 feishu webhook
- 消息格式: `[TM WARNING] {检查项}: {详情}`

## Edge Cases
- 飞书 webhook 不可达 → 记录 error 日志, 不阻塞主流程
- 多个异常同时发生 → 逐个告警
- worker task 恢复需要外部重启 → watchdog 只能检测, 不自动重启

## Out of Scope
- 自动重启 daemon
- 告警升级 (电话/短信)
- Web 健康检查 (已有 make health)
