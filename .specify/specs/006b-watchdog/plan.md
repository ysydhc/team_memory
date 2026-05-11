# Implementation Plan: In-Process HealthCheck (Watchdog 补充)

**Spec**: spec.md (006b)
**Status**: Ready for review
**Updated**: 2026-05-14 (基于现有架构重新设计)

## 现有架构

```
┌─────────────────────────────────────────────┐
│  进程 1: Web (docker, :9111)                │
│  进程 2: Daemon (python, :3901)             │
│  进程 3: ServiceWatchdog (launchd, PID29001)│
│         外部 HTTP probe: Web + Daemon        │
│         自动重启 + 飞书告警                  │
│         已运行 ✅                            │
└─────────────────────────────────────────────┘
```

**ServiceWatchdog**: 外部探活 — "端口在监听吗?"
**HealthCheck (新)**:  进程内检查 — "内部健康吗?"

两者互补，各司其职。ServiceWatchdog 不修改。

## 新模块: HealthCheck

### 职责
- 每次 daemon._tick 末尾运行 (30s 间隔)
- 检查: worker_task 状态 / SQLite 大小 / PG 连通
- backoff 飞书告警
- 异常恢复通知
- 复用已有 AlertManager (app.py 的 `send_daemon_alert`)

### 为什么不用现有 ServiceWatchdog
| | ServiceWatchdog | HealthCheck (新) |
|------|------|------|
| 运行位置 | 独立 launchd 进程 | daemon 进程内 |
| 检查方式 | HTTP probe | 直接读文件 + asyncio task + asyncpg ping |
| 能看 worker_task? | ❌ 进程外无法访问 | ✅ 直接检查 task.done() |
| 能看 SQLite 大小? | ❌ 不共享文件系统 | ✅ os.path.getsize(db_path) |
| 能看 PG (daemon视角)? | ❌ 只能 HTTP probe | ✅ pg ping |
| 挂了怎么办? | 自动重启进程 | 飞书告警 (daemon 挂了 check 也停了) |

## Implementation Steps

### Step 1: 创建 HealthCheck 类 (TDD)

### 现状
- 无进程内健康检查
- AlertManager + send_daemon_alert 已存在于 app.py
- 飞书 webhook 已配置在 monitoring.yaml

### 目标
- `HealthCheck` 类管理 3 项进程内检查 + backoff 告警
- 复用 `send_daemon_alert` (app.py 已实现)
- 4 个单元测试

### 具体改动

**文件**: src/team_memory/daemon/health_check.py (新建)

```python
class HealthCheck:
    """In-process health checks. Runs at end of each _tick."""

    BACKOFF_MINUTES = [10, 20, 40]  # 告警退避: 首次立即, 后续递增
    SQLITE_MAX_MB = 500

    def __init__(self, db_path: str, pg_url: str | None = None,
                 worker_task: asyncio.Task | None = None,
                 alert_callback: callable | None = None):
        # alert_callback: async def send(title, body, severity)
        # 通常传 send_daemon_alert
        self._alerts: dict[str, dict] = {}  # name -> {last_alert, level, started}

    async def check_all(self) -> list[str]:
        """Run all checks. Returns list of anomaly names.
        For each anomaly: call _handle_alert with backoff.
        For each recovery: call _handle_recovery."""

    async def _check_worker_task(self) -> str | None:
        """worker_task.done() -> anomaly"""

    async def _check_sqlite_size(self) -> str | None:
        """os.path.getsize > limit -> anomaly"""

    async def _check_pg_connect(self) -> str | None:
        """asyncpg.connect timeout 3s -> anomaly"""

    def _should_alert(self, name: str) -> bool:
        """Backoff: 首次 True, 之后 on cooldown"""

    async def _send_alert(self, name: str, detail: str):
        """调用 alert_callback(title, body, 'warning')"""

    async def _send_recovery(self, name: str):
        """调用 alert_callback(title, duration, 'info')"""
```

**文件**: tests/test_health_check.py (新建)
1. test_check_all_healthy → 返回空列表
2. test_check_sqlite_over_limit → 返回包含 "sqlite_size" 的异常
3. test_backoff_skips_alert → 首次告警后 10 分钟内不重复
4. test_recovery_sends_notification → 异常恢复 → alert_callback 被调用

### 依赖: 无
### 验收: 4 tests pass, ruff clean

---

### Step 2: 接入 HealthCheck 到 RefinementWorker._tick + app.py

### 现状
- _tick 末尾已调 cross_db_snapshot() (006a-metrics)
- app.py 已有 send_daemon_alert 函数
- RefinementWorker 已有 watchdog=None 参数 (预留但未使用)

### 目标
- _tick 末尾调 `await self._health_check.check_all()`
- 异常不抛给 caller
- app.py 创建 HealthCheck 并传给 worker

### 具体改动

**refinement_worker.py**:
```python
class RefinementWorker:
    def __init__(self, ..., health_check=None):
        self._health_check = health_check  # 改名: watchdog -> health_check

    async def _tick(self):
        # ... 现有精炼 + cross_db_snapshot ...
        if self._health_check:
            try:
                await self._health_check.check_all()
            except Exception:
                logger.exception("HealthCheck failed (non-critical)")
```

**app.py**:
```python
from team_memory.daemon.health_check import HealthCheck

# 在 startup 中 (MetricsCollector 之后):
health_check = HealthCheck(
    db_path=db_path,
    pg_url=pg_url,
    worker_task=refinement_task,
    alert_callback=send_daemon_alert,
)

refinement_worker = RefinementWorker(..., health_check=health_check)
```

### 依赖: Step 1
### 验收: 1862 tests 不破坏

---

### Step 3: 全量验证 + commit
- ruff check
- pytest -q -m "not integration"
- make e2e-verify
- 手动: kill worker → 飞书告警 (如果有飞书 webhook 可用)
- git commit