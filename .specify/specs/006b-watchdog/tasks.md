# Tasks: In-Process HealthCheck

**Plan**: plan.md
**Total tasks**: 5

## Task 1: TDD — 写 4 个失败测试
- File(s): tests/test_health_check.py (新建)
- Action: 写 4 个测试 (全部应先失败):
  1. test_check_all_healthy — 无异常 → check_all 返回空列表
  2. test_check_sqlite_over_limit — mock os.path.getsize 返回 600MB → 返回 ["sqlite_size"]
  3. test_backoff_skips_alert — 首次异常触发告警 → 5 分钟后再次 check_all → 不触发 (还在 backoff 窗口内)
  4. test_recovery_sends_notification — 曾有异常 → 下次 check_all 恢复 → alert_callback 被调用 (recovery)
- Test: 4 ERROR (类不存在)
- Depends on: none
- Status: [x] done

## Task 2: 实现 HealthCheck 类
- File(s): src/team_memory/daemon/health_check.py (新建)
- Action:
  1. __init__(db_path, pg_url, worker_task, alert_callback)
  2. check_all() → 调 3 个 _check_* 方法 → 对异常: _handle_alert (backoff) / 对恢复: _handle_recovery → 返回异常列表
  3. _check_sqlite_size() → os.path.getsize + 500MB 阈值
  4. _check_worker_task() → task.done() 检查
  5. _check_pg_connect() → asyncpg.connect timeout 3s
  6. _should_alert(name) → backoff 计算 (首次→10min→20min→40min)
  7. _handle_alert(name, detail) → alert_callback(title, body, warning) + 记录 last_alert
  8. _handle_recovery(name) → alert_callback(title, duration, info) + 清除 alert_state
- Test: 4 tests PASS
- Depends on: Task 1
- Status: [x] done

## Task 3: 接入 RefinementWorker._tick + app.py
- File(s): src/team_memory/daemon/refinement_worker.py, src/team_memory/daemon/app.py
- Action:
  1. RefinementWorker.__init__: 改名参数 watchdog → health_check (预留的)
  2. _tick 末尾: try/except 调 await self._health_check.check_all()
  3. app.py startup: 创建 HealthCheck (db_path, pg_url, worker_task, send_daemon_alert)
  4. 传给 RefinementWorker 构造函数
- Test: 1862 tests 不破坏
- Depends on: Task 2
- Status: [x] done

## Task 4: 手动验证告警通路
- Action:
  1. 手工制造异常 (如改 SQLITE_MAX_MB=1 → 触发告警)
  2. 观察 daemon 日志 → "HealthCheck: sqlite_size 630MB > 1MB"
  3. 观察飞书 → 收到 [TM WARNING]
  4. 恢复阈值 → 飞书收到 [TM INFO] 已恢复
- Test: 手动
- Depends on: Task 3
- Status: [x] done

## Task 5: 全量验证 + commit
- Action:
  1. ruff check
  2. pytest -q -m "not integration"
  3. make e2e-verify
  4. git commit
- Depends on: Task 3
- Status: [x] done