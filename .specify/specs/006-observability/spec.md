# Feature Specification: Observability (Metrics + Watchdog + Logging)

**Created**: 2026-05-14
**Status**: Draft
**Input**: P1 可观测性三项: metrics(SQLite+CLI+飞书), watchdog(tick检查+backoff告警), 日志(JSON+session_id+source)

## Clarified Requirements

| # | 问题 | 决策 |
|---|------|------|
| 1 | Metrics 交付方式 | SQLite 存 + CLI 查看 + 飞书定时推送 |
| 2 | Watchdog 检查频率 | 每次 tick 附带检查 (30s 间隔) |
| 3 | 告警去重 | 首次立即→10min→20min→40min backoff, 恢复时通知 |
| 4 | 日志格式 | JSON 格式, 每条关键日志带 session_id + source |
| 5 | 日志位置 | 保持 /tmp/tm-daemon.log, web 不动 |
| 6 | 日志查看 | make logs SESSION=xxx SOURCE=cursor LEVEL=ERROR |

## User Scenarios

### P1: 查看系统核心指标
- As a 系统管理员
- I want to 查看 hook 调用量 / session 数 / 精炼延迟
- So that 我能评估系统健康状况
- Acceptance Criteria:
  1. tm-cli metrics show 显示当前指标
  2. 指标包含: hook_calls(by source) / sessions / messages / refinement_latency(p50/p95) / search_hit_rate / memory_success_rate
  3. 飞书每 6 小时推送一次统计摘要
  4. 数据持久化在 SQLite (drafts.db 的 metrics 表)

### P2: 异常自动发现并告警
- As a 系统管理员
- I want to 系统自动检测异常并通知我
- So that 我不需要手动盯着日志
- Acceptance Criteria:
  1. 每次 tick 检查: daemon 进程 / worker task / SQLite 文件大小 / PG 连接
  2. 异常 → 飞书告警: 首次立即 → 10min → 20min → 40min backoff
  3. 恢复 → 飞书通知 "已恢复"
  4. 告警消息包含 source + 具体数值 (如 "SQLite 128MB > 100MB limit")

### P3: 按 session_id 追踪完整链路
- As a 开发者排查问题时
- I want to 通过 session_id 追踪一次对话全流程
- So that 我能快速定位断在哪一步
- Acceptance Criteria:
  1. 关键节点日志 (before_prompt / after_response / convergence / refinement / publish) 都带 session_id + source
  2. make logs SESSION=xxx 查看该 session 所有相关日志
  3. make logs SOURCE=cursor LEVEL=ERROR 按 source 和级别过滤
  4. make logs --tail 50 查看最近 N 条

### P4: 用户画像提取可视化
- As a 用户
- I want to 看到每次提取了哪些用户画像
- So that 我知道系统是否在学习我的偏好
- Acceptance Criteria:
  1. PersonalMemoryExtractor 提取成功后记录 metric + 日志
  2. tm-cli metrics show 显示 memory_extraction_success_rate

## Technical Requirements

### 6. Metrics 系统

#### SQLite 表
```sql
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT DEFAULT '{}',    -- JSON: {"source":"cursor","project":"xxx"}
    recorded_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
```

#### 统计指标
| 指标名 | 计算方式 | 标签 |
|--------|---------|------|
| hook_calls_total | 每次 before_prompt / after_response +1 | source, project, event |
| sessions_active | 当前 active session 数 | project |
| messages_total | 当前 message 总数 | project, role |
| refinement_latency_seconds | _refine_one 耗时 | project |
| search_hit_rate | was_used=true / total search_log | project |
| memory_extraction_success | 提取成功数 / 总尝试数 | persona |

#### CLI
- tm-cli metrics show → 打印当前所有指标
- tm-cli metrics show --name hook_calls_total → 过滤

#### 飞书推送
- 每 6 小时推送统计摘要到飞书群
- 格式: "TM Metrics (14:00): 24h hooks=342(cursor:200, hermes:142) sessions=15 latency_p50=2.3s hit_rate=12% memory=0"

### 7. Watchdog 系统

#### 检查项 (每次 _tick 执行)
```python
async def _watchdog_check():
    checks = {
        "daemon_alive": True,  # 能执行到这里说明活着
        "worker_task": not refinement_task.done(),  # task 未死
        "sqlite_size_mb": os.path.getsize(db_path) / 1024 / 1024,
        "pg_ping": await pg_pool.ping(),  # PG 连通
    }
    for name, result in checks.items():
        if is_anomaly(name, result):
            alert_if_due(name, result)
        else:
            clear_alert(name)  # 恢复 → 通知
```

#### 异常阈值
| 检查项 | 阈值 | 说明 |
|--------|------|------|
| worker_task | task.done() | worker 意外退出 |
| sqlite_size_mb | > 500MB | SQLite 文件异常增长 |
| pg_ping | timeout > 3s | PG 不可达 |

#### 告警策略
- 首次发现异常 → 立即飞书告警
- 持续异常 → 10min → 20min → 40min backoff
- 异常恢复 → 飞书通知 "✅ 已恢复"

### 8. 日志系统

#### 格式改造
```
改造前: 20:03:16 INFO [READ]  before_prompt → project=xxx query="..." results=0

改造后: {"ts":"2026-05-14T20:03:16","level":"INFO","event":"before_prompt","session":"a1b2c3d4","source":"cursor","project":"MyGuguGaga","query":"...","results":0}
```

#### 改造范围
- 处理日志: before_prompt, after_response, session_start, session_end
- 精炼日志: converge, refine_start, refine_done, refine_fail, publish
- 系统日志: startup, shutdown, error
- 不改造: debug 级别日志 (保持原样)

#### make logs 命令
```bash
make logs                          # tail -50 daemon 日志
make logs SESSION=abc123           # 按 session_id 过滤 (jq select)
make logs SOURCE=cursor            # 按 source 过滤
make logs LEVEL=ERROR              # 按日志级别过滤
make logs --tail 100               # 最近 N 条
```
实现: Makefile + shell 脚本，用 python -c + json.loads 过滤 JSON 行日志

### Data Flow
```
hook 触发
  → endpoint 入口: 记录 event=before_prompt, session=..., source=...
  → pipeline 处理: 记录搜索结果
  → 收敛检测: 记录 converge 事件
  → RefinementWorker: 记录 refine_start/done/fail
  → publish: 记录 publish 结果

Watchdog:
  → _tick 末尾 → _watchdog_check()
  → 异常 → 飞书告警 (首次/backoff)
  → 恢复 → 飞书通知

Metrics:
  → 每个事件 → MetricsCollector.record(name, value, labels)
  → SQLite INSERT
  → CLI 查询: tm-cli metrics show
  → 飞书: 定时 cron 推送
```

### Interface Contracts
| 模块 | 职责 |
|------|------|
| MetricsCollector | record() / query() / summary() |
| Watchdog | check() / alert() / backoff() |
| StructuredLogger | 包装 logger, 自动注入 session_id + source |
| make logs | shell 脚本 + jq 过滤 |

## Edge Cases
1. 飞书 webhook 不可达 → 告警失败不阻塞主流程，记录 error 日志
2. metrics 表过大 → 保留最近 30 天，定时清理
3. session_id 缺失 (某些 hook 没传) → 用 "unknown" 占位
4. source 无法识别 → 用 "unknown"

## Out of Scope
- Grafana / Prometheus 集成
- Web 日志改造 (docker stdout 够用)
- 告警升级 (电话/短信)
- 日志压缩归档
