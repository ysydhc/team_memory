# Implementation Plan: Metrics

**Spec**: spec.md (006a)
**Status**: Ready for review - 已纳入 checklist 4 项发现
**Updated**: 2026-05-14

## Checklist 处理

| # | 问题 | 方案 |
|---|------|------|
| 1 | 磁盘满 | record() 捕获 sqlite3.Error，log error，不抛异常 |
| 2 | 跨DB指标 | 每次 tick 从 PG 快照到 SQLite (search_hit_rate, memory_success_rate) |
| 3 | 表清理 | 启动时 DELETE WHERE recorded_at < 30天；summary 只查 30 天内 |
| 4 | 飞书4xx | push 脚本 try/except，失败记 error 日志，不阻塞 |

---

## Step 1: 创建 metrics 表 + MetricsCollector 类

### 现状
- 无 metrics 表，无指标收集机制
- pipeline.py / refinement_worker.py 只用 `logger.info/warning` 打印
- 无持久化指标数据

### 目标
- drafts.db 新增 `metrics` 表
- MetricsCollector 类提供 record / query / summary / cleanup / cross_db_snapshot
- 6 个单元测试

### 具体改动

**文件**: src/team_memory/daemon/metrics_collector.py (新建)

```python
class MetricsCollector:
    MAX_DAYS = 30  # 保留 30 天

    def __init__(self, db_path: str, pg_url: str | None = None):
        # CREATE TABLE IF NOT EXISTS metrics (
        #   id INTEGER PRIMARY KEY AUTOINCREMENT,
        #   name TEXT NOT NULL,
        #   value REAL NOT NULL,
        #   labels TEXT DEFAULT '{}',
        #   recorded_at TEXT DEFAULT (datetime('now'))
        # )
        # CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)
        # pg_url: 可选，用于跨 DB 指标快照

    def record(name, value, labels=None):
        """记录指标。磁盘满 -> log error, 不抛异常"""
        try: INSERT INTO metrics
        except sqlite3.Error: logger.error(...)

    def query(name=None, hours=24):
        """查询最近 N 小时记录，自动截断不超过 MAX_DAYS"""

    def summary(hours=24):
        """聚合摘要:
        返回 {
          "hook_calls_total": {"cursor_before": 100, "hermes_after": 50},
          "sessions_active": {"team_doc": 5},
          "refinement_latency_p50": 2.3,
          "search_hit_rate": 0.12,
          "memory_success_rate": 0.8
        }"""

    def cleanup():
        """DELETE FROM metrics WHERE recorded_at < (now - 30 days)"""

    async def cross_db_snapshot():
        """从 PG 快照跨 DB 指标到 SQLite (需 pg_url)。
        - search_hit_rate: SELECT COUNT(*)... search_logs WHERE was_used=true
        - memory_success_rate: 从 personal_memories 计算提取成功率"""
```

**文件**: tests/test_metrics.py (新建) — 6 tests:
1. test_record_and_query
2. test_summary_aggregates
3. test_latest_value
4. test_empty_summary
5. test_record_disk_full_not_raise: mock sqlite3.Error -> record 不崩溃
6. test_cleanup_old_records: 插入 35 天前的数据 -> cleanup -> 查询不到

### 依赖: 无
### 验收: 6 tests pass, ruff clean

---

## Step 2: 接入 MetricsCollector 到 pipeline + worker

### 现状
- process_before_prompt: (input_data, config, sink, buf, wiki) — 无 metrics 参数
- process_after_response: (input_data, config, sink, buf, detector, refiner) — 无 metrics 参数
- RefinementWorker.__init__: (config, buf, sink, queue) — 无 metrics 参数
- app.py 在 startup 中创建 buf/sink/wiki/queue，传给上述函数，没传 metrics

### 目标
- 给 3 个函数/类加 `metrics=None` 参数，向后兼容
- 关键事件点调用 `metrics.record()`
- 每次 _tick 调 `cross_db_snapshot()` 快照跨 DB 指标

### 具体改动

**pipeline.py** — 加 metrics 参数:
- `process_before_prompt` 签名加 `metrics=None`
- 在 return 之前: `if metrics: metrics.record("hook_calls_total", 1, {"source": source, "event": "before_prompt", "project": project})`
- `process_after_response` 签名加 `metrics=None`
- 同上 record after_response

**refinement_worker.py** — __init__ + _refine_one + _tick:
- `__init__` 加 `metrics=None` 参数
- `_refine_one`: 开始记 `time.time()`，finally 里 record latency
- `_tick` 末尾: `if self._metrics: await self._metrics.cross_db_snapshot()`

**app.py** — 创建并传递:
- 在 startup 里创建 `MetricsCollector(db_path, pg_url)`
- 调 `metrics.cleanup()` 清理旧数据
- 传给 process_before_prompt / process_after_response / RefinementWorker

### 依赖: Step 1
### 验收: 现有 1857 tests 不破坏，发 hook -> SQLite metrics 表有数据

---

## Step 3: tm-cli metrics show + summary

### 现状
- cli.py 有 setup/project/queue/archive 等子命令，无 metrics
- argparse 注册模式: `sub.add_parser("name")` -> `set_defaults(func=cmd_xxx)`
- db_path 解析复用 `_get_queue_db_path()`

### 目标
- `tm-cli metrics show` 打印所有当前指标 (表格)
- `tm-cli metrics show --name hook_calls_total` 过滤
- `tm-cli metrics summary --hours 24` 打印聚合摘要

### 具体改动

**cli.py** — argparse 注册 (跟在 setup 段后面):
```python
p_met = sub.add_parser("metrics", help="View system metrics")
m_sub = p_met.add_subparsers(dest="metrics_command")
p_show = m_sub.add_parser("show", help="Show current metrics")
p_show.add_argument("--name", help="Filter by name")
p_show.set_defaults(func=cmd_metrics_show)
p_sum = m_sub.add_parser("summary", help="Aggregated summary")
p_sum.add_argument("--hours", type=int, default=24)
p_sum.set_defaults(func=cmd_metrics_summary)
```

**cli.py** — 实现函数:
- `cmd_metrics_show`: 创建 MetricsCollector -> query(name=args.name) -> 按 name/labels 分组打印
- `cmd_metrics_summary`: 创建 MetricsCollector -> summary(hours=args.hours) -> 格式化输出

### 依赖: Step 1
### 验收: 发 hook -> `tm-cli metrics show` 有数据

---

## Step 4: 飞书定时推送

### 现状
- 无定时推送
- 飞书 webhook 在 config/monitoring.yaml 的 FEISHU_WEBHOOK_URL

### 目标
- 每 6h 推送统计摘要到飞书
- 飞书不可达 -> 不阻塞，记 error 日志

### 具体改动

**scripts/push-metrics.py** (新):
```python
# 1. 读 FEISHU_WEBHOOK_URL 环境变量
# 2. MetricsCollector.summary(hours=6)
# 3. 格式化为 Markdown 文本
# 4. httpx.post(webhook, json={"msg_type":"text","content":{"text":...}})
# 5. try/except: 失败 -> logger.error -> exit 0
```

**Makefile**: 加 `metrics-push` target

**cron**: 后续用 `cronjob` tool 创建每 6h 定时任务 (当前先手动验证飞书可通)

### 依赖: Step 1 + FEISHU_WEBHOOK_URL 可用
### 验收: make metrics-push -> 飞书收到消息

---

## Step 5: 全量验证 + commit

### 验收清单
- ruff check 全部改动文件
- pytest tests/ -q -m "not integration" -> 全部 pass
- make e2e-verify -> pass
- 手动发 hook -> tm-cli metrics show -> make metrics-push
- git commit