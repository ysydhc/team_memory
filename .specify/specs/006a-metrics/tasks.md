# Tasks: Metrics

**Plan**: plan.md
**Total tasks**: 7

## Task 1: TDD — 写 6 个失败测试
- File(s): tests/test_metrics.py (新建)
- Action:
  1. test_record_and_query: record -> query 能查到
  2. test_summary_aggregates: 多条 record -> summary 正确聚合
  3. test_latest_value: 同 name 多条 -> query 返回最新
  4. test_empty_summary: 无数据 -> summary 返回空 dict
  5. test_record_disk_full_not_raise: mock sqlite3.Error -> record 返回不抛异常
  6. test_cleanup_old_records: 插入 35 天前 -> cleanup -> 查询不到
- Test: 6 FAIL (类不存在)
- Depends on: none
- Status: [x] done

## Task 2: 实现 MetricsCollector
- File(s): src/team_memory/daemon/metrics_collector.py (新建)
- Action:
  1. __init__(db_path, pg_url=None): CREATE TABLE + INDEX
  2. record(name, value, labels): INSERT + try/except sqlite3.Error
  3. query(name=None, hours=24): SELECT WHERE recorded_at > cutoff
  4. summary(hours=24): SQL 聚合 -> dict
  5. cleanup(): DELETE WHERE recorded_at < 30 days ago
  6. cross_db_snapshot(): 从 PG 计算 search_hit_rate + memory_success_rate
     -> record 到 SQLite (需要 asyncpg 连接 pg_url)
- Test: 6 tests PASS
- Depends on: Task 1
- Status: [x] done

## Task 3: pipeline.py + app.py 接线
- File(s): src/team_memory/daemon/pipeline.py, src/team_memory/daemon/app.py
- Action:
  1. process_before_prompt 加 `metrics=None` 参数
     -> record hook_calls_total (event=before_prompt)
  2. process_after_response 加 `metrics=None` 参数
     -> record hook_calls_total (event=after_response)
  3. app.py startup: 创建 MetricsCollector(db_path, pg_url)
     -> metrics.cleanup() 清理旧数据
     -> 传给 process_before_prompt / process_after_response
- Test: 现有 1857 tests 不破坏
- Depends on: Task 2
- Status: [x] done

## Task 4: refinement_worker.py 接线
- File(s): src/team_memory/daemon/refinement_worker.py, src/team_memory/daemon/app.py
- Action:
  1. RefinementWorker.__init__ 加 `metrics=None` 参数
  2. _refine_one: 计时 -> finally record refinement_latency_seconds
  3. _tick 末尾: await self._metrics.cross_db_snapshot()
  4. app.py: 传 metrics 给 RefinementWorker 构造函数
- Test: 现有 1857 tests 不破坏
- Depends on: Task 2
- Status: [x] done

## Task 5: CLI — tm-cli metrics show + summary
- File(s): src/team_memory/cli.py
- Action:
  1. argparse: `metrics show --name`, `metrics summary --hours`
  2. cmd_metrics_show: 创建 MetricsCollector -> query -> 打印表格
  3. cmd_metrics_summary: 创建 MetricsCollector -> summary -> 打印
  4. db_path 复用 `_get_queue_db_path()`
- Test: 手动 — 发 hook -> tm-cli metrics show 有数据
- Depends on: Task 2
- Status: [x] done

## Task 6: 飞书推送脚本 + Makefile
- File(s): scripts/push-metrics.py (新建), Makefile
- Action:
  1. push-metrics.py: 读 FEISHU_WEBHOOK_URL -> MetricsCollector.summary(6)
     -> httpx.post -> try/except 飞书不可达不阻塞
  2. Makefile: 加 `metrics-push` target
- Test: make metrics-push -> 飞书收到消息
- Depends on: Task 2
- Status: [x] done

## Task 7: 全量验证 + commit
- Action:
  1. ruff check 全部改动文件
  2. pytest tests/ -q -m "not integration"
  3. make e2e-verify
  4. 手动: 发 hook -> tm-cli metrics show -> make metrics-push
  5. git commit
- Depends on: Task 3, Task 4, Task 5, Task 6
- Status: [x] done