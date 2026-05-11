# Feature Specification: Metrics

**Created**: 2026-05-14
**Status**: Draft
**Input**: tm-cli metrics show 查看系统指标 (SQLite存储 + CLI查询 + 飞书定时推送)

## Clarified
- SQLite 存储在 drafts.db 的 metrics 表
- tm-cli metrics show 命令行查看
- 飞书每 6 小时推送统计摘要
- 按 source (claude/hermes/cursor) 区分

## User Scenarios

### P1: 查看系统核心指标
- As 系统管理员
- I want to 运行 tm-cli metrics show
- So that 知道 hook 调用量 / session 数 / 精炼延迟
- Acceptance Criteria:
  1. tm-cli metrics show 显示当前所有指标
  2. tm-cli metrics show --name hook_calls_total 过滤
  3. 指标包含: hook_calls_total(by source), sessions_active, messages_total, refinement_latency, search_hit_rate, memory_success_rate
  4. 指标值准确 (发几条 hook → 数值正确变化)

### P2: 飞书定时推送统计
- As 系统管理员
- I want to 每天早上/下午各收到一次统计摘要
- So that 不需要手动跑 CLI
- Acceptance Criteria:
  1. 每 6 小时推送一次到飞书群
  2. 推送内容: 24h hook数量 / 活跃session / 精炼延迟 / 命中率 / 画像成功率

## Technical Requirements

### metrics 表
```sql
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT DEFAULT '{}',
    recorded_at TEXT DEFAULT (datetime('now'))
);
```

### 指标定义
| name | 计算 | labels |
|------|------|--------|
| hook_calls_total | before_prompt/after_response 计数 | source, project, event |
| sessions_active | 当前 active session 数 | project |
| messages_total | message 总数 | project, role |
| refinement_latency_seconds | _refine_one 耗时 | project |
| search_hit_rate | was_used=true / total | project |
| memory_success_rate | 提取成功/总尝试 | persona |

### MetricsCollector 类
- record(name, value, labels) → INSERT
- query(name=None) → SELECT with optional filter
- recent_summary(hours=24) → 过去 N 小时聚合

### CLI
- tm-cli metrics show → 打印表格
- tm-cli metrics show --name xxx → 过滤
- tm-cli metrics summary --hours 24 → 摘要

### 飞书推送
- 独立 cron job: 每 6h 调 tm-cli metrics summary → push to Feishu

## Edge Cases
- 表不存在 → auto-create
- 飞书不可达 → 不阻塞

## Out of Scope
- Prometheus / Grafana
- 长期趋势图表
