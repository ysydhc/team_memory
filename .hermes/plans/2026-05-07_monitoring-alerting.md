# Service Monitoring & Alerting Plan

> TM Daemon + Web 服务稳定性保障方案

## Goal

为 TM Daemon 和 Web 服务提供：
1. 心跳机制 — 进程级生命体征
2. 健康探针 — 外部可检测服务存活
3. Watchdog — 自动检测 + 重启 + 告警
4. 权限隔离 — 管理员专属监控配置

## Current Context

**已有基础：**
- Web: `/health` 端点（DB/Ollama/embedding/LLM/cache 五项检查）
- Web: `/ready` 端点（ readiness probe）
- Daemon: `/status` 端点（running 状态 + TM mode）
- Daemon: `/status/faithfulness` 端点
- Daemon: launchd 托管（KeepAlive=true，崩溃自动重启）
- Web: 无托管，手动启动，崩溃不重启

**缺失：**
- 无心跳机制（进程不主动报告存活）
- 无外部 watchdog（Web 崩溃无人处理）
- 无告警通知（服务异常管理员不知道）
- 无监控配置文件
- Daemon `/status` 不含详细健康信息

## Proposed Approach

### 1. 监控配置文件

新建 `config/monitoring.yaml`：

```yaml
admin:
  webhook: ""                    # 告警 webhook URL（Slack/飞书/企业微信）
  email: ""                      # 备用邮件告警
  restart_allowed: true          # 允许 watchdog 自动重启

health_check:
  web:
    url: "http://localhost:9111/health"
    timeout: 10                  # 超时秒数
    expected_status: 200
  daemon:
    url: "http://localhost:3901/status"
    timeout: 5
    expected_status: 200

watchdog:
  enabled: true
  interval: 60                   # 检查间隔（秒）
  max_failures: 3                # 连续失败 N 次后重启
  restart_command: "make web"    # Web 重启命令
  log_file: "/tmp/tm-watchdog.log"
  alert_cooldown: 300            # 告警冷却（秒），避免刷屏
```

### 2. 心跳机制

在 Daemon 和 Web 进程内部各加一个后台心跳任务：

**Daemon (`scripts/daemon/app.py`)：**
- 后台 asyncio Task，每 30 秒更新 `app.state.last_heartbeat`
- `/status` 端点返回 `heartbeat` 字段（最后心跳时间）
- 用于检测进程假死（alive 但不响应）

**Web (`src/team_memory/web/app.py`)：**
- 后台 asyncio Task，每 30 秒更新 `ctx.last_heartbeat`
- `/health` 端点返回 `heartbeat` 字段
- 用于检测 Web 进程假死

**心跳数据结构：**
```json
{
  "heartbeat": {
    "last_beat": "2026-05-07T15:30:00Z",
    "uptime_seconds": 3600,
    "pid": 26813
  }
}
```

### 3. Watchdog 脚本

新建 `scripts/monitoring/watchdog.py`：

```
功能:
  1. 读取 config/monitoring.yaml
  2. 定期探活（HTTP GET /health 和 /status）
  3. 连续失败 → 自动重启
  4. 重启失败 → 发送告警
  5. 写入监控日志

运行方式:
  - macOS: launchd 托管（独立 plist）
  - Linux: systemd timer 或 cron
  - 手动: python scripts/monitoring/watchdog.py

告警逻辑:
  - 首次失败: 记录日志
  - 连续 3 次失败: 自动重启 + 告警
  - 重启后恢复: 记录恢复日志
  - 告警冷却: 300 秒内不重复告警
```

### 4. 告警发送

新建 `scripts/monitoring/alerting.py`：

```
支持的告警渠道:
  1. Webhook（Slack/飞书/企业微信/自定义）
  2. macOS 通知（本地开发用）
  3. 日志文件（所有环境）

告警消息格式:
  [TM Alert] Web 服务异常
  - 服务: team-memory-web
  - 状态: down (连续 3 次)
  - 时间: 2026-05-07 15:30:00
  - 操作: 已自动重启
```

### 5. 自动重启

**Web 进程重启：**
- Watchdog 检测到 Web 不健康 → 执行重启命令
- macOS: `kill <pid> && python -m team_memory.web.app`
- Linux: `systemctl restart team-memory-web`
- 重启命令可配置（`restart_command` 字段）

**Daemon 进程重启：**
- 已有 launchd KeepAlive，崩溃自动重启
- Watchdog 补充：检测假死（alive 但 heartbeat 超时）→ kill + 等待 launchd 重启

### 6. Makefile 集成

```makefile
make watchdog-start    # 启动 watchdog
make watchdog-stop     # 停止 watchdog
make watchdog-status   # 查看 watchdog 状态
```

## Step-by-Step Plan

### Phase 1: 基础设施
1. 创建 `config/monitoring.yaml`（默认配置）
2. 创建 `scripts/monitoring/__init__.py`
3. 创建 `scripts/monitoring/config.py`（加载监控配置）

### Phase 2: 心跳
4. Daemon 加心跳后台任务 + `/status` 返回 heartbeat
5. Web 加心跳后台任务 + `/health` 返回 heartbeat
6. 更新测试

### Phase 3: Watchdog
7. 创建 `scripts/monitoring/watchdog.py`（探活 + 重启逻辑）
8. 创建 `scripts/monitoring/alerting.py`（告警发送）
9. Watchdog 读取 monitoring.yaml 配置

### Phase 4: 部署
10. macOS: 创建 launchd plist for watchdog
11. Makefile: 加 watchdog-start/stop/status
12. 文档更新

### Phase 5: 测试
13. 心跳单元测试
14. Watchdog 单元测试（mock HTTP）
15. 告警单元测试（mock webhook）

## Files to Change

**新建：**
- `config/monitoring.yaml`
- `scripts/monitoring/__init__.py`
- `scripts/monitoring/config.py`
- `scripts/monitoring/watchdog.py`
- `scripts/monitoring/alerting.py`
- `scripts/monitoring/heartbeat.py`
- `tests/test_watchdog.py`
- `tests/test_heartbeat.py`
- `tests/test_alerting.py`
- `~/Library/LaunchAgents/com.teammemory.watchdog.plist` (macOS)

**修改：**
- `scripts/daemon/app.py`（加心跳后台任务）
- `src/team_memory/web/app.py`（加心跳后台任务）
- `Makefile`（加 watchdog 命令）
- `AGENTS.md`（加监控说明）

## Tests / Validation

- 心跳: 验证后台任务更新 last_heartbeat
- Watchdog: mock HTTP 200/500/timeout，验证探活逻辑
- 告警: mock webhook，验证消息格式
- 集成: 启动 watchdog → 停止 Web → 验证自动重启 + 告警

## Risks / Tradeoffs

| Risk | Mitigation |
|------|------------|
| Watchdog 自身崩溃 | 用 launchd/systemd 托管 watchdog |
| 告警风暴 | alert_cooldown 冷却机制 |
| 远程服务器无 launchd | 支持 systemd + cron 降级 |
| 心跳增加资源消耗 | 30 秒间隔，开销极低 |

## Open Questions

1. 告警渠道优先级？（webhook > 邮件 > 日志）
2. 是否需要 Dashboard 展示服务状态？（后续迭代）
3. 多实例部署时 watchdog 如何处理？（每个实例一个 watchdog）
