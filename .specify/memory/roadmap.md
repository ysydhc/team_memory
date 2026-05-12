# TeamMemory 路线图

基准: 2026-05-11, 从 8 个维度评估 30+ 差距项

---

## P0: 基础可靠性 - 数据不能丢 (2-3 weeks)

> 核心原则: 任何情况下用户的数据都不能丢失

### 1. Hook 一键安装 (3 days)
- 问题: Claude/Cursor/Hermes 各需手动配置 hook，步骤不同
- 方案: `tm-cli setup --platform all` 一键配置所有平台
- 验证: 安装后 make e2e-verify 通过

### 2. 优雅降级 (5 days)
- 问题: PG/LiteLLM 不可用时数据直接丢弃
- 方案:
  - Daemon 加 pending queue (SQLite)，后台重试
  - PG 恢复后自动 flush
  - LiteLLM 429 时 backoff 重试 (max 3次)
- 验证: 停 PG → 发 hook → 重启 PG → 数据补入

### 3. 数据一致性 (3 days)
- 问题: multi-step writes 无事务保护
- 方案:
  - draft_save + mark_published 用 status 状态机保证
  - 加 "refining"/"publishing" 中间状态防重复
- 验证: kill daemon mid-refinement → 重启后不重复处理

### 4. 静默失败 → 可见 (2 days)
- 问题: hook 出错返回 {}，用户不知道
- 方案:
  - Hook 脚本加 return error_code
  - Daemon 端点记录所有 4xx/5xx
  - 飞书告警: hook 连续失败 N 次
- 验证: 停 daemon → 发 hook → 飞书收到告警

### 5. 即时反馈 (2 days)
- 问题: 新用户 5 分钟内看不到结果 → 以为坏了
- 方案: before_prompt 返回时带上 X 条相关经验
- 验证: 发第一条 hook → 立刻看到 recall 结果


## P1: 可观测性 - 知道系统在做什么 (2 weeks)

### 6. 核心 Metrics (5 days)
- 统计:
  - hook 调用次数 (by platform)
  - session 数 / message 数
  - RefinementWorker 处理延迟 (p50/p95)
  - search_log was_used 命中率
  - personal_memories 提取成功率
- 暴露方式: /metrics 端点 (Prometheus 格式)
- 验证: 发几条 hook → /metrics 能看到变化

### 7. Watchdog + 告警 (3 days)
- Daemon 进程活着吗?
- RefinementWorker task 还在跑吗?
- SQLite 文件大小是否异常增长?
- PG 连接池是否耗尽?
- → 任一项异常 → 飞书告警

### 8. 日志聚合 (4 days)
- daemon + web + worker 日志统一输出到文件
- 结构化 JSON 格式
- 按 session_id 可追踪整条链路


## P2: 健壮性 - 不会莫名其妙坏 (2-3 weeks)

### 9. 失败注入测试 (5 days)
- 模拟: 停 Ollama, 停 PG, 停 LiteLLM, kill daemon, 填满磁盘
- 验证: 每种情况下数据是否丢失? 恢复后是否补回?

### 10. 回归测试套件 (4 days)
- make test-regression: 用真实 daemon + PG 跑关键链路
- CI 加入回归测试
- 任何 PR 必须通过

### 11. 精炼质量评估 (4 days)
- 自动评估每次精炼的质量
- 标题提取准确率
- 内容 faithfulness
- 低于阈值 → 不发布


## P3: 安全 + 部署 - 能上线 (2 weeks)

### 12. 访问控制
- Daemon API 加 Bearer token
- Token 从 .env 读取
- 无 token → 403

### 13. 后台服务化
- macOS: launchd plist
- Linux: systemd unit
- 开机自启, crash 自动重启

### 14. 备份恢复
- PG: pg_dump cron job
- SQLite: .backup 命令
- 保留最近 7 天


## P4: 体验 + 性能 (ongoing)

### 15. CLI 经验管理
- tm-cli create / edit / delete / list
- 不用 sql 就能管理经验库

### 16. Web Dashboard
- 简单的 HTML 页面
- 看 session / message / experience 统计
- 看最近的 search_log

### 17. 性能优化
- Embedding batching
- 消息分页加载
- SQLite WAL mode

---

## 当前进度

- [x] P0-4: 静默失败 → 可见 (PayloadCapture 已捕获)
- [x] Acceptance tests (21 tests)
- [x] made e2e-verify (完整链路)
- [x] 5-role review
- [x] P0-1: Hook 一键安装
- [x] P0-2: 优雅降级
- [x] P0-3: 数据一致性
- [x] P0-5: 即时反馈
- [x] P1-6: Metrics
- [x] P1-7: HealthCheck
- [x] P1-8: 日志聚合
- [x] P2-9: 失败注入
- [x] P2-10: 回归测试
- [x] P2-11: 精炼质量评估
- [x] P3-12: 跳过 (本地 daemon 无意义)
- [x] P3-13: launchd (已有 RunAtLoad+KeepAlive)
- [x] P3-14: 备份恢复 (PG+SQLite)
- [ ] P4: 待排期

---

## 成熟度指标

| 维度 | 当前 | P0后 | P1后 | P2后 | P4后 |
|------|------|------|------|------|------|
| 可靠性 | 60% | 85% | 90% | 95% | 95% |
| 可观测性 | 30% | 40% | 80% | 85% | 90% |
| 安全性 | 40% | 50% | 50% | 70% | 80% |
| DX | 40% | 55% | 65% | 75% | 90% |
| 部署 | 35% | 45% | 45% | 60% | 80% |
