# Failure Taxonomy — 失败分类与恢复策略

> 所有可配置阈值引用 `harness-config.yaml`，不硬编码。

---

## 1. 三级恢复策略

### Level 1: 自动恢复（Hook / 即时处理）

无需 Agent 判断，由 Hook 或简单规则自动处理。

| 触发条件 | 恢复动作 |
|----------|---------|
| lint 报错 | PostToolUse Hook 自动 format |
| import 方向违规 | Hook 检测并提示修正 |
| 测试编译失败 | 自动重试（最多 `failure.auto_retry_limit` 次） |
| 命令超时（< `timeouts.command`） | 自动 kill，记录警告，继续下一步 |

### Level 2: Agent 判断恢复

需要 Agent 分析上下文后决定修复方式。

| 触发条件 | 恢复动作 |
|----------|---------|
| 测试逻辑失败 | 分析失败原因，调整实现 |
| 类型错误 | 读取上下文，修复类型 |
| 依赖缺失 | 检查 requirements，安装 |
| 安装/构建超时（< `timeouts.install`） | 判断是否换源/换方案 |
| 偏差修复（bug/缺失校验/阻塞依赖） | 自动修正（最多 `failure.max_deviation_fixes` 次） |

### Level 3: 人工介入

Agent 无法自行解决，需要人工决策。

| 触发条件 | 恢复动作 |
|----------|---------|
| 架构决策冲突 | 停下询问 |
| 外部服务不可用 | 报告并等待 |
| scope 超出契约 | 提示需要新 Plan |
| 任务总超时（> `timeouts.task`） | 保存 progress，提示人工介入 |
| 连续 `failure.escalate_after_failures` 次 Level 2 失败 | 升级到人工 |

---

## 2. Agent 行为异常检测

除代码执行错误外，Agent 自身也可能出现行为异常：

| 异常类型 | 检测条件 | 恢复动作 |
|----------|---------|----------|
| **Observer Loop** | 连续 `context.max_consecutive_reads` 次 Read/Search 无 Write | 提示基于已有信息行动 |
| **重复操作** | 同一命令/编辑执行 3+ 次 | 停止并分析根因 |
| **Context 耗尽** | 使用率 > `context.critical_threshold` | 保存状态到 progress.md，拆分子任务 |
| **无限回退** | 同一文件 undo/redo 3+ 次 | 回退到最近的 git commit，重新规划 |
| **偏离契约** | 编辑了 Plan Contract scope 外的文件 | 警告并确认是否扩展 scope |

---

## 3. 错误分类维度

| 维度 | 分类 | 恢复策略映射 |
|------|------|-------------|
| **可逆性** | 可逆 / 不可逆 | 不可逆操作前必须人工确认 |
| **自动化程度** | 可自动 / 需判断 / 需人工 | 对应 Level 1 / 2 / 3 |
| **影响范围** | 局部 / 跨模块 / 外部 | 范围越大，越倾向高 Level |
| **来源** | 代码错误 / 环境问题 / Agent 行为异常 | 行为异常走 §2 专项处理 |

---

## 4. 超时策略

三级超时防止命令或任务无限挂起：

| 级别 | 超时值 | 适用场景 | 超时后恢复 |
|------|--------|---------|-----------|
| **command** | `timeouts.command` | 单条命令（ruff、grep、git） | Level 1：kill + 警告 + 继续 |
| **install** | `timeouts.install` | 安装/构建（pip install、make build） | Level 2：Agent 判断换源/换方案 |
| **task** | `timeouts.task` | 单个 Task 总时长 | Level 3：保存 progress + 人工介入 |

---

## 5. 安全分级门控

命令执行前按风险分级处理：

| 风险等级 | 示例 | 处理方式 |
|----------|------|---------|
| **Low** | `ruff check`, `pytest`, `git status`, `ls` | 直接执行 |
| **Medium** | `git commit`, `pip install`, `alembic upgrade` | 执行 + 记录日志 |
| **High** | `rm -rf`, `git push --force`, `DROP DATABASE`, `sudo` | Hook 阻止（exit 2） |
| **Interactive** | `vim`, `nano`, `python -i`, `psql`（裸调用） | Hook 阻止 + 提示替代方案 |

具体阻止列表在 `harness-config.yaml` 的 `security` 块中配置，由 `hooks/pre-bash-guard.sh` 执行。
