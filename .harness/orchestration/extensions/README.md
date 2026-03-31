# Extensions — 编排扩展点

在 `task-flow.md` 的每个 Phase 结尾，会检查本目录下是否存在对应的扩展文件。

## 用法

放置以下文件名即可在对应 Phase 后执行额外步骤：

| 文件名 | 触发时机 |
|--------|---------|
| `after-phase-0.md` | Context Loading 后 |
| `after-phase-1.md` | Planning 后 |
| `after-phase-2.md` | Execution 后 |
| `after-phase-3.md` | Verification 后 |
| `after-phase-4.md` | Closure 后 |

## 可靠性

| 操作类型 | 可靠性 | 建议 |
|----------|--------|------|
| 新增步骤 | ✅ 高 | 直接添加文件 |
| 覆盖行为 | ⚠️ 中 | 可能与原始指令冲突，谨慎使用 |
| 条件分支 | ❌ 低 | 应放 `harness-config.yaml` 用配置控制 |

## 示例

```markdown
# after-phase-3.md — 验证后额外运行安全扫描

执行完 Phase 3 质量门禁后，额外运行：
1. `bandit -r src/` 安全扫描
2. 若发现 HIGH 级别问题，标记为 blocker
```
