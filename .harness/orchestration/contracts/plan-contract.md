# Plan Contract 模板

> 在 Phase 1（Planning）中签署。确保 Plan 有明确的 scope、完成标准和人工决策点。

---

## Plan Contract: {plan_name}

### Scope

- [ ] 明确的交付物清单
- [ ] 影响范围（哪些文件/模块）
- [ ] 预估 context 消耗：☐ S（单 session）/ ☐ M（需拆分 wave）/ ☐ L（多 session）

### Verification

- [ ] 完成标准（可机器验证的命令）
- [ ] 质量门禁命令（来自 `harness-config.yaml` quality_gates）

### Constraints

- [ ] 不做什么（scope 边界）
- [ ] 依赖 / 前置条件
- [ ] 不可逆操作清单（需人工确认）

### Human Decision Points

- [ ] 需要人工确认的节点及触发条件

### Context Plan

- [ ] `read_first` 清单（子任务启动时必须读取的文件）
- [ ] Wave 拆分方案（如需要）：
  - Wave 1: ...
  - Wave 2: ...
