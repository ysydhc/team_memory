# Phase 1 收尾清单（Harness 文档迁移）

当 Plan 涉及 Harness Phase 1 类文档/路径迁移（如知识库重组、exec-plans 迁移）时，收尾须按本清单逐项勾选。

---

## 一、引用与校验

- [ ] `./scripts/harness_ref_scan.sh` 已执行，输出无异常
- [ ] `./scripts/harness_ref_verify.sh` 已执行，exit 0
- [ ] 无 404 或指向已移动路径的引用

## 二、Tool usage 基线（可选，仅 team_memory 项目）

- [ ] 若项目启用 team_memory 且服务可用：已执行 `scripts/harness_tool_usage_baseline.sh`，产出非 placeholder 的 JSON
- [ ] 若不可用：已标注「待服务就绪后补跑」或「非 tm 项目可跳过」

## 三、文档与规则同步

- [ ] 文档维护规则已同步 harness 文档结构（如 docs/design-docs、docs/exec-plans 路径约定）
- [ ] README、AGENTS.md 等主入口已更新（若有结构变更）
- [ ] harness-engineering.mdc 与 feedback-loop 等规则衔接正确

## 四、质量门禁

- [ ] `ruff check src/` 通过
- [ ] `pytest tests/ -q` 通过（或按 Plan 约定的测试范围）
- [ ] Success Metrics 达成（如 AGENTS.md ≤ 120 行、引用零断裂）

## 五、可复用

本清单可作为后续类似 Plan（文档迁移、知识库重组）的收尾模板，按需裁剪。
