# 文档整理任务

## 0. 前置：读取规范

**必须**先读取 `doc-maintenance-guide.md`、`plan-document-structure.md` 的完整内容。

## 1. 扫描

- 执行 `make harness-doc-check`、`make harness-plan-check`
- 按**主题**汇总问题

## 2. 输出问题列表并提问

| 主题 | 违规 | 请确认 |
|------|------|--------|
| (主题) | (规则 ID) | 白名单 / 整理？ |

若无问题，输出「无待确认项」并结束。

## 3. 按用户决策执行

- **白名单**：加入 `scripts/plan-structure-whitelist.txt` 或 `scripts/doc-gardening-whitelist.txt`
- **整理**：按 plan-document-structure 执行（1-research/2-plan/3-retro 重组），合并时按「复盘必需保留清单」逐项核对
- 结构变更后同步 `docs/design-docs/README.md`、`AGENTS.md`
- 输出整理前后对比
