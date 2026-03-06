# 9 个 commit 与 9 个任务的对应关系

## 计划中的 9 个任务（来自三份 plan）

| 计划文件 | 任务 | 说明 |
|----------|------|------|
| p4_3_p2_5_p3_7 | P4-3 | CI/CD 流水线 |
| p4_3_p2_5_p3_7 | P2-5 | 工作流模板 |
| p4_3_p2_5_p3_7 | P3-7 | 详细使用日志（本次未做） |
| p1_6_p1_5_p2_3 | P1-6 | FTS 存量迁移 |
| p1_6_p1_5_p2_3 | P1-5 | 检索与解析（query_expansion、质量门控） |
| p1_6_p1_5_p2_3 | P2-3 | 请求日志与审计 |
| p1_10_p1_9_p2_2 | P1-10 | Phase 4 成功标准（文档） |
| p1_10_p1_9_p2_2 | P1-9 | extraction 配置 quality_gate/max_retries |
| p1_10_p1_9_p2_2 | P2-2 | Prometheus /metrics 端点 |

## 当前 9 个 commit 与任务对应

| # | Commit | 当前 message | 对应任务 | 建议 message |
|---|--------|--------------|----------|--------------|
| 1 | 1dfa210 | P4-3: CI... | P4-3 | 保持 |
| 2 | 283570c | P2-5: 工作流模板... | P2-5 | 保持 |
| 3 | bc6b507 | MCP: 更新 instructions... | P1-9（MCP 侧体验与 feedback 闭环） | P1-9: MCP instructions 与 feedback_hint、工具描述 |
| 4 | 1348301 | save_group/save: embedding... | 体验改进（无单独任务 ID） | 保持或：体验: embedding 含 tags、submit_feedback **kwargs |
| 5 | 1b94c91 | P2-3: 请求 JSON 日志... | P2-3 | 保持 |
| 6 | 5e68d0b | P1-5: 检索与解析... | P1-5（含 P1-9 的 config/llm_parser） | 保持 |
| 7 | ed13f88 | P1-6: FTS 存量迁移... | P1-6 | 保持 |
| 8 | 6e23245 | docs: README 更新... | P1-10 及相关文档 | P1-10: README 文档（CI/CD、FTS、tm_learn、Make 表） |
| 9 | a64c7c9 | chore: 依赖与配置... | P2-2（/metrics）+ 其余 | P2-2: /metrics 端点与依赖、bootstrap、前端、测试 |

注：P3-7 本次未做；P1-9 的 extraction 配置在 P1-5 与 P2-5 的 commit 中已有体现，MCP 条可标为 P1-9 的「MCP 侧使用与体验」。
