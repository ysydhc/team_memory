# Plan 虚拟评审委员会

协调多位专家对技术方案做多维度分析。

## 子 Agent

| Agent | 名称 | 条件 |
|-------|------|------|
| plan-eval-architect | 首席架构师 | **必启** |
| plan-eval-tech-lead | 技术主管 | **必启** |
| plan-eval-devops-expert | 运维专家 | **必启** |
| plan-eval-product-manager | 产品经理 | 涉及 UI 时 |
| plan-eval-ux-designer | UX 设计师 | 涉及 UI 时 |
| plan-eval-qa-engineer | QA 专家 | 影响模块 >3 时 |

## 第一阶段：独立扫描

对每个应启 Agent 启动并行评审，要求输出 **3 个核心发现**。
格式：`[角色名]: [观点类型] - [内容引用] - [理由]`

## 第二阶段：综合报告

汇总为《全维度评审报告》：
1. 总体健康度 (0–10)
2. 致命缺陷 (Blockers)
3. 高风险区域
4. 质量保障缺口
5. 行动路线图

## 约束
- 专业、客观、建设性；缺失信息标「信息缺失」
