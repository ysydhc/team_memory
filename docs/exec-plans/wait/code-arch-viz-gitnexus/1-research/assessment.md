# 架构可视化能力 — 评估

> 基于现有全部资料整理的可行性、风险与约束评估。

## 一、范围与边界

| 在范围内 | 在范围外 |
|----------|----------|
| config 中 architecture 配置段 | GitNexus 安装与索引命令（用户侧） |
| TM 后端：Provider 接口 + GitNexusProvider + 路由 | 多仓库、多 repo 切换 UI |
| TM 前端：架构页（概览/集群/图/影响面）+ 节点侧栏（关联经验） | 执行流步骤级交互、Cypher 控制台 |
| 经验挂载与按节点查经验（TM 库） | 自研 BuiltinProvider 实现（预留接口，本期可不实现） |
| Bridge 方案设计及与 Provider 的契约 | Bridge 具体实现语言与仓库（可独立小项目） |
| 任务完成写节点（文件级，与现有 task 流程衔接） | 任务详情页「关联架构节点」展示（可 Phase 2） |

## 二、风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| Bridge 未实现或不可用 | 架构页仅能展示「未配置」 | 明确支持 bridge_url 为空；Provider 返回 available: false；文档说明如何搭 Bridge |
| GitNexus 索引 stale | 数据与当前代码不一致 | context 返回 stale: true；前端提示「请运行 npx gitnexus analyze」 |
| Impact 目标为符号、API 传文件路径 | 语义不一致 | Bridge/Provider 内将文件路径解析为该文件下符号列表并聚合 impact；文档约定 path 语义 |
| 图数据量大导致慢 | 首屏或图加载慢 | graph 支持 cluster 过滤、限制节点/边数量；前端可先展示概览与集群，图按需加载 |

## 三、技术选型结论

| 选项 | 选用 | 说明 |
|------|------|------|
| **GitNexus** | ✅ 本期 | 已有索引（.gitnexus）、MCP/CLI、执行流与聚类；通过 Bridge 或读 meta.json + MCP 接入 |
| **自研（builtin）** | 预留 | Provider 接口已定义，后续实现 BuiltinProvider（PostgreSQL 双表 + ast 索引）即可切换 |

## 四、数据映射要点（GitNexus → Provider）

- **Context**：meta.json 的 stats（nodes→symbols, edges→relationships, processes） + git HEAD 与 lastCommit 比较得 stale。
- **Clusters**：MCP clusters / cluster/{name} → ClusterSummary[]、ClusterMembers。
- **Graph**：由 Cypher 或 query+context 聚合成 nodes+edges；file_path 请求时填 focus_node_id。
- **Impact**：GitNexus impact 为符号级；upstream/downstream 各调一次，byDepth 合并为数组；path 为文件时需解析该文件下符号再聚合。
- 完整映射见原 `code-arch-viz-gitnexus-data-mapping.md`（已合并至本主题）。
