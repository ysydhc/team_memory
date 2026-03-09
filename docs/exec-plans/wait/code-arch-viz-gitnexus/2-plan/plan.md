# 架构可视化能力 — 实施计划

> 基于 [code-arch-viz-gitnexus-requirements-spec](../1-research/assessment.md) 与 [options](../1-research/options.md) 整理的可执行步骤。  
> 执行时配合：`code-arch-viz-operations.md`（操作步骤，见原主题目录）、Provider 接口契约（见 options）。

## 一、目标与范围

1. 在 TM Web 主导航增加「架构」入口，在 TM 面板内直接预览代码架构（概览、集群、图、影响面），无需跳转 Cursor 或外部工具。
2. 架构数据当前由 **GitNexus** 提供，通过 **Provider 抽象** 接入，配置 `architecture.provider: gitnexus`；预留 `builtin` 自研方案切换能力。
3. 经验可挂载到架构节点；支持按节点查经验、任务完成时写入架构节点绑定；tm_solve/tm_search 支持按位置检索。

## 二、任务列表（按依赖排序）

| 序号 | 任务 | 产出 | 依赖 |
|------|------|------|------|
| T1 | 配置与模型 | config.yaml 增加 architecture 段；Pydantic 模型（ArchitectureContext、ArchitectureGraph、ClusterSummary、ClusterMembers、ImpactResult、ExperienceRef） | — |
| T2 | Provider 协议与工厂 | ArchitectureProvider 协议/抽象类；get_provider(config) 工厂（仅 gitnexus 分支） | T1 |
| T3 | GitNexusProvider 实现 | GitNexusProvider：读 meta.json 或调 Bridge 的 context/clusters/cluster/impact；graph 可先返回简化数据或依赖 Bridge 实现 | T2；Bridge 契约 |
| T4 | 路由层 | routes/architecture.py：注册 context/graph/clusters/cluster/{name}/impact/experiences；鉴权；无 Provider 时 503 或 available:false | T2 |
| T5 | 经验绑定与 experiences API | experience_architecture_binding 表或 code_anchor 约定；GET /architecture/experiences 查 TM 库 | T1；可与 T4 合并 |
| T6 | 前端架构页 | 主导航「架构」；page-architecture；调 context，有数据展示概览+stale，无数据展示「未配置」；可选 Tab 集群/图/影响面 | T4 |
| T7 | 图与侧栏 | 图组件调 graph API 渲染；点节点侧栏调 impact + experiences | T6 |
| T8 | 经验挂载入口 | 经验创建/编辑「挂载点」选择；写入 binding 或 code_anchor | T5，T6 |
| T9 | 文档与验收 | README/.debug 更新；按验收清单执行并记录 | T1～T8 |
| B1 | Bridge 服务 | 连接 GitNexus MCP；暴露 GET /context、/clusters、/cluster/:name、/impact、/graph | 可与 T3 并行 |
| B2 | Bridge 与 TM 联调 | TM 配置 bridge_url 后 context/clusters/impact 正常；graph 可首版简化 | 依赖 B1，T3 |

## 三、验收标准

| 编号 | 验收项 | 通过标准 |
|------|--------|----------|
| V1 | 配置 | config.yaml 含 architecture.provider、architecture.gitnexus.bridge_url；应用启动无报错 |
| V2 | Context API | GET /api/v1/architecture/context 返回 200 或 503；有 bridge 时 available=true 且含 symbols/relationships/processes；无 bridge 时 available=false 或 503 |
| V3 | Clusters API | GET /api/v1/architecture/clusters 返回集群列表；GET /api/v1/architecture/cluster/{name} 返回成员 |
| V4 | Graph API | GET /api/v1/architecture/graph 返回 nodes、edges；支持 query file_path 且响应含 focus_node_id（可为 null） |
| V5 | Impact API | GET /api/v1/architecture/impact?path=...&depth=2 返回 upstream、downstream 数组 |
| V6 | Experiences API | GET /api/v1/architecture/experiences?node=... 返回 TM 侧经验列表（可空） |
| V7 | 前端架构页 | 主导航有「架构」；进入后可见概览或「未配置」提示；有数据时可选展示集群/图/影响面 |
| V8 | 经验挂载 | 经验创建/编辑可选择挂载点（node）；节点侧栏可查看关联经验 |
| V9 | 降级 | 关闭 bridge 或清空 bridge_url 后，架构页显示「未配置或不可用」，经验列表与任务照常可用 |

## 四、参考文档（原主题目录）

- `code-arch-viz-operations.md` — 逐步操作步骤（配置→Bridge→后端→路由→前端→验收）
- `code-arch-viz-provider-interface.md` — Provider 枚举、统一 API 契约、预留代码联动
- `code-arch-viz-gitnexus-data-mapping.md` — GitNexus 数据与契约对比、映射与实现要点
- `code-arch-viz-gitnexus-integration.md` — GitNexus 接入方案、Bridge/读索引/iframe 选项
- `code-arch-viz-balanced-solution.md` — 三视角综合均衡方案、与 GitNexus 对比
