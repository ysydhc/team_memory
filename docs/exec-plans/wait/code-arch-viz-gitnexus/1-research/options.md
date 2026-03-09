# 架构可视化 — 方案对比与选型

> 综合三视角（技术最简、收益最大、使用最易）后的均衡方案，及「复用 GitNexus」vs「自建图谱」的选型结论。

## 一、均衡方案摘要（三视角综合）

| 维度 | 技术最简 | 收益最大 | 使用最易 | 均衡取舍 |
|------|----------|----------|----------|----------|
| **图谱粒度** | 仅文件级 + IMPORTS | 文件→函数/模块均可 | 先模块级看图 | **MVP：文件级 + IMPORTS；默认展示模块级视图** |
| **存储** | PostgreSQL 双表 | 关联表 + 可选图谱 | 不强调 | **双表 + 经验–节点关联表，不引入图库** |
| **经验挂载** | 约定 code_anchor/git_refs | 任务完成写节点 + 按位置检索 | 挂/查 ≤3 步 | **关联表 + 任务完成写节点（文件级）+ 按位置检索** |
| **入口** | API + 可选 MCP | tm_solve/file_path 增强 | Web 架构页 + MCP | **Web「架构」主导航 + 只读 API + MCP 查询/挂载** |

## 二、复用 GitNexus vs 自建图谱

| 项 | 自建（原均衡方案） | 复用 GitNexus（已采纳） |
|----|--------------------|---------------------------|
| 图谱数据 | TM 自己 ast 解析、PostgreSQL 双表 | 完全用 GitNexus（执行流、CALLS、聚类、Cypher 全有） |
| 实现量 | 建表、解析器、索引、BFS 影响面、前端图 | 后端代理/读索引 + 前端图 + 经验绑定 |
| 执行流 / 细粒度 | Phase 2+ 再扩展 | 直接用 GitNexus 已有能力 |
| 经验/任务 | TM 全管 | 不变，仍 TM 管；仅节点 key 与 GitNexus 对齐 |

**结论**：若团队已用或可接受 GitNexus，优先采用复用方案，在 TM 面板直接预览并保留「经验挂载 + 按位置检索 + 任务写节点」的 TM 闭环。

## 三、GitNexus 接入方式

| 方式 | 说明 | 本期 |
|------|------|------|
| **Bridge 服务** | 独立进程连接 GitNexus MCP，暴露 HTTP（/context、/graph、/clusters、/impact 等），TM 用 httpx 调用 | 推荐；若无 Bridge 则 Provider 返回 available: false |
| **直接读 .gitnexus** | 读 meta.json + 若格式开放则读图数据 | 可选补充（如 context 可直接读 meta.json 降级） |

## 四、Provider 接口与预留

- **配置键**：`architecture.provider`，取值 `gitnexus` | `builtin`。
- **统一 API**：`/api/v1/architecture/context`、`graph`、`clusters`、`cluster/{name}`、`impact`、`experiences`。
- **预留**：`focus_node_id`（代码联动）、`GET /architecture/code-context`（可选）。
- 完整契约见原 `code-arch-viz-provider-interface.md`（已合并至本主题）。
