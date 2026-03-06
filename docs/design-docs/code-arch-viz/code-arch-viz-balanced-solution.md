# 代码架构可视化 — 三视角综合均衡方案

> 综合三个 Agent 视角（**技术实现最简化**、**项目收益最大化**、**使用最易**）后的均衡方案，用于类似 GitNexus 的代码架构可视化能力在 team_memory 中的最简落地。

---

## 一、综合结论摘要

| 维度 | 技术最简 | 收益最大 | 使用最易 | 均衡取舍 |
|------|----------|----------|----------|----------|
| **图谱粒度** | 仅文件级 + IMPORTS | 文件→函数/模块均可 | 先模块级看图 | **MVP：文件级 + IMPORTS；默认展示模块级视图** |
| **存储** | PostgreSQL 双表 | 关联表 + 可选图谱 | 不强调 | **双表 + 经验–节点关联表，不引入图库** |
| **分析方式** | ast 仅 import | 可接受轻量/外部图谱 | 零配置出图 | **Python ast 解析 import；索引进 Makefile** |
| **经验挂载** | 约定 code_anchor/git_refs | 任务完成写节点 + 按位置检索 | 挂/查 ≤3 步 | **关联表 + 任务完成写节点（文件级）+ 按位置检索** |
| **入口** | API + 可选 MCP | tm_solve/file_path 增强 | Web 架构页 + MCP | **Web「架构」主导航 + 只读 API + MCP 查询/挂载** |
| **易用底线** | — | — | 零配置、三句话术、降级 | **MVP 必须：零配置出图、三句话术、失败不阻塞保存/检索** |

---

## 二、目标用户与核心功能（对齐三视角）

### 2.1 目标用户

- **日常开发 / Agent**：改代码前查「这里/相关模块」的经验；任务完成后经验自动挂到改动位置。
- **新人/接手者**：按模块/文件浏览经验，快速建立上下文。
- **任务负责人/维护**：看任务关联的文件/模块、影响面，评审与排期有据。

### 2.2 核心功能（MVP 与分期）

| 功能 | MVP | Phase 2 | Phase 3+ |
|------|-----|---------|----------|
| 文件/模块级图谱（IMPORTS） | ✅ | — | 可选 CALLS |
| 影响面查询（1～2 层） | ✅ | — | 深度/执行流 |
| 经验–架构节点关联与存储 | ✅ | — | 双向 API |
| 按位置检索（tm_search/tm_solve） | ✅ | 推荐挂载点 | 按节点筛选 |
| 任务完成写入架构节点 | ✅ 文件级 | 解析 symbol | 自动 diff |
| Web 架构图页（零配置、模块级默认） | ✅ | 节点侧栏完善 | 执行流/影响面一键 |
| MCP 查询与挂载 | ✅ 只读 + 挂载 | 深链、推荐 | — |
| 模块聚类/热点 | — | ✅ | 热点分析 |

---

## 三、技术选型（均衡）

### 3.1 语言与存储

- **语言**：Python，与 team_memory 一致。
- **存储**：复用现有 **PostgreSQL**。
  - **architecture_nodes**：id, project, kind(file|module), path, name, meta (JSONB)，可选 checksum。
  - **architecture_edges**：source_id, target_id, relation_type(IMPORTS), meta。
  - **experience_architecture_binding**（或 Experience.structured_data 约定）：experience_id, node_type, node_id（或 project+path），created_at；便于「按节点查经验」与「任务完成写节点」。

### 3.2 分析方式

- **MVP**：仅 **ast** 解析 `import` / `from ... import`，输出 (from_path, to_path)；相对 import 解析为项目内 path。
- **索引**：CLI 脚本 + **Makefile** 的 `make index-architecture`（与固化结论一致）；按需或定时，不常驻。
- **不引入**：tree-sitter、图数据库、执行流追踪（留 Phase 3+）。

### 3.3 与 TM 的集成点

- **数据**：经验通过 **experience_architecture_binding** 或 **structured_data.code_anchor** / **git_refs** 与 (project, path) 关联；不强制改 Experience 表结构时可先用扩展字段。
- **任务完成**：在现有 task_sediment 后增加「把 sediment 经验绑定到架构节点」；节点来源：调用方传入 changed_files 或后端从 title/description 解析，MVP 仅文件级。
- **检索**：tm_solve / tm_search 支持 **file_path** 与可选 **architecture_context**；有上下文时优先返回绑定在该节点/同文件/同模块的经验，并对绑定到当前节点的结果做 score boost。
- **API**：只读 `GET /api/architecture/graph`、`GET /api/architecture/impact?path=...&depth=2`；经验挂载用现有经验 API + 绑定表写入。
- **MCP**：`tm_architecture_graph`、`tm_architecture_impact`、按节点查经验（或 tm_search 带 file_path）；可选 `tm_architecture_attach` 挂经验；结果可带「在 Web 打开」深链。

---

## 四、使用路径与心智模型（均衡）

### 4.1 统一话术（文档与引导必用）

- **架构图**：项目的代码地图，看模块和关系。
- **节点**：图上的一个点，代表模块/文件/函数；点开可看详情和关联经验。
- **挂载**：把经验绑到某个节点，方便以后「按代码位置」找到这条经验。

### 4.2 关键路径（≤3 步）

- **看全貌**：Web 点「架构」→ 默认模块级图。
- **按节点查经验**：点节点 → 侧栏「关联经验」；或 MCP/搜索带 file_path。
- **挂经验**：经验创建/编辑选「挂载点」→ 选节点确认；或架构图点节点 →「挂一条经验」。
- **改前估影响**：点节点 →「影响面」；或 MCP「改 X 会影响什么」。

### 4.3 入口与降级

- **Web**：主导航「架构」→ 架构图页（图 + 节点侧栏：调用关系、影响面、关联经验；挂经验入口）。
- **MCP**：只读图/影响面 + 按节点查经验 + 挂载；返回带 Web 深链。
- **降级**：索引/图不可用时提示「架构图暂时不可用，可稍后再试」，**不阻塞**经验保存与检索。

---

## 五、实施步骤（可执行、分阶段）

### Phase 1（MVP 最小闭环）

1. **建表与迁移**  
   新增 `architecture_nodes`、`architecture_edges`、`experience_architecture_binding`（或等价扩展），Alembic 迁移；不破坏现有 Experience 表。

2. **Import 解析与索引**  
   单模块：ast 解析指定目录下 .py 的 Import/ImportFrom，输出 (from_path, to_path)；脚本写节点表与边表（仅 IMPORTS）；Makefile 增加 `make index-architecture`。

3. **影响面查询**  
   实现 project + path + depth 的 BFS 查询（被谁依赖 / 依赖了谁）；内部 API 或直接查库。

4. **只读 API**  
   `GET /api/architecture/graph`、`GET /api/architecture/impact`；供 Web 与 MCP 使用。

5. **经验挂载与按位置检索**  
   - 绑定：经验创建/更新时可写 experience_architecture_binding（或 structured_data）；任务完成时在 sediment 后写绑定（文件级，来自 changed_files 或解析）。
   - 检索：tm_search/tm_solve 支持 file_path/architecture_context，对绑定到该节点/文件/模块的经验过滤或加权。

6. **Web 架构页与易用底线**  
   - 主导航「架构」→ 架构图页，默认模块级图（由节点聚合或目录结构），零配置。
   - 点节点 → 侧栏：关联经验、影响面（调用现有 API）；经验编辑有「挂载点」选择。
   - 文档/帮助中写入「架构图 / 节点 / 挂载」三句话术；索引失败时提示且不阻塞保存/检索。

7. **MCP 工具**  
   `tm_architecture_graph`、`tm_architecture_impact`、按节点查经验（或 tm_search 带 file_path）；可选 `tm_architecture_attach`；返回可带 Web 深链。

### Phase 2（在 MVP 验收后）

- 可选 CALLS、简单模块聚类（基于 IMPORTS）；节点侧栏「执行流」入口（若已有数据）。
- 推荐挂载点（最近修改文件/关键词匹配）；任务详情展示关联架构节点。
- 按节点筛选经验列表/搜索。

### Phase 3+

- 执行流/调用链与经验深度结合；多仓库；更细粒度（函数/类）自动解析与挂载。

---

## 六、验收标准（综合）

| 来源 | 验收项 |
|------|--------|
| 技术最简 | 对 `src/team_memory` 跑通 `make index-architecture`，能查任意文件「依赖我/我依赖」且结果合理 |
| 收益最大 | 任务完成后 sediment 经验具备至少一个架构绑定；带 file_path 的 tm_solve 能优先返回该位置经验 |
| 使用最易 | 打开架构页即有图（零配置）；看全貌/按节点查经验/挂一条经验 ≤3 步；三句话术在帮助中一致；图不可用时不影响保存与检索 |

---

## 七、与 GitNexus 的对比

| 维度 | GitNexus | 本方案（TM 架构可视化） |
|------|----------|--------------------------|
| **定位** | 独立 MCP，通用代码知识图谱（任意仓库） | TM 内置能力，**为经验挂载与按位置检索服务**，与任务/经验深度集成 |
| **图谱节点** | File, Function, Class, Interface, Method, **Community**, **Process** | MVP：**File / Module**；Phase 2+ 可选 Function/Class |
| **边类型** | **CALLS**, IMPORTS, EXTENDS, IMPLEMENTS, DEFINES, MEMBER_OF, **STEP_IN_PROCESS** | MVP：**仅 IMPORTS**；Phase 2 可选 CALLS |
| **执行流** | ✅ 有（Process + 步骤级 STEP_IN_PROCESS），query 按概念找流程 | ❌ MVP 无；Phase 2+ 可接外部/轻量数据 |
| **影响面（blast radius）** | ✅ impact 工具，按 depth 1/2/3 + 置信度，与 process 关联 | ✅ MVP 有，**基于 IMPORTS 的 BFS 1～2 层**，无置信度与 process |
| **存储与索引** | 自建图索引（如 Neo4j/图结构），`npx gitnexus analyze` 全量分析 | **PostgreSQL 双表**，Python **ast 仅解析 import**，`make index-architecture` |
| **语义/概念查询** | ✅ query：按自然语言概念找相关 execution flows | ❌ 无「按概念找流程」；有「按 file_path/节点查经验」 |
| **Cypher / 图查询** | ✅ 支持 raw cypher，灵活图查询 | ❌ 无；用 SQL + 边表 BFS 即可满足 MVP |
| **多仓库** | ✅ list_repos，多仓库发现与切换 | MVP 单项目；Phase 3+ 多仓库 |
| **重构支持** | ✅ rename：多文件协同重命名 + 置信度标签；detect_changes 与 impact 结合 | ❌ 不做重命名/重构工具；只做「看依赖 + 经验挂载」 |
| **与经验/任务** | 无：纯代码图，不挂业务数据 | ✅ **核心差异**：经验–节点绑定、任务完成写节点、**tm_solve/tm_search 按位置检索**、Web 架构页「关联经验」与挂载 |
| **入口** | MCP 资源 + 工具（context/query/impact/rename/detect_changes/cypher） | **Web「架构」页** + MCP（graph/impact/按节点查经验/挂载）+ **只读 REST API** |
| **易用与降级** | 依赖索引新鲜度，stale 需重新 analyze | **零配置出图、三句话术、图不可用时不影响经验保存与检索** |

**总结**：GitNexus 是**通用、深度的代码图**（细粒度符号、执行流、Cypher、重构）；本方案是**TM 内嵌、轻量图谱 + 经验/任务闭环**（文件级 IMPORTS、影响面、经验挂载、按位置检索、任务写节点）。二者可并存：TM 用自建轻量图做「经验按位置」与「任务–架构」；若需执行流/细粒度 CALLS/多仓库，可后续复用或对接 GitNexus 数据。

**若直接复用 GitNexus、在 TM 面板预览**：见 `code-arch-viz-gitnexus-integration.md`（后端代理 MCP / 读 .gitnexus / iframe 三种方式，推荐后端代理 + 架构页 Tab：概览 / 集群 / 图 / 影响面，经验挂载仍 TM 管）。

---

## 八、三方案来源与综合逻辑

- **技术最简化**（`docs/analysis/代码架构可视化-技术最简化方案.md`）：采纳 Python + ast、PostgreSQL 双表、仅 IMPORTS、文件级 MVP、7 步实现、经验挂载约定与只读 API；不采纳图库、执行流、细粒度全量图谱进 MVP。
- **项目收益最大化**（`docs/代码架构可视化-项目收益最大化视角.md`）：采纳 P0/P1/P2 优先级（关联模型、按位置检索、任务完成写节点、影响面、手动挂载）；不采纳 P3/P4 进 MVP（任务–架构展示、模块聚类、执行流深度结合留 Phase 2+）。
- **使用最易**（`docs/代码架构可视化-使用最易视角方案.md`）：采纳零配置出图、≤3 步路径、三句话术、Web 架构页 + 节点侧栏 + 经验挂载/展示、MCP 只读与挂载、错误降级；不采纳「执行流/影响面一键」进 MVP（放 Phase 2），推荐挂载可简化实现。

本均衡方案在**不引入新数据库与重型依赖**的前提下，**优先实现「经验按位置挂载与按位置检索」和「任务完成写节点」**，并满足**零配置、低步数、话术统一、可降级**的易用底线，便于一期交付与后续迭代。
