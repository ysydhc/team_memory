# 复用 GitNexus 接入方案 — 在 TM 面板直接预览

> 不自建图谱，直接复用现有 GitNexus 能力，在 team_memory Web 面板中提供「架构」入口并预览图/集群/影响面；经验挂载仍由 TM 管理（按节点 key 关联）。  
> **已确认采用本方案**；并**预留接口**便于后续切换为自研方案、以及**架构图与代码实时结合**，见 `code-arch-viz-provider-interface.md`。

---

## 一、目标与约束

- **目标**：用户打开 TM Web → 点「架构」→ 在 TM 内直接看到代码架构（图/集群/影响面），无需切到 Cursor 或单独开 GitNexus。
- **约束**：GitNexus 当前以 MCP 形式存在（资源 + 工具），浏览器无法直连 MCP；索引在 `.gitnexus`（或 GitNexus 自管存储），需通过「可被 TM 后端或前端访问」的方式拿到数据。
- **保留 TM 价值**：经验–节点关联、任务完成写节点、按位置检索仍由 TM 实现；节点标识与 GitNexus 对齐（如 `filePath`、symbol id）即可。

---

## 二、GitNexus 数据从哪里来（三种可行方式）

| 方式 | 说明 | 适用条件 |
|------|------|----------|
| **A. 后端代理 MCP** | TM 后端启动或连接 GitNexus MCP（stdio/SSE），按需调用 query/context/impact，转成 JSON 供前端 | GitNexus 支持被另一进程以 Client 身份调用（MCP Client in Python/Node） |
| **B. 读取索引文件** | GitNexus 将索引写入 `.gitnexus/` 或约定路径；TM 后端读取并解析（若格式公开），提供 graph/clusters/impact API | `.gitnexus` 结构稳定、可读（JSON/SQLite 等） |
| **C. GitNexus 独立 HTTP 服务** | 若 GitNexus 提供 `gitnexus serve --port 9xxx` 一类 HTTP 服务，TM 前端 iframe 或后端代理该 URL | 官方或社区提供 HTTP 模式 |

当前仓库已有 `npx gitnexus analyze` 与 MCP 资源（context、clusters、processes、schema），**优先按 A（后端代理 MCP）设计**；若后续确认 B 或 C 更简单，可替换数据源，接口不变。

---

## 三、架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│  TM Web 前端                                                     │
│  ┌─────────────┐  ┌──────────────────────────────────────────┐ │
│  │ 主导航      │  │ 架构页：图 / 集群 / 影响面 / 关联经验       │ │
│  │ + 架构      │  │ - 调用 TM 后端 /api/v1/architecture/*     │ │
│  └─────────────┘  │ - 或 iframe 到「架构 UI 源」（若采用 C）    │ │
│                   └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  TM 后端 (FastAPI)                                               │
│  /api/v1/architecture/context   → 概览 + 陈旧度                 │
│  /api/v1/architecture/graph     → 节点+边（可限定 cluster）      │
│  /api/v1/architecture/clusters  → 集群列表                       │
│  /api/v1/architecture/cluster/{name} → 某集群成员                │
│  /api/v1/architecture/impact    → 影响面（path + depth）        │
│  /api/v1/architecture/experiences?node=... → 该节点关联经验(TM) │
└─────────────────────────────────────────────────────────────────┘
        │                                    │
        │ (A) 调用 GitNexus MCP               │ (B) 读 .gitnexus
        ▼                                    ▼
┌──────────────────┐              ┌──────────────────┐
│ GitNexus MCP      │              │ .gitnexus/       │
│ (stdio/SSE)      │              │ 索引文件         │
│ 或 (C) HTTP 服务  │              │                  │
└──────────────────┘              └──────────────────┘
```

- **经验数据**：始终在 TM；`/architecture/experiences?node=...` 查 TM 的 experience_architecture_binding（按 node key = filePath 或 GitNexus symbol id）。

---

## 四、方案 A：TM 后端代理 GitNexus MCP（推荐）

### 4.1 思路

- TM 后端在**同一台机器**上通过 MCP Client 连接 GitNexus（例如子进程 stdio，或连接已启动的 GitNexus SSE 端点）。
- 收到前端请求时，后端调用 GitNexus 的 `context` / `query` / `impact` 等，将结果转为统一 JSON，返回给前端。
- 索引仍由用户在本机执行 `npx gitnexus analyze`（或 CI）；TM 只消费，不负责建索引。

### 4.2 实现要点

1. **MCP Client 选型**  
   - **Node**：若 GitNexus 为 Node MCP Server，可用 `@modelcontextprotocol/sdk` 在 TM 侧起一个 Node 子进程或轻量服务，TM 用 HTTP 调该服务。  
   - **Python**：若存在 Python 的 MCP Client 库，可在 TM 进程内直接调 GitNexus（需 GitNexus 以 stdio 方式可被 spawn）。  
   - 折中：**单独小服务「gitnexus-bridge」**（Node 或 Python），负责连 GitNexus MCP 并暴露 `GET /context`, `GET /graph`, `GET /impact?path=...`；TM 后端用 httpx 调 bridge，无语言绑定。

2. **配置**  
   - `config.yaml` 增加可选段，例如：  
     ```yaml
     architecture:
       provider: gitnexus   # 仅支持 gitnexus
       gitnexus:
         # 方式1：后端直接调 bridge
         bridge_url: "http://127.0.0.1:9321"
         # 方式2：若后端内嵌 MCP Client，则 repo_path + command
         # repo_path: "/path/to/team_doc"
         # analyze_command: "npx gitnexus analyze"
       ```
   - 未配置或 bridge 不可达时，架构页显示「架构服务未配置或不可用」，不阻塞经验/任务。

3. **API 契约（TM 后端）**  
   - `GET /api/v1/architecture/context`  
     - 返回：`{ "repo_name": "team_memory", "symbols": 1607, "relationships": 4718, "processes": 134, "stale": false }`  
     - 来源：GitNexus context 资源或 bridge 的 `/context`。  
   - `GET /api/v1/architecture/graph`  
     - Query：`?repo=team_memory&cluster=可选`  
     - 返回：`{ "nodes": [...], "edges": [...] }`（可先只返回 clusters 聚合视图，减少前端负载）。  
     - 来源：GitNexus query/cypher 或 bridge 的 `/graph`。  
   - `GET /api/v1/architecture/clusters`  
     - 返回：`[{ "name": "...", "cohesion": ... }, ...]`  
     - 来源：GitNexus clusters 资源或 bridge。  
   - `GET /api/v1/architecture/cluster/{name}`  
     - 返回：该集群成员（文件/符号列表）。  
   - `GET /api/v1/architecture/impact`  
     - Query：`?path=src/foo/bar.py&depth=2`  
     - 返回：`{ "upstream": [...], "downstream": [...] }`  
     - 来源：GitNexus impact 工具或 bridge。  
   - `GET /api/v1/architecture/experiences`  
     - Query：`?node=src/foo/bar.py` 或 `?node=Function:bar.baz`  
     - 返回：TM 侧绑定到该节点的经验列表（来自 experience_architecture_binding 或 code_anchor/file_path）。  
     - 纯 TM 逻辑，不经过 GitNexus。

### 4.3 前端「架构」页

- **导航**：主导航增加「架构」，对应 `#architecture`，与现有 list/tasks/search/usage/settings 一致。
- **内容**：  
  - **Tab1：概览** — 调 `context`，展示 symbols/relationships/processes 及 stale 提示；若 stale，提示「请在本机运行 npx gitnexus analyze」并给出命令。  
  - **Tab2：集群** — 调 `clusters` 列表，点某集群调 `cluster/{name}` 看成员；可选「在图上看」拉取 `graph?cluster=xxx`。  
  - **Tab3：图** — 调 `graph`，用前端图库（如 Cytoscape.js / D3 / vis-network）渲染节点与边；支持点节点 → 侧栏展示「影响面」「关联经验」（再调 impact + experiences）。  
  - **Tab4：影响面** — 输入 path（或从图选节点），调 `impact`，展示上下游。
- **关联经验**：节点侧栏中「关联经验」调用 `/api/v1/architecture/experiences?node=...`，展示 TM 经验列表并跳转经验详情；「挂载经验」仍走现有经验编辑 + 挂载点（写 experience_architecture_binding，node 与 GitNexus 的 filePath/symbol 对齐）。

### 4.4 依赖与部署

- **GitNexus**：用户本机已安装且已 `npx gitnexus analyze`（或 CI 写 .gitnexus）。  
- **Bridge（若采用）**：与 TM 同机或同网，TM 配置 `bridge_url` 即可；bridge 需配置 GitNexus 的 repo 路径或 MCP 连接方式。  
- **TM**：无新增重型依赖；仅增加路由与 httpx 调 bridge（或内嵌 MCP Client 时增加对应库）。

---

## 五、方案 B：TM 后端直接读 .gitnexus（若格式可用）

- 若 GitNexus 索引为**文件形式**（如 `.gitnexus/graph.json` 或 SQLite），且格式稳定或文档化：  
  - TM 配置 `repo_path`（或从 project 配置推导），后端直接读该路径下 `.gitnexus`。  
  - 实现 `/graph`、`/clusters`、`/impact` 的本地解析（如 impact 用 BFS 在内存图计算），无需 MCP 或 bridge。  
- **优点**：无额外进程、延迟低。**前提**：需确认 `.gitnexus` 结构并随 GitNexus 版本做兼容。

---

## 六、方案 C：iframe 嵌入 GitNexus 自带 UI

- 若 GitNexus 提供 **Web UI**（如 `gitnexus serve --port 9200`）：  
  - TM 仅新增「架构」页，内嵌 iframe，`src` 为配置项（如 `http://localhost:9200` 或通过 TM 后端代理 `/api/v1/architecture/proxy?path=/` 避免跨域）。  
  - 经验挂载仍在 TM：架构页可再提供「当前节点」→「在 TM 中查看/挂载经验」的入口（通过 URL 参数或 postMessage 与 iframe 协同，若 GitNexus UI 支持）。  
- **优点**：实现量最小。**缺点**：依赖 GitNexus 是否有官方 UI、是否可传 node 参数；且「关联经验」需在 TM 侧做第二块 UI（如侧栏）。

---

## 七、经验与任务在「复用 GitNexus」下的做法

- **节点标识对齐**：TM 的 experience_architecture_binding 使用与 GitNexus 一致的 key，例如：  
  - 文件级：`filePath`（如 `src/team_memory/server.py`）；  
  - 符号级：若 GitNexus 暴露稳定 id，可用 `symbolId` 或 `symbol:name`。  
- **任务完成写节点**：与自建方案相同：任务完成时把 sediment 经验绑定到「本次改动的文件/节点」；节点从 `changed_files` 或 title/description 解析，写入 binding 表。  
- **按位置检索**：tm_solve/tm_search 带 file_path 或 architecture_context 时，优先返回 binding 到该 path/节点的经验（逻辑不变，仅数据源为 GitNexus 时节点 key 与之一致）。

---

## 八、推荐实施顺序（方案 A）

1. **Bridge 最小实现**（若选 bridge）：用 Node 或 Python 写一个小服务，连接 GitNexus MCP，暴露 `/context`、`/clusters`、`/cluster/:name`、`/graph`、`/impact`；TM 配置 `bridge_url`。  
2. **TM 后端**：新增 `routes/architecture.py`，注册上述 `/api/v1/architecture/*`，内部用 httpx 调 bridge；`/experiences` 查 TM 库。  
3. **TM 前端**：主导航加「架构」，新页面 3～4 个 Tab（概览 / 集群 / 图 / 影响面），图用简单图库渲染；节点侧栏调 impact + experiences。  
4. **配置与降级**：无配置或 bridge 不可达时，架构页展示「架构服务未配置或不可用」，并提示配置 `architecture.gitnexus.bridge_url` 或运行 `npx gitnexus analyze`。  
5. **经验挂载**：经验编辑/创建时「挂载点」支持按 filePath 或节点选择；binding 表与 GitNexus 的 filePath 对齐；任务完成时写入 binding（文件级）。

---

## 九、与「自建图谱」方案的对比

| 项 | 自建（原均衡方案） | 复用 GitNexus（本方案） |
|----|--------------------|---------------------------|
| 图谱数据 | TM 自己 ast 解析、PostgreSQL 双表 | 完全用 GitNexus（执行流、CALLS、聚类、Cypher 全有） |
| 实现量 | 建表、解析器、索引、BFS 影响面、前端图 | 后端代理/读索引 + 前端图 + 经验绑定 |
| 执行流 / 细粒度 | Phase 2+ 再扩展 | 直接用 GitNexus 已有能力 |
| 经验/任务 | TM 全管 | 不变，仍 TM 管；仅节点 key 与 GitNexus 对齐 |
| 依赖 | 仅 Python + PG | 需 GitNexus 可用（+ 可选 bridge） |
| 索引更新 | `make index-architecture` | 用户/CI 执行 `npx gitnexus analyze` |

结论：若团队**已用或可接受 GitNexus**，优先采用本接入方案，在 TM 面板直接预览并保留「经验挂载 + 按位置检索 + 任务写节点」的 TM 闭环；自建方案可作为「无 GitNexus 环境」或「完全内网无 Node」时的备选。

---

## 十、预留切换与代码实时结合（必读）

- **Provider 抽象**：后端不直接写死 GitNexus 调用，而是通过 **ArchitectureProvider** 接口（context / graph / clusters / impact）取数；配置 `architecture.provider: gitnexus | builtin` 即可切换实现，详见 **`code-arch-viz-provider-interface.md`**。
- **切换自研**：实现 `BuiltinProvider`（读 architecture_nodes/edges + 自研索引），在 config 中改为 `provider: builtin`，路由与前端无需改动。
- **架构图与代码实时结合**：同一文档中预留了：
  - 请求：`/graph` 的 `file_path`、`symbol`、`cursor_line`（可选）；
  - 响应：`focus_node_id`、后续可扩展 `highlight_node_ids`；
  - 可选接口：`GET /architecture/code-context?file_path=...&symbol=...` 供编辑器/前端「当前代码 → 图中聚焦」；
  - 反向跳转、WebSocket 推送等留作后续扩展，不写死当前 API。
- 实施时**先按 Provider 接口实现 GitNexusProvider**，再挂到路由层，便于后续无缝切换与扩展。

- **操作步骤（按顺序执行）**：见 **`code-arch-viz-operations.md`**（配置 → Bridge → 后端模型与 Provider → 路由 → 前端架构页 → 验收与切换自研）。
