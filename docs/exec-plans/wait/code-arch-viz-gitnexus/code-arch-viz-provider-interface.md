# 架构能力 Provider 接口 — 预留切换与代码实时结合

> 当前采用 **GitNexus** 作为架构数据源；通过**统一 Provider 抽象**预留切换为**自研方案**的能力，并预留**架构图与代码实时结合**的扩展点。

---

## 一、Provider 枚举与配置

- **配置键**：`architecture.provider`，取值：
  - `gitnexus` — 当前实现：通过 bridge 或 MCP 调用 GitNexus。
  - `builtin` — 自研实现：读 TM 自建表（architecture_nodes/edges）+ ast 索引。
- **未配置或不可用**：架构页展示「架构服务未配置或不可用」，不阻塞经验/任务；API 返回 503 或统一 `{ "available": false, "reason": "..." }`。

```yaml
# config.yaml 示例
architecture:
  provider: gitnexus   # 可选: gitnexus | builtin
  gitnexus:
    bridge_url: "http://127.0.0.1:9321"
  builtin:             # 仅当 provider=builtin 时使用
    repo_path: null    # 默认用项目根或 config 的 workspace
    index_command: "make index-architecture"
```

---

## 二、统一 API 契约（对前端与路由层）

以下为 **TM 后端对外暴露的唯一边界**。前端与 MCP 只依赖这些路径与响应形状；**不**直接依赖 GitNexus 或自研实现细节。

| 方法 | 路径 | Query/Body | 响应形状（统一） |
|------|------|------------|------------------|
| GET | `/api/v1/architecture/context` | `repo` 可选 | `ArchitectureContext` |
| GET | `/api/v1/architecture/graph` | `repo`, `cluster`, `file_path` 可选 | `ArchitectureGraph` |
| GET | `/api/v1/architecture/clusters` | `repo` 可选 | `ClusterSummary[]` |
| GET | `/api/v1/architecture/cluster/{name}` | — | `ClusterMembers` |
| GET | `/api/v1/architecture/impact` | `path`, `depth`, `repo` | `ImpactResult` |
| GET | `/api/v1/architecture/experiences` | `node`, `repo` | `ExperienceRef[]`（TM 侧） |
| GET | `/api/v1/architecture/code-context` | 见下节「代码实时结合」 | `CodeContextResponse`（预留） |

### 2.1 响应类型定义（JSON）

```ts
// 仅作契约说明，实现可用 Python dataclass / Pydantic

interface ArchitectureContext {
  available: boolean;
  repo_name?: string;
  symbols?: number;
  relationships?: number;
  processes?: number;
  stale?: boolean;
  provider?: "gitnexus" | "builtin";  // 可选，便于调试
  reason?: string;                   // available=false 时说明
}

interface ArchitectureGraph {
  nodes: Array<{ id: string; label?: string; kind?: string; path?: string; meta?: object }>;
  edges: Array<{ source: string; target: string; type?: string; meta?: object }>;
  focus_node_id?: string;            // 预留：代码联动时高亮/聚焦的节点
}

interface ClusterSummary {
  name: string;
  cohesion?: number;
  member_count?: number;
}

interface ClusterMembers {
  name: string;
  members: Array<{ id: string; path?: string; kind?: string }>;
}

interface ImpactResult {
  upstream: Array<{ id: string; path?: string; depth?: number }>;
  downstream: Array<{ id: string; path?: string; depth?: number }>;
}

interface ExperienceRef {
  experience_id: string;
  title?: string;
  node?: string;  // 与架构节点 key 一致
}
```

---

## 三、后端实现分层（预留切换）

- **路由层**：`routes/architecture.py` 只做参数校验、鉴权、调用 **Provider 接口**，不写 GitNexus/builtin 具体逻辑。
- **Provider 接口**（Python 抽象）：
  - 定义 `ArchitectureProvider` 协议/抽象类，方法：`get_context()`, `get_graph(...)`, `get_clusters()`, `get_cluster(name)`, `get_impact(path, depth)`，返回上述统一类型。
  - **GitNexusProvider**：内部用 httpx 调 bridge（或 MCP Client），将 GitNexus 返回转为统一类型。
  - **BuiltinProvider**：内部查 `architecture_nodes` / `architecture_edges`，BFS 算 impact，返回统一类型。
- **工厂**：根据 `config.architecture.provider` 实例化对应 Provider；若不可用则返回 `None`，路由层返回 503 或 `available: false`。

这样**切换实现只需改配置与实现类**，路由与前端无需改动。

---

## 四、预留：架构图与代码实时结合

为后续「在编辑器中看当前文件/符号在架构图中的位置」或「在架构图点节点跳转代码」预留扩展点。

### 4.1 请求侧（由前端或 Cursor 传入）

- **现有**：`GET /api/v1/architecture/graph` 已预留 Query `file_path`（可选），用于「只返回与该文件相关的子图」或「标注该文件对应节点」。
- **预留**：
  - `file_path`：当前打开或选中的文件路径（相对项目根）。
  - `symbol`：可选，当前符号（如函数/类名），用于聚焦到更细粒度节点。
  - `cursor_line`：可选，行号，便于后端在自研方案中做行级映射（Phase 2+）。

### 4.2 响应侧（供前端高亮/聚焦）

- **现有**：`ArchitectureGraph` 已预留 `focus_node_id?: string`。
- **约定**：当请求带 `file_path`（及可选 `symbol`）时，Provider 在响应中填充 `focus_node_id`，前端用该 id 在图内高亮或居中该节点。
- **预留**：后续可增加 `highlight_node_ids?: string[]`（如影响面上下游），用于多节点高亮。

### 4.3 后续可选能力（不写死接口）

- **反向跳转**：架构图点节点 → 前端打开「在编辑器中打开该文件」（如 `vscode://file/...` 或 Cursor 协议），由前端或 IDE 扩展实现。
- **实时推送**：若需「编辑器光标移动 → 架构图自动聚焦」，可后续增加 WebSocket 或 SSE 通道，由前端/IDE 推送 `current_file` / `current_symbol`，后端仅做转发或返回 `focus_node_id`；当前不纳入必须 API，仅预留设计说明。

---

## 五、经验与节点标识（与 Provider 无关）

- **节点 key 约定**：无论 provider 是 gitnexus 还是 builtin，TM 侧**经验挂载**与**按位置检索**统一使用**稳定节点 key**：
  - 文件级：`filePath`（如 `src/team_memory/server.py`），与 GitNexus 的 filePath 及自研的 `(project, path)` 对齐。
  - 符号级（可选）：双方约定同一套 id（如 `Function:module.func` 或 GitNexus 的 symbol id），便于切换后经验仍可关联。
- **experience_architecture_binding** 表只存 `node_key`（或 project+path）；不存 provider 名，切换 provider 时无需迁移绑定数据，只要两边对同一文件/符号产出相同 node_key 即可。

---

## 六、文档与实施检查清单

- [ ] `config.yaml` 增加 `architecture.provider` 与 `architecture.gitnexus` / `architecture.builtin` 配置项。
- [ ] 定义 `ArchitectureProvider` 协议及 `ArchitectureContext` 等响应模型（Pydantic）。
- [ ] 实现 `GitNexusProvider`，对接 bridge；路由只调 Provider，不直接写 httpx 到 GitNexus。
- [ ] `/api/v1/architecture/*` 所有响应包含统一形状；`/graph` 支持 `file_path` 且返回 `focus_node_id` 当请求带 file_path 时。
- [ ] 前端架构页仅依赖上述 API；不写死「GitNexus」或「自研」文案，可用 `context.provider` 做可选展示（如「数据源：GitNexus」）。
- [ ] 预留 `GET /api/v1/architecture/code-context`（可选）：入参 `file_path`, `symbol`；返回 `{ "focus_node_id", "related_node_ids", "snippet" }` 等，供后续「代码与图联动」使用。

完成以上，即做到**当前用 GitNexus、后续可切换自研、并为架构图与代码实时结合预留接口**。

---

## 七、如何操作（步骤清单）

**具体执行顺序与每步验证方式**见：**`code-arch-viz-operations.md`**。概要：

1. **配置** — 在 `config.yaml` 增加 `architecture.provider` 与 `architecture.gitnexus.bridge_url`。
2. **Bridge（二选一）** — 搭 GitNexus bridge 服务并配置 URL，或留空由前端展示「未配置」。
3. **后端** — 定义 Pydantic 模型与 `ArchitectureProvider` 协议 → 实现 `GitNexusProvider` → 工厂 `get_provider()`。
4. **路由** — 新增 `routes/architecture.py`，注册 `/architecture/context|graph|clusters|cluster/{name}|impact|experiences`，仅调 Provider。
5. **前端** — 主导航加「架构」、新页面调 context API，有数据则展示概览，无则展示「未配置或不可用」。
6. **验收** — 按操作文档第七步清单逐项打勾。
7. **切换自研** — 实现 `BuiltinProvider`，改配置为 `provider: builtin`，重启即可。
