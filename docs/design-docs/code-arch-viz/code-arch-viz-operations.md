# 架构能力（GitNexus + Provider）— 操作步骤

> 按顺序执行即可完成「TM 面板架构页 + GitNexus 数据源 + 预留切换自研」；每步都可单独验证。

---

## 前置条件

- 本仓库已配置 GitNexus MCP 且可读资源（如 `gitnexus://repo/team_doc/context`），见 `.debug/30-gitnexus-mcp-fix.md`。
- 已执行过 `npx gitnexus analyze`（可选，索引可 stale，但需能连上 MCP 或 bridge）。

---

## 第一步：配置

**1.1 在 `config.yaml` 末尾增加架构配置段**

```yaml
# ╔══════════════════════════════════════════════════════════╗
# ║  [可选] 架构可视化 — GitNexus 或自研，预留切换             ║
# ╚══════════════════════════════════════════════════════════╝
architecture:
  provider: gitnexus   # gitnexus | builtin（自研）
  gitnexus:
    bridge_url: "http://127.0.0.1:9321"   # 见第二步，若暂不搭 bridge 可先留空或注释
  builtin:
    repo_path: null
    index_command: "make index-architecture"
```

**1.2 若使用 bridge（推荐）**：将 `bridge_url` 设为实际地址；**若不使用 bridge**：可先不填或填空字符串，后端实现时需支持「无 bridge 时返回 available: false」，前端展示「架构服务未配置」。

**验证**：启动应用后，能读到 `config.get("architecture", {}).get("provider")` 为 `gitnexus`。

---

## 第二步：GitNexus Bridge（二选一）

**方式 A：使用独立 bridge 服务（推荐，便于 TM 用 HTTP 调 GitNexus）**

1. 新建一个 Node 小服务（或 Python），连接 GitNexus MCP（stdio：`npx -y gitnexus mcp`，cwd 为本仓库根）。
2. 暴露 HTTP：`GET /context`、`GET /graph`、`GET /clusters`、`GET /cluster/:name`、`GET /impact?path=...&depth=2`，返回与 Provider 契约一致的 JSON。
3. 将 bridge 启动在固定端口（如 9321），在 `config.yaml` 中填写 `architecture.gitnexus.bridge_url`。

**方式 B：暂不搭 bridge**

- 将 `bridge_url` 留空；后端 `GitNexusProvider` 在无 bridge 时返回 `available: false`，前端架构页展示「架构服务未配置或不可用」，并提示「可配置 bridge 或使用自研方案」。

**验证**：若已搭 bridge，`curl http://127.0.0.1:9321/context` 应返回含 symbols/relationships 的 JSON。

---

## 第三步：后端 — 模型与 Provider 接口

**3.1 定义响应模型（Pydantic）**

- 新建 `src/team_memory/web/architecture_models.py`（或放在现有 schemas 下），定义：
  - `ArchitectureContext`
  - `ArchitectureGraph`（含 `nodes`, `edges`, `focus_node_id`）
  - `ClusterSummary`, `ClusterMembers`, `ImpactResult`, `ExperienceRef`
- 与 `code-arch-viz-provider-interface.md` 第二节的 JSON 形状一致。

**3.2 定义 Provider 协议**

- 新建 `src/team_memory/architecture/base.py`（或 `web/architecture/providers/base.py`）：
  - 定义协议/抽象类 `ArchitectureProvider`，方法：
    - `get_context(repo: str | None) -> ArchitectureContext`
    - `get_graph(repo, cluster, file_path: str | None) -> ArchitectureGraph`
    - `get_clusters(repo) -> list[ClusterSummary]`
    - `get_cluster(name: str, repo) -> ClusterMembers`
    - `get_impact(path: str, depth: int, repo) -> ImpactResult`
  - 各方法返回类型使用 3.1 的模型。

**3.3 实现 GitNexusProvider**

- 新建 `src/team_memory/architecture/gitnexus_provider.py`（或同目录下）：
  - 从配置读取 `architecture.gitnexus.bridge_url`。
  - 使用 `httpx.AsyncClient` 调 bridge 的 `/context`、`/graph`、`/clusters`、`/cluster/:name`、`/impact`。
  - 将 bridge 返回转换为上述 Pydantic 模型；若 bridge 不可达或超时，返回 `ArchitectureContext(available=False, reason="...")` 等。
  - 当 `get_graph` 收到 `file_path` 时，若 bridge 支持，在响应中填 `focus_node_id`；否则可先不填。

**3.4 Provider 工厂**

- 在 `architecture/` 或 `web/architecture/` 下实现 `get_provider(config) -> ArchitectureProvider | None`：
  - 若 `config.architecture.provider == "gitnexus"` 且 `bridge_url` 非空，返回 `GitNexusProvider(bridge_url=...)`；
  - 否则返回 `None`（或返回一个返回 `available: false` 的占位实现）。

**验证**：单元测试或手动构造 `get_provider`，调用 `get_context()`，有 bridge 时得到 `available: True` 的上下文。

---

## 第四步：后端 — 路由

**4.1 新增路由模块**

- 新建 `src/team_memory/web/routes/architecture.py`：
  - 依赖注入：从 app 或 config 获取 `get_provider()` 得到的 Provider（若为 None，各端点返回 503 或 `{"available": false, "reason": "..."}`）。
  - 实现：
    - `GET /architecture/context` → Provider.get_context()
    - `GET /architecture/graph` → Provider.get_graph()，query 支持 `repo`, `cluster`, `file_path`
    - `GET /architecture/clusters` → Provider.get_clusters()
    - `GET /architecture/cluster/{name}` → Provider.get_cluster(name)
    - `GET /architecture/impact` → Provider.get_impact()，query：`path`, `depth`, `repo`
    - `GET /architecture/experiences` → 查 TM 库（experience_architecture_binding 或 code_anchor），query：`node`, `repo`，返回经验列表
  - 所有响应使用 3.1 的模型，保证形状统一。
  - 鉴权：与现有 API 一致（如 `get_current_user`）。

**4.2 注册路由**

- 在 `src/team_memory/web/routes/__init__.py` 的 `mount_all` 中：
  - `from team_memory.web.routes import architecture`
  - `parent_router.include_router(architecture.router)`（注意 prefix 若在父级已统一为 `/api/v1`，则路由内路径为 `/architecture/...`）。

**验证**：启动 Web，`curl http://localhost:9111/api/v1/architecture/context`（带鉴权头或 cookie）返回 JSON；无 bridge 时返回 503 或 `available: false`。

---

## 第五步：前端 — 架构页与导航

**5.1 主导航增加「架构」**

- 在 `src/team_memory/web/static/index.html` 中，与「任务列表」「经验管理」「语义搜索」等并列增加：
  - 链接：`<a ... data-page="architecture">架构</a>`，点击触发 `navigate('architecture')`。

**5.2 新增架构页面容器**

- 同一文件中增加 `id="page-architecture"` 的 `div.page`，内容可为：
  - 顶部标题「架构」；
  - 占位区域：如「概览」「集群」「图」「影响面」四个 Tab 的容器（或先做一个「概览」+ 一句「数据源：GitNexus」）。

**5.3 加载逻辑**

- 在 `navigate` 或页面加载逻辑中，当 `page === 'architecture'` 时：
  - 调用 `GET /api/v1/architecture/context`；
  - 若 `available === false`，展示「架构服务未配置或不可用」及配置说明（或 `reason`）；
  - 若 `available === true`，展示概览（symbols、relationships、processes、stale 提示），并可展示 `provider`（如「数据源：GitNexus」）。

**5.4 可选：图、集群、影响面**

- 后续再实现：调用 `/graph`、`/clusters`、`/impact`，用图库渲染图、列表展示集群与影响面；节点侧栏调 `/architecture/experiences?node=...`。第一步可只做「概览 + 不可用时的降级文案」。

**验证**：打开 TM Web，点「架构」，应进入架构页；有 bridge 时看到概览，无 bridge 时看到「未配置或不可用」。

---

## 第六步：经验与节点（可选，与首期架构页并行）

- **经验挂载**：经验创建/编辑时增加「挂载点」选择（filePath 或节点 id）；写入 experience_architecture_binding 或 structured_data.code_anchor。
- **按节点查经验**：`/architecture/experiences?node=...` 已在上一步由后端实现，前端在节点侧栏调用即可。
- **任务完成写节点**：在 task 完成时，将 sediment 经验与 `changed_files` 或解析出的 path 写入 binding；可与现有任务完成接口一起做。

---

## 第七步：验收清单

- [ ] 配置中有 `architecture.provider: gitnexus` 和 `architecture.gitnexus.bridge_url`（或留空）。
- [ ] 有 bridge 时：`GET /api/v1/architecture/context` 返回 `available: true` 及 symbols/relationships 等。
- [ ] 无 bridge 时：同一接口返回 `available: false` 或 503，前端显示「未配置或不可用」。
- [ ] Web 主导航有「架构」，点击进入架构页，不报错。
- [ ] 响应形状与 Provider 接口文档一致（便于后续切换 builtin）。
- [ ] `/graph` 支持 query `file_path`，且响应中有 `focus_node_id` 预留（可为 null）。

---

## 后续切换自研

1. 实现 `BuiltinProvider`（读 architecture_nodes/edges，BFS 算 impact），实现同一套接口方法。
2. 在工厂中当 `provider == "builtin"` 时返回 `BuiltinProvider`。
3. 将 `config.yaml` 中 `architecture.provider` 改为 `builtin`，重启服务。
4. 前端无需改；仅数据源从 GitNexus 变为自建表。

---

## 相关文档

- 接口契约与预留：`code-arch-viz-provider-interface.md`
- GitNexus 接入方案：`code-arch-viz-gitnexus-integration.md`
- GitNexus MCP 识别失败：`.debug/30-gitnexus-mcp-fix.md`
