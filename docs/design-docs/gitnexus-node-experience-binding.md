# GitNexus 与节点经验绑定 — 技术设计文档

> 版本：1.0 | 日期：2025-03-11  
> 面向：新加入项目的开发者、需要理解架构的维护者

---

## 一、概述

### 1.1 背景与目标

TeamMemory（TM）是团队经验库，支持 MCP 工具（`tm_save`、`tm_solve`、`tm_search` 等）和 Web 界面。**节点经验绑定**将经验与代码架构图中的节点关联，实现：

1. **架构页展示**：在代码架构图侧边栏查看某节点关联的经验
2. **检索加分**：Agent 调用 `tm_solve` / `tm_search` 时，若传入 `file_paths`，绑定到这些路径的经验会获得分数加成
3. **设计意图传承**：Agent 修改代码时能看到相关历史经验，减少重复踩坑

### 1.2 核心概念

| 概念 | 说明 |
|------|------|
| **GitNexus** | 代码知识图谱工具，通过 `npx gitnexus analyze` 分析代码，生成 `.gitnexus` 索引，提供节点（File、Function、Class、Method 等）和边（CALLS、IMPORTS 等） |
| **node_key** | 经验与节点关联的键，可为文件路径（如 `src/team_memory/server.py`）或完整 node id（如 `Function:src/team_memory/server.py:tm_save`） |
| **ExperienceArchitectureBinding** | 绑定表，存储 `(experience_id, node_key, project)`，一条经验可绑定多个节点 |

---

## 二、GitNexus 架构

### 2.1 数据流

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  源代码仓库                                                                               │
│  (Python/JS/... 文件)                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
         │
         │  npx gitnexus analyze
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  .gitnexus/                                                                               │
│  ├── meta.json          # 索引元数据、统计信息                                             │
│  └── (图数据)            # 节点与边                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
         │
         │  npx gitnexus cypher "MATCH ..."
         │  npx gitnexus impact ...
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  GitNexus Bridge (tools/gitnexus-bridge/server.js)                                        │
│  HTTP API: /context, /clusters, /cluster/:name, /graph, /impact, /search, /cypher         │
│  端口: 9321                                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
         │
         │  httpx 调用
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  GitNexusProvider (src/team_memory/architecture/gitnexus_provider.py)                      │
│  实现 ArchitectureProvider 接口，供 Web 架构路由调用                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 节点 id 格式

GitNexus 节点 id 遵循 `Kind:path:name` 格式（部分节点例外）：

| 节点类型 | id 示例 | path | name |
|----------|---------|------|------|
| File | `File:src/team_memory/server.py` 或 `src/team_memory/server.py` | `src/team_memory/server.py` | - |
| Function | `Function:src/team_memory/server.py:tm_save` | `src/team_memory/server.py` | `tm_save` |
| Class | `Class:src/team_memory/services/experience.py:ExperienceService` | `src/team_memory/services/experience.py` | `ExperienceService` |
| Method | `Method:src/.../experience.py:ExperienceService:search` | 同上 | `search` |

**解析规则**（前端 `architecture-viewer.js`）：

```javascript
// pathFromNodeId: Kind:path:name -> path
// parts.length >= 3: parts.slice(1, -1).join(':')
// File 节点: parts[1]

// nodeNameFromNodeId: Kind:path:name -> name
// parts.length >= 3: parts[parts.length - 1]

// kindFromNode: Kind:path:name -> Kind
// parts[0]
```

### 2.3 Meta 节点（排除）

以下节点不参与架构图展示，仅用于内部聚类/流程：

- `comm_<num>`：Community（集群）
- `proc_<num>_<name>`：Process（执行流）
- `Folder:<path>`：目录结构

---

## 三、存储层设计

### 3.1 表结构

```sql
-- experience_architecture_bindings
CREATE TABLE experience_architecture_bindings (
    id UUID PRIMARY KEY,
    experience_id UUID NOT NULL REFERENCES experiences(id) ON DELETE CASCADE,
    node_key VARCHAR(500) NOT NULL,
    project VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(experience_id, node_key)
);
CREATE INDEX idx_exp_arch_binding_node_key ON experience_architecture_bindings(node_key);
CREATE INDEX idx_exp_arch_binding_experience_id ON experience_architecture_bindings(experience_id);
```

### 3.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `experience_id` | UUID | 关联的经验 id |
| `node_key` | String(500) | 与 GitNexus 对齐的键，可为文件路径或完整 node id |
| `project` | String(100), nullable | 项目标识，用于多项目隔离 |

### 3.3 node_key 规范化

`normalize_node_key()`（`src/team_memory/utils/path_utils.py`）：

- 去除首尾空白
- 反斜杠转正斜杠
- 去除 `./` 前缀
- 去除 leading `/`
- 空字符串返回 `""`

**注意**：当前实现仅针对路径格式，对 `Kind:path:name` 格式会原样返回（不解析中间 path 部分）。

---

## 四、写入流程

### 4.1 入口汇总

| 入口 | 调用方 | node_key 来源 | 代码位置 |
|------|--------|---------------|----------|
| MCP `tm_save` | Agent / 规则 | `architecture_nodes` 参数 | `server.py` L1299-1302 |
| MCP `tm_save_typed` | Agent / 规则 | `architecture_nodes` 参数 | `server.py` L1410-1413 |
| MCP `tm_task` (completed) | Agent | Git 变更文件列表 | `server.py` L2580-2625 |
| MCP `tm_solve` / `tm_search` | Agent | `file_paths` → `node_keys`（仅用于 boost，不写入） | - |
| Web 创建经验 | 用户 | 表单「架构挂载点」每行一个 | `experiences.py` L345 |
| Web 编辑经验 | 用户 | 同上 | `experiences.py` L449-450 |
| 架构页「新建经验并挂载到此节点」 | 用户 | `nodeKey = path \|\| nodeId` | `architecture-viewer.js` L659, L738 |

### 4.2 写入调用链

```
入口 (tm_save / Web / ...)
    │
    ├─ architecture_nodes 或 file_paths
    │
    ▼
normalize_node_key(p) for p in nodes
    │
    ▼
ExperienceService.save() / update()
    │
    └─ architecture_nodes=normalized_nodes
           │
           ▼
       ExperienceRepository.replace_architecture_bindings(experience_id, node_keys)
           │
           ├─ DELETE FROM experience_architecture_bindings WHERE experience_id = ?
           └─ INSERT INTO experience_architecture_bindings (experience_id, node_key, project) VALUES (...)
```

### 4.3 关键代码片段

**server.py（tm_save）**：

```python
normalized_nodes = (
    [k for k in (normalize_node_key(p) for p in architecture_nodes) if k]
    if architecture_nodes
    else None
)
await service.save(..., architecture_nodes=normalized_nodes)
```

**architecture-viewer.js（挂载按钮）**：

```javascript
const nodeKey = path || nodeId;  // 有 path 时用 path，否则用 nodeId
// ...
const mountBtn = `<button ... data-mount-node="${esc(nodeKey)}">新建经验并挂载到此节点</button>`;
```

**store.js**：点击挂载后设置 `architectureMountNode = nodeKey`，打开创建模态框时预填。

---

## 五、读取流程

### 5.1 架构页侧边栏

**流程**：

1. 用户点击图中节点 → `highlightArchitectureNode(nodeId, nodePath)`
2. 调用 `showArchitectureNodeSidebar(nodeId, path)`
3. `nodeKey = path || nodeId`
4. `GET /api/v1/architecture/experiences?node={nodeKey}`
5. `ExperienceRepository.list_experiences_by_node(node_key=nodeKey)`
6. SQL: `WHERE node_key = ?` 精确匹配
7. 返回经验列表，渲染侧边栏

**API**：`GET /architecture/experiences?node=<node_key>&project=&repo=`

### 5.2 tm_solve / tm_search 的 node_keys_boost

**流程**：

1. Agent 调用 `tm_solve(problem=..., file_paths=[...])` 或 `tm_search(..., file_paths=[...])`
2. `node_keys = [normalize_node_key(p) for p in file_paths]`
3. 执行向量/FTS 检索，得到 `results`
4. `_apply_node_keys_boost(results, node_keys, ...)`：
   - `list_experiences_by_nodes(node_keys)` → 命中绑定的 experience_id 集合
   - 对 `results` 中 id 在集合内的经验：`score += 0.15`
   - 按 score 重新排序
5. 返回排序后的结果

**关键逻辑**（`experience.py` L136-154）：

```python
async def _apply_node_keys_boost(self, results, node_keys, project, user_name, repo):
    rows = await repo.list_experiences_by_nodes(node_keys, project=project, current_user=user_name)
    bound_ids = {r["experience_id"] for r in rows}
    for r in results:
        eid = r.get("group_id") or r.get("id")
        if eid and str(eid) in bound_ids:
            base = r.get("score", r.get("similarity", 0))
            r["score"] = base + 0.15
    results.sort(key=lambda x: x.get("score", x.get("similarity", 0)), reverse=True)
```

### 5.3 读取调用链

```
架构页: nodeKey → GET /architecture/experiences?node= → list_experiences_by_node(node_key)
        精确匹配 node_key = ?

检索:   file_paths → node_keys → search() → _apply_node_keys_boost()
        list_experiences_by_nodes(node_keys) → node_key IN (...)
        命中则 score + 0.15
```

---

## 六、问题与解法

### 6.1 问题 1：架构页 Function 节点存成文件路径

**现状**：`nodeKey = path || nodeId`，有 `path` 时一律用 `path`。Function 节点的 `path` 来自 `pathFromNodeId(nodeId)`，即文件路径，导致符号级节点也存成文件路径。

**影响**：无法区分「绑定到文件」与「绑定到该文件下某函数」，后续做文件级聚合时难以精确统计。

**解法**：改为 `nodeKey = (kindFromNode(nodeId) === 'File') ? (path || nodeId) : nodeId`，File 节点用 path，其余用完整 nodeId。

### 6.2 问题 2：文件级聚合查询复杂

**现状**：`list_experiences_by_node` 仅做 `node_key = ?` 精确匹配。若需「查文件时返回该文件 + 该文件下所有符号的绑定」，需解析 `node_key` 格式并做 `LIKE` 或子串匹配，易出错且难建索引。

**解法**：增加 `file_path` 列，写入时从 `node_key` 解析并填充。文件级查询用 `WHERE file_path = ?`，符号级仍用 `WHERE node_key = ?`。

### 6.3 问题 3：tm_solve 无 symbol 上下文

**现状**：MCP 只传 `file_paths`，无法得知「当前编辑的是哪个函数」。检索时只能按文件 boost，无法精确到符号。

**解法**：短期用 `file_path` 做文件级聚合；中期在 MCP 或 Cursor 集成中传入「当前编辑符号」上下文。

### 6.4 问题 4：重构后绑定变孤儿

**现状**：函数重命名、文件移动后，`node_key` 不再对应任何 GitNexus 节点，绑定成为孤儿。

**解法**：每晚跑归档脚本，调用 GitNexus 获取当前有效节点集合，将 `node_key` 不在集合内的绑定标记 `archived_at`。

### 6.5 问题 5：查询逻辑与格式耦合

**现状**：Repository 直接使用 `node_key`，格式判断分散在各处。

**解法**：引入 Node Binding Resolver 中间层，按规则解析输入，输出统一查询条件（`file_path` 或 `node_key`），Repository 只接收解析结果。

---

## 七、配置与部署

### 7.1 配置项

`config.yaml`：

```yaml
architecture:
  provider: gitnexus  # gitnexus | builtin
  gitnexus:
    bridge_url: "http://127.0.0.1:9321"
```

### 7.2 启动顺序

1. 在项目根目录执行 `npx gitnexus analyze` 生成索引
2. 启动 GitNexus Bridge：`node tools/gitnexus-bridge/server.js` 或 `make gitnexus-restart`
3. 启动 TM Web / MCP

### 7.3 健康检查

```bash
curl -s http://127.0.0.1:9321/context
# 返回 available: true 表示 Bridge 正常
```

---

## 八、关键文件索引

| 模块 | 文件 | 职责 |
|------|------|------|
| 模型 | `src/team_memory/storage/models.py` | `ExperienceArchitectureBinding` 定义 |
| 存储 | `src/team_memory/storage/repository.py` | `list_experiences_by_node`、`list_experiences_by_nodes`、`replace_architecture_bindings`、`get_architecture_bindings` |
| 工具 | `src/team_memory/utils/path_utils.py` | `normalize_node_key()` |
| 服务 | `src/team_memory/services/experience.py` | `_apply_node_keys_boost`，search/solve 中调用 |
| MCP | `src/team_memory/server.py` | `tm_save`、`tm_save_typed`、`tm_task`、`tm_solve` 的 architecture_nodes / file_paths 处理 |
| Web 路由 | `src/team_memory/web/routes/architecture.py` | `GET /architecture/experiences` |
| Web 路由 | `src/team_memory/web/routes/experiences.py` | 创建/编辑经验的 `architecture_nodes` 处理 |
| 前端 | `src/team_memory/web/static/js/architecture-viewer.js` | 架构图、侧边栏、`nodeKey`、`pathFromNodeId`、`nodeNameFromNodeId`、`kindFromNode` |
| 前端 | `src/team_memory/web/static/js/store.js` | `architectureMountNode` 预填 |
| Bridge | `tools/gitnexus-bridge/server.js` | GitNexus CLI 封装、HTTP API |
| Provider | `src/team_memory/architecture/gitnexus_provider.py` | 调用 Bridge 的 Provider 实现 |
| 协议 | `src/team_memory/architecture/base.py` | `ArchitectureProvider` 抽象 |

---

## 九、扩展规划

### 9.1 细粒度节点绑定

- 支持 Function/Class/Method 级 `node_key`
- 前端按节点类型选择 path 或 nodeId
- 后端增加 `file_path` 列，支持文件级聚合

### 9.2 归档脚本

- 每晚执行，调用 Bridge Cypher 获取当前有效节点
- 将孤儿绑定标记 `archived_at`
- 可选：智能重绑定（根据 Git 历史推断新位置）

### 9.3 Node Binding Resolver

- 输入：node_key 或 file_path
- 规则：解析格式，决定匹配策略
- 输出：统一查询条件，隔离 Repository 与格式细节

---

## 十、附录：数据流总览图

```
                        存入
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Agent/用户  →  tm_save / Web / 架构页  →  normalize_node_key             │
    │       │                    │                        │                   │
    │       │  architecture_     │  nodeKey = path||nodeId │  node_key         │
    │       │  nodes / file_paths│                        │  + project         │
    │       └────────────────────┴────────────────────────┴──────────────────│
    │                                         │                               │
    │                                         ▼                               │
    │                    replace_architecture_bindings()                      │
    │                    → experience_architecture_bindings 表                 │
    └────────────────────────────────────────────────────────────────────────┘

                        读取
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  架构页:  nodeKey  →  GET /architecture/experiences  →  list_by_node    │
    │                       node_key 精确匹配                                  │
    │                                                                         │
    │  检索:    file_paths  →  tm_solve / tm_search  →  list_by_nodes         │
    │                       node_key IN (...)  →  boost +0.15                  │
    └────────────────────────────────────────────────────────────────────────┘
```

---

*文档结束。相关设计见 `docs/design-docs/`、`docs/exec-plans/completed/code-arch-viz-gitnexus/`。*
