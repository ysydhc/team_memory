# GitNexus 分析样本与 Provider 接口数据对比

> 基于当前仓库已生成的 GitNexus 索引（`.gitnexus/` + CLI 输出），对比「GitNexus 实际提供的数据」与「我们 Provider 契约」的符合度，并给出映射与缺口说明。

---

## 一、本地样本概况

- **索引路径**：`.gitnexus/`（含 `meta.json`、Kuzu 图库文件 `kuzu`）。
- **meta.json**（当前内容）：
  ```json
  {
    "repoPath": "/Users/yeshouyou/Work/agent/team_doc",
    "lastCommit": "a7124f6a1d1afdfe3c1310ee6f6ea1b66170c253",
    "indexedAt": "2026-03-01T15:15:24.694Z",
    "stats": {
      "files": 172,
      "nodes": 2066,
      "edges": 6085,
      "communities": 199,
      "processes": 172
    }
  }
  ```
- **CLI**：`npx gitnexus list` 显示 repo 名为 **team_doc**；`npx gitnexus status` 可判断 **stale**（当前 commit 与 indexed commit 不一致时）。

---

## 二、GitNexus 提供的接口与数据形状

### 2.1 概览 / Context（对应我们的 `ArchitectureContext`）

| 来源 | 数据 |
|------|------|
| **meta.json** | `stats.files`, `stats.nodes`, `stats.edges`, `stats.communities`, `stats.processes`, `lastCommit`, `indexedAt`, `repoPath` |
| **MCP 资源** | `gitnexus://repo/team_doc/context` — 官方说明为「Codebase stats, staleness check, and available tools」 |
| **stale** | 需自行计算：当前 `git rev-parse HEAD` 与 `meta.json.lastCommit` 是否一致 |

**结论**：概览数据**符合预期**。映射关系：

- `repo_name` ← 从 `repoPath` 取最后一段（如 `team_doc`）或 MCP context 若返回 name。
- `symbols` ← `stats.nodes`（2066）。
- `relationships` ← `stats.edges`（6085）。
- `processes` ← `stats.processes`（172）。
- `stale` ← 比较当前 HEAD 与 `lastCommit`。
- `available` ← 索引存在且可读即为 true。

---

### 2.2 集群 / Clusters（对应我们的 `ClusterSummary[]` 与 `ClusterMembers`）

| 来源 | 数据 |
|------|------|
| **MCP 资源** | `gitnexus://repo/team_doc/clusters` — 「All functional areas with cohesion scores」；`gitnexus://repo/team_doc/cluster/{name}` — 「Cluster members and details」 |
| **meta.json** | 仅有 `stats.communities: 199`，无具体名单与 cohesion |

**结论**：集群**符合预期**，但具体字段需以 MCP 资源实际返回为准。

- **ClusterSummary**：`name` 必有；`cohesion`、`member_count` 若 MCP 返回则直接映射。
- **ClusterMembers**：`name` + `members: [{ id, path?, kind? }]`，用 MCP `cluster/{name}` 返回的成员列表做映射（id 用 symbol uid，path 用 filePath）。

---

### 2.3 图 / Graph（对应我们的 `ArchitectureGraph`）

| 来源 | 数据 |
|------|------|
| **CLI** | `query` 返回 **processes** + **process_symbols**（按执行流分组），非「节点+边」平面图。 |
| **Cypher** | 可查节点与边，例如 `MATCH (a:Function) RETURN a.name LIMIT n`；Kuzu 语法与 Neo4j 略有差异。 |
| **MCP** | 无直接「整图 nodes+edges」资源；需通过 **query** 工具或 **cypher** 工具拼出。 |

**结论**：图数据**需做转换**，接口语义一致但需一层适配。

- **nodes**：可从 Cypher 查 `Function`/`Class`/`File`/`Community` 等节点，或从 `process_symbols` 去重得到；每项映射为 `{ id, label?, kind?, path?, meta? }`（id = symbol uid，path = filePath）。
- **edges**：需通过 Cypher 查关系（如 `CALLS`、`IMPORTS`），或从 **context** 的 incoming/outgoing 拼边；映射为 `{ source, target, type?, meta? }`。
- **focus_node_id**：请求带 `file_path` 时，用 filePath 解析出对应节点 id（如某 Function 的 uid），填到响应。

Bridge 或 GitNexusProvider 可封装：  
- 用 **cypher** 执行「节点列表 + 边列表」查询，或  
- 用 **query** 得到 process_symbols + 用 **context** 得到部分边，再聚合成统一 graph。

---

### 2.4 影响面 / Impact（对应我们的 `ImpactResult`）

| 来源 | 数据 |
|------|------|
| **CLI** | `gitnexus impact <symbol> -r team_doc --depth n [--direction upstream|downstream]` 返回 JSON。 |

**实际 CLI 返回示例（upstream）**：

```json
{
  "target": { "id": "Function:...:get_current_user", "name": "get_current_user", "filePath": "src/team_memory/web/app.py" },
  "direction": "upstream",
  "impactedCount": 1,
  "risk": "LOW",
  "summary": { "direct": 1, "processes_affected": 0, "modules_affected": 1 },
  "byDepth": {
    "1": [
      { "depth": 1, "id": "Function:...:_check_role", "name": "_check_role", "filePath": "src/...", "relationType": "CALLS", "confidence": 0.9 }
    ]
  }
}
```

**结论**：影响面**符合预期**，需做字段映射与两次调用（upstream + downstream）。

- **ImpactResult.upstream**：调用 `impact(..., direction=upstream)`，将 `byDepth["1"]`、`byDepth["2"]`… 合并为数组，每项映射为 `{ id, path: filePath, depth }`。
- **ImpactResult.downstream**：同上，`direction=downstream`。
- **注意**：GitNexus 的 **impact 目标为符号**（如 `Function:path:name`），不是纯文件路径。我们 API 的 `path` 若为文件路径，Bridge/Provider 需：  
  - 要么解析为该文件下某代表符号（如第一个 Function），  
  - 要么按「该文件下所有符号」分别调 impact 再去重合并（推荐用于「按文件看影响面」）。

---

### 2.5 单符号上下文 / Context（对应「代码实时结合」与可选 code-context）

| 来源 | 数据 |
|------|------|
| **CLI** | `gitnexus context <name> -r team_doc` 返回 JSON。 |

**实际返回示例**：

```json
{
  "status": "found",
  "symbol": { "uid": "Function:src/.../app.py:get_current_user", "name": "get_current_user", "filePath": "src/team_memory/web/app.py", "startLine": 445, "endLine": 445 },
  "incoming": { "calls": [ { "uid": "Function:...:_check_role", "name": "_check_role", "filePath": "..." } ] },
  "outgoing": { "calls": [ { "uid": "Function:...:_decode_api_key_cookie", ... }, { "uid": "Function:...:authenticate", ... } ] },
  "processes": []
}
```

**结论**：与「架构图 + 代码联动」的预留一致。

- **focus_node_id**：可直接用 `symbol.uid`。
- **相关节点**：incoming.calls / outgoing.calls 的 uid 可作 `highlight_node_ids` 或边数据来源。

---

## 三、与 Provider 契约的逐项符合度

| 我们的接口 | GitNexus 数据源 | 符合度 | 说明 |
|------------|-----------------|--------|------|
| **ArchitectureContext** | meta.json + stale 计算 或 MCP `repo/.../context` | ✅ 符合 | 字段一一对应，stale 需自算或 MCP 已提供 |
| **ClusterSummary[]** | MCP `repo/.../clusters` | ✅ 符合 | 以 MCP 实际返回为准，name/cohesion/member_count 可映射 |
| **ClusterMembers** | MCP `repo/.../cluster/{name}` | ✅ 符合 | members 从返回列表映射 id/path/kind |
| **ArchitectureGraph** | Cypher 或 query+context 聚合 | ⚠️ 需转换 | 无现成「整图 JSON」，需 Bridge 用 cypher/query 生成 nodes+edges |
| **ImpactResult** | CLI/MCP `impact`，upstream + downstream 各一次 | ✅ 符合 | byDepth → upstream/downstream 数组；目标为符号时需从 file_path 解析符号 |
| **ExperienceRef[]** | TM 自管 | — | 不依赖 GitNexus |

---

## 四、Bridge / Provider 实现要点

1. **context（概览）**：读 `.gitnexus/meta.json` 或调 MCP 资源 `gitnexus://repo/team_doc/context`，映射为 `ArchitectureContext`；stale 用 `git rev-parse HEAD` 与 `lastCommit` 比较。
2. **clusters**：调 MCP `clusters` 与 `cluster/{name}`，直接映射为 `ClusterSummary[]` 与 `ClusterMembers`。
3. **graph**：用 **cypher** 查询节点与边（或 query 的 process_symbols + context 的 incoming/outgoing），聚合成 `ArchitectureGraph`；请求带 `file_path` 时解析出对应 symbol uid 填 `focus_node_id`。
4. **impact**：  
   - 入参为 **符号**时：调 `impact(symbol, direction=upstream)` 与 `impact(symbol, direction=downstream)`，合并 byDepth 为 upstream/downstream。  
   - 入参为 **文件路径**时：先解析该路径下符号列表（Cypher 或 list 接口），再对每个符号做 impact 并去重合并（或仅取「该文件为 target」的 impact 结果）。
5. **experiences**：仅查 TM 库，按 node_key（filePath 或 symbol uid）过滤，不调 GitNexus。

---

## 五、小结

- GitNexus 分析样本（meta.json + CLI 的 query/context/impact）**与我们的 Provider 契约在语义和字段上基本一致**，可直接作为 Bridge 数据源。
- **需适配的部分**：  
  - **Graph**：由 Cypher/query+context 聚合为 nodes+edges；  
  - **Impact**：由两次调用（upstream/downstream）合并为 `ImpactResult`，且支持「按文件路径」时先解析为符号再调 impact。  
- **节点标识**：统一使用 GitNexus 的 **uid**（如 `Function:src/.../app.py:get_current_user`）作为 node_key，与 experience_architecture_binding 和 focus_node_id 对齐，便于架构图与代码实时结合。
