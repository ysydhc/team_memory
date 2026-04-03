# Plan：个人记忆 / 用户画像（对齐 Supermemory 思路）

> **状态**：✅ 已完成（MVP A～D；Phase E 见 p2-backlog）  
> **执行跟踪**：[execute.md](../execute.md) · **任务清单**：[tasks.md](../tasks.md)  
> **参考（外部产品）**：[Supermemory 文档](https://supermemory.ai/docs)（User Profiles 等能力以官方为准；项目内不再维护独立对比稿）  
> **MCP 默认入口**：[mcp-lite-default.md](../../../../design-docs/ops/mcp-lite-default.md)（`make mcp` = `team_memory.server`）  
> **日期**：2026-04-01  
> **工程现状（2026-04-03 校对）**：`server_lite.py` 已删除；MCP 仅 **`src/team_memory/server.py`** 中的 **`memory_*`**（现为 **5** 工具，含 **`memory_get_archive`**）；单测见 **`tests/test_server.py`**。下文若仍写 `server_lite` / `test_server_lite` / 「四处工具」，视为历史表述。

---

## 导读：读 Plan 的顺序与两套交付

本 Plan 分 **两次可独立发版** 的交付包，避免把「画像形状」和「到期失效」绑死在同一认知里：

| 交付包 | 阶段 | **交付物（一句话）** |
|--------|------|----------------------|
| **MVP** | **Phase A～D** | `personal_memories` 增加 **`profile_kind`**（`static`/`dynamic`）；对外 **Lite MCP** 返回 Supermemory 形 **`profile: { static: string[], dynamic: string[] }`**；**不新增 MCP 工具名**——仅扩 **`memory_context`** 的 JSON；**可选** 扩 **`memory_recall`** 的可选参数（如 `include_user_profile`）。 |
| **续包** | **Phase E** | 增加 **`valid_until`**：到期行 **不进** Agent 可见 `profile`；抽取可选填到期时间；可选后台物理清理。 |

**MCP 范围（已决）**：产品对外 **仅** `server_lite.py` 注册的 **`memory_save`、`memory_recall`、`memory_context`、`memory_feedback`** 四处工具；**不再**维护或文档承诺完整 **`tm_*`** MCP 面。原 **`tm_suggest` / `tm_search` / `tm_learn`** 等不作为 Cursor/Agent 接入路径；与服务层的共享代码（若有）属实现细节，**不**写入本 Plan 的 MCP 契约。

**术语**：下文 **「画像 / profile」** 指注入 Agent 的 **`static`+`dynamic` 字符串数组**；**「个人记忆行」** 指表 `personal_memories` 中的一条记录（HTTP 可见字段多于 MCP）。

**推荐阅读顺序**：§0 已决 → §4 路线图（执行入口）→ §3 技术细节 → §6 验收 → §9～10（产品与治理）。

---

## 0. 已决事项（2026-03-30）

| 议题 | 决议 |
|------|------|
| **对外结构** | 与 Supermemory 一致：`profile.static`、`profile.dynamic` 为 **字符串数组**。 |
| **MCP 接口** | **不新增**工具名；**仅**在 Lite 工具上扩展 **JSON 返回** 或 **可选参数**；MCP **仅限**上述四个 `memory_*`。 |
| **产品形态** | **全面 Lite MCP**；完整 **`tm_*`** MCP **不提供**、不作为验收对象。 |
| **时间敏感事实** | **必须做**：在 **Phase E** 交付 `valid_until` 及读路径过滤（详见 §3.6）；与 **MVP 可分开发布**。 |

---

## 1. 背景与目标

### 1.1 Supermemory「User Profile」参照

```typescript
const { profile } = await client.profile({ containerTag: "user_123" });
// profile.static  → ["Senior engineer at Acme", "Prefers dark mode", "Uses Vim"]
// profile.dynamic → ["Working on auth migration", "Debugging rate limits"]
```

### 1.2 本 Plan 在 Team Memory 中要达成什么

前提：**自托管**，存储为 **`personal_memories` + pgvector**，对外 MCP **仅 Lite**。

- **MVP（Phase A～D）**  
  1. **`memory_context`**（及 execute 决定是否扩展的 **`memory_recall`**）使 MCP 读路径能返回 **`profile: { static: string[], dynamic: string[] }`**（匿名固定空数组，见 §7）。  
  2. 写入侧：`parse_personal_memory` 与落库支持 **`profile_kind` ∈ {`static`,`dynamic`}**，并能组装成上述两数组。  
  3. **读路径（不增加新工具）**：**主路径**——任务开始调用 **`memory_context`**，响应中带 **`profile`** 与既有「相关经验/知识」结构（与现 Lite 行为对齐，不追求与已移除的 `tm_suggest` 1:1）。**可选**——在 **`memory_recall`** 上增加 **`include_user_profile: bool = False`**，便于「只 recall、不先 context」的集成也能捎带画像（**优先级见 §3.4 / §7**）。  
  4. 日志：`personal_memory:` 可统计 static/dynamic 写入条数；与 [troubleshooting §12](../../../../design-docs/ops/troubleshooting.md) 可对照排查。  

- **Phase E（续包）**  
  - 支持 **`valid_until`**：过期记录 **不参与** `profile` / `pull` / 聚合读；抽取可填到期；细节见 **§3.6**。  

### 1.3 Non-Goals（本 Plan 不做）

- 官方连接器（Gmail/Drive/Notion 等）。  
- **全量矛盾推理**（多事实自动裁决、复杂时间线推理）。**说明**：到期隐藏 **不等于** 自动解决矛盾。  
- **新增 MCP 工具**（含 `tm_profile`、第五个 `memory_*` 等）。  
- **完整 `tm_*` MCP**：恢复或并行维护对外 **`tm_suggest` / `tm_search` / `tm_learn`** 等工具集（与「全面 Lite」冲突）。  
- SaaS 计费。  

### 1.4 用户、场景与与团队经验的边界

- **Persona**：Coding Agent（**`memory_context` / `memory_recall` / `memory_save`**）；Web 查看与纠错（Phase C3）。  
- **场景**：冷启动空 `profile` 行为明确；长跑项目中 `dynamic` 偏近期、`static` 偏稳定偏好。  
- **边界**：**profile** = 个人偏好与语境；**Experience** = 团队可复用知识。抽取须避免把团队公共条目误写入 `personal_memories`。  

### 1.5 隐私与多租户（原则）

- 数据按 **登录用户** 隔离；若存在 `project` 等字段，execute 须 **二选一定稿**：跨 project 是否共享同一 profile。  
- 默认不把「管理员批量查看他人画像全文」作为 MVP 目标；若现有 RBAC 已允许，retro 评估是否加固。  

---

## 2. 现状 → 目标（按交付包分列）

| 维度 | 当前 | **MVP 目标（A～D）** | **Phase E 增量** |
|------|------|----------------------|------------------|
| 结构 | `content` + `scope` | **+ `profile_kind`**；对外 **`profile.static` / `profile.dynamic`** | **+ `valid_until`（可 NULL）** |
| 抽取 | 单 prompt、无 kind | 每条 **static/dynamic** + 落库 | 每条可选 **到期时间** → `valid_until` |
| 读 MCP（Lite） | `memory_context` 无统一 Supermemory 形 `profile` | **`memory_context` 标准带 `profile`**；**可选** **`memory_recall`** 捎带 `profile` | 组装 **前过滤** `valid_until` |
| 失效 | 仅语义相似 **覆盖** 同行 | 不变 | **时间到期** 即对 Agent **不可见**（可选物理删） |

---

## 3. 技术方案概要

### 3.0 MVP 与 Phase E 的边界（避免实现时混线）

- **仅有 MVP 迁移时**：表上 **无** `valid_until`；所有读写与 **当前生产行为** 一致（语义 upsert 仍可覆盖相近行）。  
- **Phase E 迁移后**：增加列并开始过滤；`parse_personal_memory` 中 `valid_until` 字段才写入 DB。  

### 3.1 数据模型

**MVP**

- `personal_memories.profile_kind`：`static` | `dynamic`；存量行用 **execute 写明** 的映射规则（如原 `scope` → kind）做回填。  
- 组装 `profile`：按用户取行 → 按 `profile_kind` 分到 `static` / `dynamic` 字符串列表（去空、去重、**排序与截断**见下）。  

**Phase E 起**

- `personal_memories.valid_until`：`TIMESTAMPTZ NULL`，`NULL` = **永不过期**。  
- 任何参与 **`profile` / pull / build_profile_for_user** 的查询：**`valid_until IS NULL OR valid_until > now() UTC`**。  
- **Web**：列表默认 **不展示** 过期行；可支持 `include_expired=true`（调试用）。**MCP `profile`** 永远不包含过期文案。  

**排序与截断（共用）**

- 侧内排序：默认 **`updated_at DESC`**（execute 可改为 `created_at` 但须全链路一致）。  
- MCP：`static` / `dynamic` **各最多 N 条**（如 20，与 §5 一致）。  
- Web/HTTP：列表可 **全量或分页**，不以 MCP 截断为准。  

### 3.2 抽取（LLM）

- **MVP**：`parse_personal_memory` 每条含 **`profile_kind`**；非法 → `static`。写入持久化 **`profile_kind`**。  
- **Phase E**：每条可选 **`valid_until`**（ISO-8601 UTC 或相对 `+7d` 等，由 prompt + 解析器约定）；解析失败 → `NULL`。Prompt：**仅短期事实**（考试、发布窗、on-call）填到期；**长期偏好** 不填。  
- **产品完整性**：**Lite `memory_save(content=…)`**（及与其绑定的服务端抽取路径）须可按 [troubleshooting §12](../../../../design-docs/ops/troubleshooting.md) 跑通（LLM、超时、表存在）。实现上若与内部非 MCP 入口共用函数，**验收仍以 Lite 为准**。  

### 3.3 服务层

- **`build_profile_for_user` → `{ static: list[str], dynamic: list[str] }`**：内部可继续用向量/`scope` 辅助选取 context 行（与现 `pull` 逻辑对齐），再按 kind 折叠为字符串。  
- **语义 upsert（MVP）**：与现仓库一致方向——**同用户**下高相似度 **更新** 同一行；Plan 要求 **同 `profile_kind` 内** 合并策略在 execute/B3 写清阈值与单测。  
- **Phase E**：`find_most_similar` / `upsert_by_semantic` **优先未过期行**；若只命中 **已过期** 行 → **原地更新**（复活）；无命中 → `create`。  

### 3.4 MCP（仅 Lite；无新工具）

| 入口 | 变更 |
|------|------|
| **`memory_context`** | 响应**必须**含标准形 **`profile: { static: string[], dynamic: string[] }`**；在**同一次调用**中继续提供与 Lite 一致的「与当前文件/任务相关的经验或知识摘要」（具体键名以现 API 为准）。**不**承诺字段级与已移除的 `tm_suggest` 一致。旧字段若保留须定 **废弃日**（§6）。 |
| **`memory_recall`（可选扩展）** | 增加 **`include_user_profile: bool = False`**（默认值保持兼容）。为 `true` 时在返回 JSON **附加**与 `build_profile_for_user` 一致的 `profile`。**策略**：默认 **P1**（主路径靠 `memory_context`）；若观测到集成方 **从不调 context、只 recall**，在 execute 升为 **P0** 并补测。 |
| **`memory_save` / `memory_feedback`** | **不改名**；`memory_save` 触发个人记忆抽取的行为与 troubleshooting 对齐。 |
| **HTTP** | list/pull 支持按 **`profile_kind`** 过滤；字段与 MCP 字符串数组的对应写 API 文档。 |

**文档债务**：根 `README.md` / `docs/design-docs/ops/`、Cursor 规则、Lite Instructions **只描述五个 `memory_*`**；删除或替换仍指向 **`tm_preflight` / `tm_suggest` / `tm_search`** 的幽灵表述。  

### 3.5 观测与文档同步

- 日志：`personal_memory: saved ...`（可含 static/dynamic 计数；E 后可 debug 统计含 `valid_until` 的写入）。  
- 更新：troubleshooting §12、对外产品对照与画像说明（可链到 Supermemory 官方文档；E 后「自动遗忘」叙事与实现一致）、**README / MCP 运维文档中的 Lite 工具与返回体**（不登记新工具名）。  

### 3.6 Phase E：`valid_until` 详规（摘要表）

| 条目 | 约定 |
|------|------|
| 语义 | 现时刻 **`valid_until ≤ now()`** 的行 **不进入** Agent 可见的 `profile` / `pull` / `build_profile_for_user`。 |
| 存储 | `TIMESTAMPTZ`，比较统一 **UTC**。 |
| 清理（可选） | 定时 **物理删除**「过期早于 now−retention」的行；**可配置**；**删库不是** Agent 侧「看不见」的前提（以查询过滤为准）。 |
| MCP | `profile` **不返回** `valid_until`；调试走 Web/HTTP。 |
| 约定 | **`dynamic` 更常带短期到期**；`static` 多为 `NULL`（非强制）。 |

---

## 4. 阶段与任务拆分（执行清单）

### 4.1 路线图总览

```
Phase A（模型） → Phase B（抽取/写入） → Phase C（Lite MCP + Web） → Phase D（规则/文档）
       ↓ 全部完成后为「MVP 可发布」
Phase E（valid_until + 过滤 + 可选清理）→ 「画像 + 到期失效」完整对齐 Supermemory 该维度
```

| Phase | 周期（估） | 依赖 | 完成标志 |
|-------|------------|------|----------|
| **A** | ~1 周 | — | `profile_kind` 迁移 + ORM + `build_profile_for_user` 骨架 + 单测 |
| **B** | ~1 周 | A | Prompt/解析/落库 kind + upsert 文档与测试 |
| **C** | ~1～1.5 周 | B | **`memory_context` + `profile`**；**可选** `memory_recall` 参数；C3 Web |
| **D** | ~0.5 周 | C | **`server_lite` Instructions** + rules + **README / ops MCP 文档（Lite）**；无幽灵 `tm_*` / `tm_preflight` |
| **MVP** | **~3～4 周** | A～D | §6.1～6.2 + §9.3 一轮 |
| **E** | ~1～1.5 周 | MVP 或并行最后一周（由团队定） | §6.3 全满足 |

### 4.2 Phase A — 数据与模型

| ID | 任务 | 验收 |
|----|------|--------|
| A1 | Alembic：`profile_kind` + 默认/回填 | 迁移可验证 |
| A2 | ORM / `to_dict` / Repository | 测试通过 |
| A3 | `build_profile_for_user`（kind → 两列表） | 单元测试 |

### 4.3 Phase B — 抽取与写入

| ID | 任务 | 验收 |
|----|------|--------|
| B1 | `parse_personal_memory`：kind | Mock LLM |
| B2 | 服务端个人记忆写入路径写 kind + 日志（与 **`memory_save`** 链一致） | 日志 / 手工 |
| B3 | **`profile_kind` 维度上的** upsert 策略 + 文档 | 与 §3.3 一致、有测 |

### 4.4 Phase C — Lite MCP 与 Web

| ID | 任务 | 验收 |
|----|------|--------|
| C1 | **`memory_context`**：响应含 **`profile.static` / `profile.dynamic`**；匿名空数组 | **`test_server_lite.py`** |
| C2 | （可选）**`memory_recall`**：`include_user_profile`（**P0/P1 见 §7**） | **`test_server_lite.py`** + 文档 |
| C3 | Web：static / dynamic 分组展示 | `make lint-js` |

### 4.5 Phase D — 产品与规则

| ID | 任务 | 验收 |
|----|------|--------|
| D1 | **`server_lite` Instructions** + `.cursor/rules`：任务开始 **`memory_context`**；消费 **`profile`**；§1.4 边界；**不提 `tm_*` MCP** | 评审通过 |
| D2 | execute：**P2 backlog**（§10）单独列表，不阻塞 MVP | backlog 有链接 |

### 4.6 Phase E — 时间敏感失效

| ID | 任务 | 验收 |
|----|------|--------|
| E1 | Alembic：`valid_until`；ORM；Web 可选展示 | 迁移可测 |
| E2 | 解析 + 写入 `valid_until` | Mock + 集成 |
| E3 | 所有读路径 **统一过滤**；upsert **优先未过期 / 复活过期行** | 单测 + 回归 |
| E4 | （可选）定时物理清理 + 配置项 | 文档 + 日志 |
| E5 | 对比文档 / troubleshooting：自动遗忘 **部分支持** | 已更新 |

---

## 5. 风险与依赖

| 风险 | 缓解 |
|------|------|
| MCP 仅字符串数组 | Web/HTTP 保留 id；调试以 DB/API 为准 |
| `profile` 体积 | 每侧条数上限（如 ≤20） |
| 多次 **`memory_context`** | 规则写清 **任务开始一次**（或明确豁免场景） |
| 旧客户端 | `profile_version` 或双字段 + **废弃日** |
| LLM 填错 `valid_until` | 解析保守 + Web 删除 + §9.4 纠错成本 |
| 时区 | **UTC** 存储与比较 |
| **Lite 与旧文档分叉** | 全局检索替换 training 材料中对 **`tm_suggest`** 的依赖描述 |

---

## 6. 验收总清单（Definition of Done）

### 6.1 MVP 工程

- [ ] **A～B**：`profile_kind` + `build_profile_for_user` + 抽取与 upsert 有测  
- [ ] **C1～D**：**`memory_context`** 返回 **`profile: { static, dynamic }`**；**（若做 C2）** `memory_recall` 行为与文档一致；**无新 MCP 工具名**  
- [ ] 写入链路至少 **1 条** 自动或半自动用例（**`memory_save` / 抽取**，mock LLM 或本机 Ollama，execute 注明）  
- [ ] **`make verify`** 以仓库现状为准；**Lite 相关测试**（如 `test_server_lite.py`）覆盖 `profile`  
- [ ] **Instructions / 文档** 中 **无** 要求用户调用 **`tm_*` MCP** 或 **`tm_preflight`**  
- [ ] 无硬编码密钥；敏感不全量打日志  
- [ ] 若有旧字段并存 → execute 记 **废弃日期**  

### 6.2 MVP 产品演示

- [ ] 同用户两次间隔会话：**`dynamic`** 能随语境变、`static` 不无故乱跳（脚本 + DB/日志核对）  
- [ ] **§9.3 最小评价协议** 完成 **≥1 轮** 并附记录（retro 或 execute）  

### 6.3 Phase E

- [ ] 过期测试行 **不出现在** `profile` / pull  
- [ ] 未过期：语义 upsert 行为正确；仅命中过期行：**可复活**  
- [ ] 若实现 E4：**不得** 删除 `valid_until IS NULL` 的行  

---

## 7. 落地时须拍板的常量（写入 execute）

1. **`profile` 每侧条数上限**（默认建议 20）。  
2. **匿名用户**：统一返回 **`profile: { static: [], dynamic: [] }`**（推荐，解析简单）。  
3. **`memory_recall(include_user_profile)`**：按 §3.4 定 **P0 或 P1**，并更新 **README / `docs/design-docs/ops/mcp-server.md`（Lite）**。  
4. **跨 project 是否共享 profile**（§1.5）。  
5. Phase E 的 **物理清理 retention**（若做 E4）。  

---

## 8. 工时粗估

| 项 | 周期 |
|----|------|
| Phase A + B | ~2 周 |
| Phase C + D | ~1～1.5 周（Lite 单面可能略减 C 工期） |
| **MVP 合计** | **~3～4 周** |
| Phase E | **~1～1.5 周** |
| **MVP + E** | **~4～5.5 周** |

P2（§10 中关闭自动抽取、导出、RBAC 加固等）：**+0.5～1.5 周**（与 Phase E 并行时应分拆核算，避免重复计入）。  

---

## 9. 产品成功与有效性评审

### 9.1 价值主张（何谓「对项目有帮助」）

满足 **至少一条** 并可举例：**减少重复语境**；**`profile` 不替代** 团队经验检索（**`memory_recall` 等**）；**可纠错 / 关画像不劣化基线**。  

### 9.2 指标分层

- **L0**：非空率、错误率（只说明链路活）。  
- **L1**：static/dynamic 是否合理、团队事实误入行数（抽查）。  
- **L2**：固定剧本下 **有/无 profile** 的 **重问次数、跑偏次数**（人工，**§9.3**）。  
- **L3**：真实 PR/issue 返工（可选，不强求 KPI）。  

### 9.3 最小评价协议（MVP 合并后 2 周内）

1. 选本仓库典型开发任务。  
2. **A**：不提供 profile（忽略返回或清空）；记 **重复语境追问次数**、**明显违背偏好/项目次数**。  
3. **B**：提供相关 profile；同任务再记。  
4. 结论：**B 不劣于 A** 且至少一项改进，或 **无差异** 则记原因（规则未注入、画像空等）。  

**控制变量**：同一仓库、同一规则文件、尽量同一模型；只变是否向 prompt **注入 profile**。  

### 9.4 过程指标（运营）

| 指标 | 说明 |
|------|------|
| 画像可用率 | 认证用户 `profile` 非空比例 |
| dynamic 新鲜度 | `updated_at` 与活跃窗口 |
| 工具错误率 | **`memory_context`** / **`memory_recall`**（可抽样日志） |
| 纠错成本 | HTTP 删除条数 |
| 到期命中（E 后） | 抽样确认过期不出现在 MCP `profile` |

### 9.5 复盘

- **T+2 周**：§9.3 ≥1 轮 + L0 扫一眼。  
- **T+1 月**：可用率持续极低且无 §9.3 正例 → 先查规则与写入，再议收缩范围。  
- **对外宣称「画像就绪」**：须 **§6.1～6.2** 与 **§9.3 一轮** 完成（§9.4 可备注「无遥测」）。  

---

## 10. 用户可控与治理（MVP / P2）

| 能力 | MVP | P2 |
|------|-----|-----|
| 查看 | Web 分组；HTTP 列表 | — |
| 删错条 | HTTP DELETE（若无则 C3 补） | — |
| 编辑 / 撤销自动写 | 删 + 重抽 | 行级编辑、历史 |
| 关自动抽取 | 随 **`memory_save`** | 按用户/项目开关 |
| 导出 | — | GDPR 等 |
| 到期 | Phase E 自动对 Agent 不可见；Web 可选看过期 | 手改 `valid_until` |

对外一句：**错误画像可通过 Web/HTTP 删除或管理员协助**（细节见根 `README.md` / 安全页）。  

---

## 11. 附录：修订纪要（精简）

- **产品评审**：补成功指标、用户信任、profile 与 Experience 边界 → **§9、§10**。  
- **有效性**：补 L2 协议与归因 → **§9.2～9.3**。  
- **时间敏感**：`valid_until` + Phase E + 与 MVP 分拆 → **导读、§2、§3、§4.6、§6.3**。  
- **通读 2026-03-30**：导读、§3.0、§4.1、§2 分列表格、§6～7。  
- **Lite MCP 单轨 2026-03-30**：MCP **仅** 四处 `memory_*`；删除 Plan 内 **`tm_suggest` / `tm_search` / `tm_learn` MCP** 验收与 Phase C 双轨描述；**`memory_recall` 可选参数** 承接原「只搜也要 profile」场景；§6 **`test_server_lite`** 为主。  
