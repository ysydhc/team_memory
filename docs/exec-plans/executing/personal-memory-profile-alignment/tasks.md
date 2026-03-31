# 任务清单：个人记忆 / 用户画像（Supermemory 对齐）

> **来源 Plan**： [2026-04-01-personal-memory-profile-supermemory-alignment.md](../../../plans/2026-04-01-personal-memory-profile-supermemory-alignment.md)  
> **MCP 范围**：对外 **仅 Lite**（`memory_save` / `memory_recall` / `memory_context` / `memory_feedback`）；**不包含** 完整 `tm_*` 工具验收。  
> **任务拆分口径**：对齐「产品经理」关注点——**是否解决用户痛点、与需求对齐、MVP 可验收、成功标准可检查**；每条任务区分 **工程验收** 与 **产品/对齐验收**（可合并为勾选表）。  
> **执行顺序**：按 **依赖列** 串行；同 Phase 内可按 **ID 序号** 执行。  
> **MVP 工程状态（2026-03-30）**：A～D 与 T-W01 已在代码库落地；验收与勾选以 [execute.md](./execute.md) **进度速览**为准。**T-V01** 仍待人测；**Phase E** 见 [p2-backlog.md](./p2-backlog.md) 与 tasks 末段。

---

## 任务字段说明

| 字段 | 含义 |
|------|------|
| **业务价值** | 对用户/Agent/团队的直接作用（为何做这一项） |
| **工程验收** | 代码、测试、迁移、构建 |
| **产品验收** | 与 Plan §1.2 / §9 对齐的可观察结果（含文档与规则） |
| **成功信号（L0/L1）** | 可选；便于 retro 填数 |

---

## 仓库级迁移 — 默认 Lite MCP（可与 Phase A 并行）

> **决策与细节**：[mcp-lite-default.md](../../../design-docs/ops/mcp-lite-default.md)（含 **PyPI 发版清单**、**`test_server.py` / 完整 MCP 移除时间线**）。

### T-W01 — CHANGELOG 与对外公告闭环

| 项 | 内容 |
|----|------|
| **阶段** | 仓库级（先于或与 Phase A 并行） |
| **优先级** | P0（与默认 MCP 切换同步对外可见） |
| **依赖** | 无（工程侧 `make mcp` / `pyproject` 已落地） |
| **业务价值** | 装包用户与集成方从 **CHANGELOG + 文档** 得知 **`team-memory` → Lite**；降低 silent break。 |
| **范围** | 根目录 **[CHANGELOG.md](../../../../CHANGELOG.md)** 维护 `[Unreleased]` / 版本条目；下次 **PyPI 发布** 按 **mcp-lite-default § PyPI/发版公告** 核对清单（版本号与 CHANGELOG 对齐、Release 摘要含 Breaking）。 |
| **工程验收** | 仓库内存在可追溯的 CHANGELOG；mcp-lite-default 已与本文链互通。 |
| **产品验收** | 新用户按 getting-started / README 配置 **server_lite**；旧 **`tm_*`** 用户能查到迁移路径（`team-memory-full` / `make mcp-full`）。 |

---

## Phase A — 数据与模型

### T-A01 — `profile_kind` 数据库迁移与存量回填

| 项 | 内容 |
|----|------|
| **阶段** | Phase A |
| **优先级** | P0（阻塞后续） |
| **依赖** | 无 |
| **业务价值** | 个人事实可分区为 **长期 static / 近期 dynamic**，对齐 Supermemory 心智，避免混在一列里无法组装 `profile`。 |
| **范围** | Alembic：`personal_memories.profile_kind`；枚举约束或 CHECK；存量行 **回填规则** 写在迁移注释 + `execute.md`（如 `scope=generic`→`static`，`context`→`dynamic`，或统一默认 `static` 后人工修正策略二选一）。 |
| **工程验收** | `alembic upgrade` / `downgrade` 在干净库与「已有 personal_memories」库上可验证；CI 或本地脚本记录版本号。 |
| **产品验收** | 回填策略经 **一人评审** 并写入 execute，避免「静默错误归类」。 |
| **成功信号** | 存量用户升级后 **无** 因 NOT NULL 导致写入失败；抽样 5 行 kind 符合预期。 |

---

### T-A02 — ORM、`to_dict`、Repository 读写 `profile_kind`

| 项 | 内容 |
|----|------|
| **阶段** | Phase A |
| **优先级** | P0 |
| **依赖** | T-A01 |
| **业务价值** | Web/服务层能稳定读写 kind，用户可在管理界面区分「偏好」与「当前语境」。 |
| **范围** | `PersonalMemory` 模型；`to_dict` 含 `profile_kind`；`PersonalMemoryRepository` create/update/list 不传漏。 |
| **工程验收** | 单元或仓储测试：创建/更新/列出带 kind；`make verify` 相关子集通过。 |
| **产品验收** | HTTP API 响应（若已有 list/detail）中 **客户端可见** `profile_kind`（与 Plan §3.4 HTTP 行一致）。 |
| **成功信号** | L0：相关测试全绿。 |

---

### T-A03 — `build_profile_for_user` 与截断策略

| 项 | 内容 |
|----|------|
| **阶段** | Phase A |
| **优先级** | P0 |
| **依赖** | T-A02 |
| **业务价值** | 统一「从多行个人记忆 → **两串字符串**」逻辑，供 MCP/Web 复用，避免各端各写一套分叉。 |
| **范围** | `PersonalMemoryService.build_profile_for_user`（或等价命名）；按 `profile_kind` 分桶；`updated_at DESC`；每侧 **最多 N 条**（N 默认 20，与 Plan §7 一致，可配置项）；去空；去重策略（execute 二选一：严格去重首条 / 保留顺序）。 |
| **工程验收** | 单测：多行混 kind → 输出 `static`/`dynamic` 列表正确顺序与截断；匿名/无行返回空列表。 |
| **产品验收** | **对齐 Plan §1.2**：输出类型为 **`list[str]`**，不包含 id（MCP 侧）。 |
| **成功信号** | L1：手工 1 例 DB 数据与函数输出对照一致。 |

---

## Phase B — 抽取与写入

### T-B01 — `parse_personal_memory`：prompt 与结构化解析

| 项 | 内容 |
|----|------|
| **阶段** | Phase B |
| **优先级** | P0 |
| **依赖** | T-A02 |
| **业务价值** | Agent 从对话中**自动**得到带种类的个人条目，减少手工维护画像。 |
| **范围** | Prompt 要求每条 JSON/schema 含 `profile_kind`；非法值回落 `static`；单测使用 **Mock LLM** 固定返回。 |
| **工程验收** | `pytest` 覆盖：合法 static/dynamic、非法 kind、空列表。 |
| **产品验收** | 与 Plan **§1.4** 一致：示例/说明中 **不** 鼓励把团队公共经验写入个人记忆（prompt 中一句约束）。 |
| **成功信号** | L1：抽查 3 组 mock，分类符合人设预期。 |

---

### T-B02 — 服务端写入路径：`kind` 落库 + 日志

| 项 | 内容 |
|----|------|
| **阶段** | Phase B |
| **优先级** | P0 |
| **依赖** | T-B01、T-A02 |
| **业务价值** | **Lite `memory_save`**（及共用抽取路径）写入的画像可被 **观测、排查**（运维与 PM 看日志即懂）。 |
| **范围** | `_try_extract_and_save_personal_memory`（或等价）传入并保存 `profile_kind`；日志含 static/dynamic **计数**或条数（与现有 `personal_memory:` 前缀一致）。 |
| **工程验收** | 集成或单测：mock 解析结果 → DB 行 kind 正确；超时/失败不阻塞主经验保存（保持现有契约）。 |
| **产品验收** | [troubleshooting §12](../../../design-docs/ops/troubleshooting.md) 可引用本节日志字段做排查。 |
| **成功信号** | L0：一次本机或 CI 跑通「解析 → 至少 1 行 kind 正确」。 |

---

### T-B03 — 语义 upsert 与 `profile_kind` 一致

| 项 | 内容 |
|----|------|
| **阶段** | Phase B |
| **优先级** | P0 |
| **依赖** | T-A02、T-B02 |
| **业务价值** | **相似**的个人事实在同一 kind 内合并，避免无限刷 duplicate；与「过期失效」前现有的 mem0 式体验一致。 |
| **范围** | `upsert_by_semantic`（或上层）在 **同一 `user_id` + `profile_kind`**（或文档规定的等价约束）内找最相似；阈值与 Plan §3.3 一致并在 `execute.md` **写死数字**；文档更新 `repository`/服务注释。 |
| **工程验收** | 单测：同 kind 高相似更新一行；跨 kind 不错误覆盖（用例由 execute 定义）。 |
| **产品验收** | **不过度设计**：矛盾裁决仍属 Non-Goal；本任务**仅**语义合并 + kind 边界。 |
| **成功信号** | L1：2 条用例（同 kind 合并 / 跨 kind 保留两行）通过。 |

---

## Phase C — MCP / Web

### T-C01 — Lite：`memory_context` 返回标准 `profile`

| 项 | 内容 |
|----|------|
| **阶段** | Phase C |
| **优先级** | P0 |
| **依赖** | T-A03、T-B02（读路径需有数据可选；可 mock） |
| **业务价值** | Cursor Lite 用户在 **单次调用** 内同时拿到 **用户画像 + 相关团队知识**，对齐「任务开始」心智。 |
| **范围** | `memory_context` JSON 增加或改为 `profile: { static: [], dynamic: [] }`；匿名：`{ static: [], dynamic: [] }`（Plan §7）；旧字段若保留须登记废弃日（Plan §6）。 |
| **工程验收** | `test_server_lite.py`（或等价）断言结构与非空场景；`make verify`。 |
| **产品验收** | **不新增工具名**；与 Plan §0 已决一致。 |
| **成功信号** | L0：Lite 侧 E2E 或契约测试绿。 |

---

### T-C02 — （可选）`memory_recall`：`include_user_profile`

| 项 | 内容 |
|----|------|
| **阶段** | Phase C |
| **优先级** | **P1**（默认）；若 execute 升为 **P0**，则与 T-C01 同列发布门槛。 |
| **依赖** | T-A03 |
| **业务价值** | **只调 recall、不调 `memory_context`** 的集成仍能捎带 `profile`，不新增 MCP 工具。 |
| **范围** | `memory_recall` 增加 **`include_user_profile: bool = False`**；`True` 时在返回 JSON 附加 `build_profile_for_user` 结果。 |
| **工程验收** | **`test_server_lite.py`**：`False` 行为不变；`True` 带回 `profile`。 |
| **产品验收** | Plan §3.4 / §7：P0/P1 在 execute 勾选；**`mcp-patterns.md` 只写 Lite 工具**。 |
| **成功信号** | L0：测试 + 文档同时落地。 |

---

### T-C03 — Web：用户画像按 static / dynamic 分组展示

| 项 | 内容 |
|----|------|
| **阶段** | Phase C |
| **优先级** | P1（强烈建议与 MVP 同学期，支撑 **可纠错** 产品叙事） |
| **依赖** | T-A02 |
| **业务价值** | 用户 **看得见、删得到** 错误画像，降低「AI 乱记」不信任（Plan §10）。 |
| **范围** | UI 分组；必要时 list API `profile_kind` 过滤（Plan §3.4）。 |
| **工程验收** | `make lint-js`；冒烟或 Playwright（若项目已有）。 |
| **产品验收** | **MVP 叙事闭环**：对外文档一句「错误条可通过 Web/HTTP 删除」（Plan §10）。 |
| **成功信号** | L1：PM 或内测 **5 分钟内** 完成「找到一条 → 删除 → 刷新消失」。 |

---

## Phase D — 规则、文档、发布门槛

### T-D01 — Instructions + `.cursor/rules` + 调用约定

| 项 | 内容 |
|----|------|
| **阶段** | Phase D |
| **优先级** | P0 |
| **依赖** | T-C01 |
| **业务价值** | **实际 Agent** 在任务开始时拉取画像；否则功能「做了但无人用」，ROI 为 0。 |
| **范围** | **`server_lite.py` Instructions** + `.cursor/rules`：**任务开始 `memory_context`**；消费返回中的 **`profile`**；**不提 `tm_*` MCP**；不写幽灵 **`tm_preflight`**。与 Plan **§1.4**（profile vs Experience）一致。 |
| **工程验收** | 文本 diff 评审；面向用户的文档/规则中 **不得** 将 **`tm_suggest` / `tm_search` / `tm_learn`** 列为必选接入（除非代码仓库恢复完整 MCP）。 |
| **产品验收** | **§9.3 前置条件**：规则可被 dogfood 执行（非僵尸文档）。 |
| **成功信号** | L2 准备就绪：可按 Plan §9.3 做 A/B。 |

---

### T-D02 — 运维与对比文档同步

| 项 | 内容 |
|----|------|
| **阶段** | Phase D |
| **优先级** | P0 |
| **依赖** | T-B02、T-C01 |
| **业务价值** | 降低支持成本；对齐竞品叙事便于对外沟通。 |
| **范围** | `troubleshooting.md` §12 / 个人记忆；`supermemory-comparison.md` 用户画像行更新为 **「部分/全部支持」**（以实际 MVP 为准）；`getting-started` 或 `security` **一句** 信任与删除（Plan §10）。 |
| **工程验收** | 文档 PR 通过 harness/doc 约定（若有）。 |
| **产品验收** | 对比表 **不夸大**（无 `valid_until` 前勿写「自动遗忘」）。 |
| **成功信号** | L0：新人按文档可完成一次排错。 |

---

### T-D03 — Execute 常量化与 P2 Backlog 挂链

| 项 | 内容 |
|----|------|
| **阶段** | Phase D |
| **优先级** | P1 |
| **依赖** | T-A03（T-C02 **memory_recall** P0/P1 决策可与本任务合并勾选） |
| **业务价值** | 避免「每个开发者猜 N 和策略」；P2 不遗忘、不阻塞发版。 |
| **范围** | 在 `execute.md` 写入 Plan §7 **全部常量**（N、匿名形态、**`memory_recall(include_user_profile)` P0/P1**、跨 project、E4 retention）；新建或链接 **P2 backlog**（关自动抽取、导出、RBAC 等）。 |
| **工程验收** | `execute.md` / tasks 交叉链接完整。 |
| **产品验收** | **范围可控**：P2 明确 **不** 进本迭代 DoD。 |
| **成功信号** | 任意任务执行人 **无需再问** 默认值。 |

---

## MVP 收口（产品向，可与 Phase D 并行）

### T-V01 — MVP 演示与 §9.3 最小评价协议 **一轮**

| 项 | 内容 |
|----|------|
| **阶段** | MVP 收口 |
| **优先级** | P0（**对外宣称「画像就绪」门槛**，Plan §9.5） |
| **依赖** | T-D01、T-C01（若集成方 **仅 recall**：须 T-C02 已完成再测「profile 注入」） |
| **业务价值** | 证明 **对项目有帮助** 或有记录地说明「无差异」，避免仅工程 DoD 即宣称成功。 |
| **范围** | 按 Plan **§9.3** 执行 1 轮 A/B；结果写入 `execute.md` 或 `3-retro/retro.md`；可选附录简单表格（A/B 次数对比）。 |
| **工程验收** | 无强制代码；须有 **书面记录**。 |
| **产品验收** | **至少满足之一**：B 不劣于 A 且一项改进；或 **无差异** 且记录原因（规则/数据/任务类型）。 |
| **成功信号** | L2：**可追溯**；满足 Plan §6.2。 |

---

### T-V02 — MVP 工程总闸门

| 项 | 内容 |
|----|------|
| **阶段** | MVP 收口 |
| **优先级** | P0 |
| **依赖** | Phase A～D 全部 P0 任务 |
| **业务价值** | 一次性发布质量门禁。 |
| **范围** | `make verify`（或 repo 标准 `make lint` + `make test`）；无硬编码密钥；个人记忆写入路径 **至少 1 条** 自动或半自动用例（Plan §6.1）。 |
| **工程验收** | CI / 本地全绿；清单勾选 Plan §6.1。 |
| **产品验收** | **不新增 MCP 工具名**；**发布说明 / 规则** 不以 **`tm_*` MCP** 为对外接口；grep `tm_profile` 等应为 0 或仅 Non-Goal 文档。 |
| **成功信号** | L0：发布标签可打 `mvp-profile-v1`。 |

---

## Phase E — 时间敏感失效（续包）

### T-E01 — `valid_until` 迁移与 ORM

| 项 | 内容 |
|----|------|
| **阶段** | Phase E |
| **优先级** | P0（E 包阻塞） |
| **依赖** | T-V02（或团队约定可与 MVP 末周并行） |
| **业务价值** | 「我明天考试」类事实 **到期不再误导 Agent**，对齐 Supermemory 自动遗忘维度。 |
| **范围** | `valid_until TIMESTAMPTZ NULL`；ORM；Web list **默认过滤**过期。 |
| **工程验收** | 迁移 up/down；`NULL` 行行为不变。 |
| **产品验收** | MCP `profile` **仍不暴露**日期字段（Plan §3.6）。 |
| **成功信号** | L0：升级后未登用户/MCP 行为与 MVP 一致。 |

---

### T-E02 — 抽取与写入 `valid_until`

| 项 | 内容 |
|----|------|
| **阶段** | Phase E |
| **优先级** | P0 |
| **依赖** | T-E01、T-B01 |
| **业务价值** | 时间敏感事实 **可进库可解析**，否则列空置。 |
| **范围** | Prompt + 解析 ISO/相对日期 → UTC；失败 → `NULL`；保守策略写 execute。 |
| **工程验收** | Mock + 1 条集成：写入后 DB 列正确。 |
| **产品验收** | **无矛盾推理**：文档一句强调到期 **≠** 解决互斥事实。 |
| **成功信号** | L1：短周期 + 长周期各 1 测例。 |

---

### T-E03 — 读路径过滤与 upsert「优先未过期 / 复活」

| 项 | 内容 |
|----|------|
| **阶段** | Phase E |
| **优先级** | P0 |
| **依赖** | T-E01、T-A03、T-B03 |
| **业务价值** | Agent **看不见**过期行；新事实可 **覆盖/复活** 过期占位，避免僵尸行挡路。 |
| **范围** | `list_for_pull`、`build_profile_for_user`、`list_by_user`（MCP 用）统一 `valid_until` 条件；`find_most_similar` 优先未过期，仅过期命中则 update。 |
| **工程验收** | 单测：过期不出现在 profile；复活路径；未过期 upsert 回归。 |
| **产品验收** | Plan §6.3 三项全勾选。 |
| **成功信号** | L1 + §9.4「到期命中」抽样可填。 |

---

### T-E04 — （可选）物理清理任务

| 项 | 内容 |
|----|------|
| **阶段** | Phase E |
| **优先级** | P2（可选） |
| **依赖** | T-E03 |
| **业务价值** | 减表体积；**不**替代查询过滤。 |
| **范围** | 定时任务；配置 retention；**禁止**删 `valid_until IS NULL`。 |
| **工程验收** | 单测或集成：仅删过期且超 retention；日志可审计。 |
| **产品验收** | 运维文档 **一行** 说明与 Agent 可见性无关。 |
| **成功信号** | L0：staging 跑一次无误删。 |

---

### T-E05 — 文档：自动遗忘「部分支持」

| 项 | 内容 |
|----|------|
| **阶段** | Phase E |
| **优先级** | P0（随 E 发布） |
| **依赖** | T-E03 |
| **业务价值** | 对外承诺与实际一致，利于信任与选型沟通。 |
| **范围** | `supermemory-comparison.md`、troubleshooting 更新；**不** 宣称全量矛盾处理。 |
| **工程验收** | 文档 PR。 |
| **产品验收** | PM 快速审：无营销夸大。 |
| **成功信号** | L0：与 Plan §3.6 表一致。 |

---

## 任务依赖图（摘要）

```
T-A01 → T-A02 → T-A03
              → T-B01 → T-B02 → T-B03
T-A03,T-B02 ──┬→ T-C01 ──→ T-D01 → T-V01
              ├→ T-C02（可选）        ↘
              └→ T-C03（Web）         T-V02
T-D02、T-D03 与 D 阶段交错；D01 仅依赖 C01

MVP 后：
T-E01 → T-E02 → T-E03 → T-E05
                  └→ T-E04（可选）
```

---

## 与 Plan 章节对照

| Plan | 任务 |
|------|------|
| §4.2 Phase A | T-A01～A03 |
| §4.3 Phase B | T-B01～B03 |
| §4.4 Phase C | T-C01～C03（Lite + Web） |
| §4.5 Phase D | T-D01～D03 |
| §6.2 / §9.3 | T-V01 |
| §6.1 总闸门 | T-V02 |
| §4.6 Phase E | T-E01～E05 |
