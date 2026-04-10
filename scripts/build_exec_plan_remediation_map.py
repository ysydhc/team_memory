#!/usr/bin/env python3
# ruff: noqa: E501
"""Emit .tmp/data/tm-remediation/exec-plan-bodies.json for POST /api/v1/archives.

Run from repo root: python scripts/build_exec_plan_remediation_map.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".tmp/data/tm-remediation/exec-plan-bodies.json"

COMMON_TAGS_PREFIX = ["exec-plan", "completed", "team_memory"]


def _p(
    slug: str,
    value_summary: str,
    overview: str,
    solution_doc: str,
    *,
    extra_tags: list[str] | None = None,
) -> dict:
    tags = [*COMMON_TAGS_PREFIX, slug, *(extra_tags or [])]
    return {
        "title": f"【exec-plan·completed】{slug}",
        "content_type": "tech_design",
        "value_summary": value_summary,
        "scope": "project",
        "scope_ref": (
            f"主题 {slug}：原 completed 执行计划目录已自仓库移除，全文见 Team Memory 档案馆附件"
        ),
        "tags": tags,
        "overview": overview.strip(),
        "solution_doc": solution_doc.strip(),
    }


BODIES: dict[str, dict] = {
    "archive": _p(
        "archive",
        "Harness 九任务与九个 commit 的对应与命名建议（非功能 plan）",
        """## 说明
本主题导出正文并非某条功能 Plan，而是 **9 个已完成 Harness 任务与仓库中 9 个 commit 的映射表**（多份 plan 合并跟踪）。用于审计「任务 ID ↔ 落地提交」。

## 要点
- 计划来源三组：p4_3_p2_5_p3_7、p1_6_p1_5_p2_3、p1_10_p1_9_p2_2。
- **P3-7**（详细使用日志）标注为本次未做。
- 若干 commit 合并多议题（如依赖与 /metrics）；表格给出「对应任务」与可选 message 建议。
- 若需功能级归档，请以对应主题为 slug 的档案为准。""",
        """## 1 三组计划与任务 ID
| 计划分组 | 任务 |
|----------|------|
| p4_3_p2_5_p3_7 | P4-3 CI/CD；P2-5 工作流模板；P3-7 详细日志（未做） |
| p1_6_p1_5_p2_3 | P1-6 FTS 迁移；P1-5 检索解析/质量门；P2-3 请求日志 |
| p1_10_p1_9_p2_2 | P1-10 成功标准文档；P1-9 extraction 配置；P2-2 /metrics |

## 2 当前 commit 映射（语义压缩）
原文表逐条列出：commit 短哈希、当前 message、对应任务、建议 message。核心结论：**P1-9** 在 MCP instructions 提交中体现；**P1-5** 与 config/llm_parser 共享部分 commit；**P1-10** 以 README 文档提交承载；**P2-2** 与 chore 依赖合并等。

## 3 使用建议
- 代码考古时：先查本表再 `git show`。
- 与 **archive-attachment / archive-knowledge-system** 等功能 Plan 区分：本档不描述档案馆或上传实现。
""",
    ),
    "archive-file-upload-mvp": _p(
        "archive-file-upload-mvp",
        "档案馆 multipart 本地上传 MVP；失败表 + curl；已被知识归档总 Plan 取代",
        """## 目标
在**无对象存储**前提下，为**已存在档案**提供 HTTP multipart 上传/流式下载；大文件落盘；检索仍以 title/overview/solution_doc 为主。

## 状态
**SUPERSEDED**：并入 `archive-knowledge-system` 总 Plan；本文档作历史需求与任务拆分索引。

## 原则（摘）
- 附件属于档案；先建档案再上传；L2 附件 JSON 与 MCP 同形。
- 失败可运营：`archive_upload_failures` + Web 横幅 + **可复制 curl**（不含真实 API Key）。""",
        """## 1 范围
**做**：POST/GET upload+download、本地目录配置、失败记录 API、Web 列表＋失败 UI、路径安全、L2 字段 `storage`/`download_api_path`。
**不做**：对象存储、分片、扫描、列表页上传入口等。

## 2 设计冻结点
- **§4.4 顺序**：.part → replace → **仅成功后** INSERT 附件；禁止先 INSERT 再异步落盘山岩
- **鉴权**：与 `GET archive` 一致；404 路径 **不写** failure 行
- **uploads.disabled**：POST 503；GET 历史附件仍可下载（推荐）

## 3 REST（摘要）
- `POST .../archives/{id}/attachments/upload`
- `GET .../attachments/{aid}/file`
- `GET/PATCH .../upload-failures`

## 4 任务与 DoD（压缩）
配置、仓储、上传下载 pytest、路径安全、503、Web、L2 字段、文档、失败表迁移与 UI、（可选）列表角标 Phase 2。

## 5 验证
门禁 `make verify`；手工 H1–H6 见原文 §六（创建档案→上传→失败→curl 补救）。

## 6 风险
DB 与磁盘需同周期备份；多副本本地盘见 Helm 约束。
""",
    ),
    "archive-knowledge-system": _p(
        "archive-knowledge-system",
        "档案馆 OpenViking 式 L0/L1/L2；原子 upsert；Skill 双通路；scope=archive 过渡期",
        """## 目标
将档案馆从「上传仓库」升级为 **可渐进暴露的知识体**：Agent/Skill 写 L0/L1/L2；搜索侧增强档案 L0；`include_archives` 可灰度。

## Phase 0（动工前 Blocker）
- **P0.1** `title+project` 部分唯一索引 + 原子 `INSERT ON CONFLICT DO UPDATE`
- **P0.2** Skill **双通路**：默认 `curl POST /api/v1/archives`，Phase 4 起可选 `tm-cli`
- **P0.3** `memory_save(scope=archive)` **deprecation** 响应字段，下一 major 删除
- **P0.4** `.claude/skills/archive` **纳入 git**
- **P0.5** upsert 返回 `action`/`previous_updated_at`
- **P0.6** embedding：**overview 过短**时 fallback `solution_doc` 前 1000 字

## 共识（摘）
`memory_save`（原子经验）+ `/archive`（会话全景）；`content_type`/`value_summary`/`tags`；经验可 `linked_experience_ids`；反向 `archive_ids` on search。""",
        """## 1 模型与迁移（006）
Archive 新增 `content_type`、`value_summary`、`tags`；`overview` 升格为 L1；`solution_doc` 为 L2 主体；**唯一索引** `uq_archives_title_project`。

## 2 搜索
- `search_archives` L0：含 `content_type`、`value_summary`、`tags`、`overview_preview`、`attachment_count`。
- **include_archives**：dev 默认 True；prod 默认 False 直到稳定 `[R]`。
- 经验结果可带 `archive_ids`。

## 3 去重与 upsert
Repository 返回 created/updated；覆盖字段含 overview/solution_doc/tags/embedding；**附件不自动删**；经验链接 **每次全量替换**（POST 须带齐 `linked_experience_ids` 或接受清空——补救脚本从 meta 回填）。

## 4 `/archive` Skill
生成 L0/L1/L2 → **通路 A curl** / **通路 B tm-cli** → 可选附件上传 → 本地 plan 迁至 completed。

## 5 tm-cli
`archive` / `upload` 子命令；HTTP 调 API；`TEAM_MEMORY_API_KEY`。

## 6 memory_save(scope=archive)
保留行为 + `deprecated`/`migration` 字段 + 日志 WARNING。

## 7 Phase 路线图（压缩）
P0 阻塞项 → 模型/搜索/Skill/CLI/Web/API → 单测与文档。详细 Task 表、评审 `[R]` 与 Stop Hook 原文见附件 `plan.md`。

## 8 备份
附件含 `execute.md`、渐进式披露执行记录等；检索以本 L2 摘要 + 附件为准。
""",
    ),
    # Historical exec-plan copy: TM Web 「架构」页与 /api/v1/architecture/* 已下线。
    "code-arch-viz-gitnexus": _p(
        "code-arch-viz-gitnexus",
        "Web 架构页 + GitNexus Provider；context/graph/clusters/impact/经验挂载",
        """## 目标
TM Web 内嵌**架构可视化**（概览/集群/图/影响面），数据来自 **GitNexus**；`architecture.provider: gitnexus`；经验可绑定架构节点。

## 任务骨架
T1 配置与模型 → T2 Provider 工厂 → T3 GitNexusProvider → T4 FastAPI 路由 → T5 经验绑定与 experiences API → T6 前端页 → T7 图与侧栏 → T9 文档验收；**Bridge** B1/B2 可并行。

## 验收（摘）
配置可启动；context/clusters/graph/impact/experiences API 行为符合表 V1–V9；无 bridge 时降级提示；经验挂载与侧栏可读。""",
        """## 1 Provider
抽象 `ArchitectureProvider`；首版实现 **GitNexus**（meta.json 或 Bridge HTTP）。

## 2 HTTP
`/api/v1/architecture/context|clusters|cluster/{name}|graph|impact|experiences`；鉴权；无 Provider 时 503 或 `available:false`。

## 3 数据与绑定
`experience_architecture_binding` 或 `code_anchor` 约定；GET experiences by node。

## 4 前端
导航「架构」；context/stale；Tab 集群/图/影响面；Cytoscape 或等价；节点侧栏 impact + 经验。

## 5 Bridge（可选并行）
暴露 context/clusters/impact/graph 只读 API 对接 MCP/GitNexus。

## 6 参考文档
operations、provider-interface、data-mapping、integration、balanced-solution —— 详见原主题 `design/` 附件。

## 7 DoD
README/V1–V9；关闭 bridge 时页面不崩；经验主流程不受影响。
""",
    ),
    "dedup-group-misjudge": _p(
        "dedup-group-misjudge",
        "去重页经验组误判：子信息 API + Jaccard 过滤 + 合并/reembed/前端 v2",
        """## 问题
仅比父经验向量时，**两组父文案极似但子任务完全不同**会被标为高相似。

## 策略
API 暴露子信息 → **组感知过滤**（双组且子标题 Jaccard < 阈值则剔除该对）→ 合并/reembed 风险闭环 → 回归与防回退测试。

## 版本
- **v1**：Task 2+3+7（+ Task 1 最小 + 可选 Task 5）
- **v2**：Task 4 合并双父、Task 5 完整、Task 6 前端子信息与筛选、Task 1 文档完整""",
        """## 1 Task 2
`find_duplicates` 返回 `children_count`/`children_preview`；selectinload 批量加载。

## 2 Task 3
双方皆经验组且 **Jaccard < 0.2**（可配）→ **从结果剔除**（非降权）。

## 3 Task 7
回归：父极似子不同 → 不在列表；父向量含子后相似度行为；**防回退测试**：恢复仅父比较则失败。

## 4 v2
- **Task 4**：双父合并禁止或 reparent 子经验
- **Task 5**：reembed 分批限流 errors
- **Task 6**：卡片展示组与子 preview；筛选

## 5 验收
误判消失 + 测试与注释/原则文档到位 + ruff/pytest。

## 6 依赖顺序
2 → 3 → 7；6 依赖 2。
""",
    ),
    "experience-file-location-binding": _p(
        "experience-file-location-binding",
        "经验↔文件位置绑定；指纹单点；批量绑定查询；location_weight 融合",
        """## 目标
经验绑定 **path + 行范围 + 内容指纹**；变更后可重锚定；**TTL**（默认 30 天）与访问刷新；检索 `final_score = rrf_score + location_weight * location_score`。

## 锁定决策
指纹算法仅在 **`location_fingerprint.py`**；**禁止 N×M** 仓储调用，须 `list_bindings_by_paths` 批量 + 管道内内存打分；`location_weight` 默认 **0.15**；得分档 **1.0 / 0.7 / 0**。

## 任务跨度
配置与迁移 → 仓储 replace/list/delete_expired → utils → Experience save →（历史）tm_save MCP → Web schema/UI → SearchPipeline → 设置 API → 清理任务与 E2E。""",
        """## 1 模型 `experience_file_locations`
FK experience；path/start_line/end_line；content_fingerprint；mtime/hash_at_bind；expires_at；last_accessed_at；索引见 plan。

## 2 检索管道
1) 收集 paths → 2) **一次** `list_bindings_by_paths` → 3) 内存算 `location_score`（重叠+指纹）→ 4) 加权重排 → 5) 可选批量刷新 TTL。

## 3 得分（唯一口径）
- EXACT 1.0：范围重叠且 hash/指纹一致或重锚成功
- SAME_FILE 0.7：同 path 无重叠或仅 path
- NONE 0：多位置取 max 不累加

## 4 任务压缩对照表
| 任务 | 要义 |
|------|------|
| 1 | SearchConfig `location_weight` + TTL/cleanup 配置 |
| 2 | Alembic + ORM |
| 3 | replace + list_bindings_by_paths + delete_expired |
| 4 | normalize/fingerprint/window=20/find_in_lines |
| 5 | experience.save file_locations |
| 6 | MCP 透传（历史 tm_save） |
| 7 | Web create/update/detail 展示 |
| 8 | SearchPipeline 批量 + 日志 |
| 9 | MCP search 透传 current_file_locations |
| 10 | 设置 Web/API |
| 11+ | 定时清理、E2E、边界 Case |

## 5 验收
单测 + 配置默认值 + 防 N×M；retro/execute 见附件。

## 6 Non-goals
档案馆 TTL 另案；全库矛盾推理。
""",
    ),
    "harness-methodology-follow-up": _p(
        "harness-methodology-follow-up",
        "Harness 方法论四项跟进：通知时机、notify 日志、四文档强制、断点恢复 3.5",
        """## 背景
Phase 0～5 已完成；本 Plan **仅文档与规则**，补：系统通知何时调用、execute 是否记录 notify、Agent 必须预读文档清单、断点恢复步骤。

## 任务
1. **harness-plan-execution**：4 个 notify 时机（开始/人类决策点/中断/完成）
2. **harness-workflow-execution** 3.2/3.3：execute 条目含「通知：已调用 notify_plan_status.sh」
3. 两文件 5.3 / plan-execution：**必须加载 4 文档**（workflow-execution、subagent-workflow、feedback-loop、human-decision-points）；未加载不得进入 step-0
4. **3.5 断点恢复**：触发条件、reload execute、重载 4 文档、是否重跑 step-0
5. **harness-follow-up-backlog.md**：CI/tm/能力限制后续表

## 验收
规则与文档中可逐条 grep 到上述约定；backlog 文件存在。""",
        """## 1 Task 1～4 并行面
改动的文件路径：`harness-plan-execution.mdc`、`harness-workflow-execution.md`；交叉引用 step-0 与 3.5。

## 2 Task 5 产出
`harness-follow-up-backlog.md`（原在 harness-phase 主题下；目录已删）：编号/内容/现状/难度/依赖。

## 3 风险
notify 脚本不可用 → best-effort 仅写 execute 不中断；路径变更 → 人工确认。

## 4 成功指标
与原文 §四表格一一对应（通知时机、notify 字段、文档加载门槛、3.5 存在、backlog 完整）。
""",
    ),
    "harness-phase": _p(
        "harness-phase",
        "Harness Phase 2～5 合并：工作流、分层门禁、日志 JSON+doc-gardening、规则分离",
        """## Phase 2
先规划再执行、自审清单、`tm-commit-push-checklist` 功能验证、`human-decision-points`、AGENTS/docs、全量验证。

## Phase 3
`architecture-layers.md`、`harness_import_check.py`、Makefile/CI、`harness-engineering`、反向依赖修复。

## Phase 4
bootstrap JSON 日志（`src/team_memory/bootstrap.py`）、doc-gardening 设计+Golden Set+CI、**step-0 强制**基线报告。

## Phase 5
Phase3 收尾、Harness/tm 规则按 boundary 分离、Phase4 step-0 模板、feedback-loop 归档、Phase 编号重排、索引与 execute。
""",
        """## 1 Phase 2 Task 1～6
固化规划优先、plan-self-review、E2E 验证约定、人类决策点文档、AGENTS 行数等门槛。

## 2 Phase 3 Task 1～5
分层 L0–L3、import 脚本白名单、Makefile `harness-check` 类入口、文档同步、违规清零。

## 3 Phase 4 Task 1～6
可观测性 JSON、doc-gardening 扫描范围与 CI continue-on-error 策略、Golden Set。

## 4 Phase 5 Task 1～6
规则边界、workflow 文档补强、历史 Phase 文案迁移、索引。

## 5 后续
`harness-follow-up-backlog`：CI doc-gardening、harness-check、Phase 6 等。

## 6 执行模式
Subagent-Driven；依赖顺序见各 Phase 原文表。
""",
    ),
    "logging-system": _p(
        "logging-system",
        "io_logger 层级、文件日志 QueueListener、与 mcp_debug 边界",
        """## 配置（摘）
`LOG_IO_ENABLED`、`LOG_IO_DETAIL`（mcp/service/pipeline/full）、`LOG_IO_TRUNCATE`、`LOG_FILE_*`；环境变量 `TEAM_MEMORY_*` 覆盖。

## 架构
**bootstrap** 配置 Handler/Formatter；**QueueHandler + QueueListener** 异步写文件；有界队列满则丢记录+warning；shutdown 等待 drain（超时）。

## 边界
**io_logger** 与 **mcp_debug_log** 独立；可同时开；后者仅 MCP 本地 IO 调试。

## DEBUG
DEBUG 或 `TEAM_MEMORY_DEBUG=1` 时可不限制单日志文件大小（见原文）。""",
        """## 1 节点映射
track_usage → MCP；ExperienceService → service；SearchPipeline → pipeline；full 全链路。

## 2 调用图（语义）
config → bootstrap → 各层业务代码调用 `io_logger.log_internal`（或与现实现等价 API）。

## 3 QueueListener
start/stop 时机；`stop_background_tasks` 衔接 asyncio/thread 等待。

## 4 生产建议
增大 `file_max_bytes`、备份份数；可选按日轮转（后续）。

## 5 格式规范
见 `src/team_memory/bootstrap.py` 中 `_JsonFormatter`（必填字段、脱敏）与 `tests/test_logging_json.py`。

## 6 验收
execute/retro 与单测、`make verify` 原要求。
""",
    ),
    "mcp-io-debug-log": _p(
        "mcp-io-debug-log",
        "MCP 工具 IO 调试：track_usage 挂钩、截断、落盘、脱敏、Chat 目录隔离",
        """## 目标
`TEAM_MEMORY_MCP_DEBUG` 或 `TEAM_MEMORY_DEBUG` 时，在 **track_usage** 打工具入/出；控制台截断；完整内容可选落临时文件并限额清理。

## 关键采纳（评审）
- **R1.2 Chat 隔离**子目录 + 时间戳+UUID 文件名
- 异步/线程写盘；敏感字段列表脱敏；try/except 不改变工具异常传播
- 20 文件上限、权限 0o600、生产禁开文档

## 涉及
`mcp_debug_log.py` + `server.py` 装饰器。""",
        """## 1 实现要点
开关检测；序列化前长度判断；脱敏参数表；落盘路径解析（可写性降级 tempfile）。

## 2 风险表（压缩）
R1 性能；R2 敏感；R3 与 Hook/logger 语义；R4 路径；R5 序列化/磁盘失败 —— 缓解列见原 plan §2.2。

## 3 与 io_logger 关系
并行独立；debug MCP 专用；io_logger 服务可观测。

## 4 DoD
单测覆盖清理边界、脱敏样例；文档声明双环境变量语义。
""",
    ),
    "personal-memory-profile-alignment": _p(
        "personal-memory-profile-alignment",
        "personal_memories：profile_kind + MCP profile{static,dynamic}；Phase E valid_until",
        """## MVP（A～D）
- 表 `profile_kind`：`static`|`dynamic`
- **`memory_context`** 标准返回 `profile: { static:[], dynamic:[] }`；可选 **`memory_recall(include_user_profile)`**
- **不新增** MCP 工具名；全面 Lite
- 抽取、upsert、Web 分组、文档与规则同步

## Phase E（续包）
`valid_until`；过期行不参与 profile/pull；可选物理清理。

## Non-goals
官方连接器、全量矛盾推理、恢复 tm_* 完整 MCP、SaaS 计费。""",
        """## 1 数据流
parse_personal_memory → 带 kind 落库 → build_profile_for_user 分桶截断（如各 ≤20 条）。

## 2 Phase E 查询
所有读 profile：**`valid_until IS NULL OR valid_until > now()`**；Web 可 `include_expired`。

## 3 MCP 约定
仅扩展 JSON；匿名用户 **空 profile**；与 Experience 边界写清。

## 4 Phase 路线图（压缩）
| Phase | 内容 |
|-------|------|
| A | 迁移 + ORM + build_profile 骨架 |
| B | 抽取 kind + upsert |
| C | memory_context + 可选 recall + Web |
| D | rules + README ops |
| E | valid_until + 过滤 + 可选清理 |

## 5 验收 DoD
§6 MVP + §9.3 最小评价协议；Phase E §6.3。

## 6 文档债务
清除幽灵 `tm_suggest`/`tm_search` 作为对外契约。

## 7 附件
execute、tasks、Supermemory 链；实现以 `server.py`/`test_server.py` 为准。
""",
    ),
    "verification": _p(
        "verification",
        "TeamMemory 计划完成度：MCP 用例矩阵 + 双 Agent 执行 + 两轮审查",
        """## 用途
从**使用者视角**用 MCP + 单测验证核心路径是否兑现 Plan（评分、embedding、FTS、同义词、短查询、提取、Rules、工具描述、文档）。

## 结构
- **§一** 核心路径与验收表（Step 1～7）
- **§二/三** Agent1/Agent2 用例集 A/B（tm_preflight、tm_search、tm_solve、tm_save、tm_learn、tm_task、tm_feedback 等）
- **§四** Agent3 两轮审查要点
- **§五** 执行顺序与输出物

## 说明
文中 **tm_*** 工具名为历史全量 MCP 时期；当前仓库以 **Lite `memory_*`** 为准，本档保留作**当时的验证剧本**。""",
        """## 1 Step 与验证类型（摘）
| Step | 焦点 |
|------|------|
| 1 | feedback_hint、rating_weight |
| 2 | embed 含 tags/子标题 |
| 3 | FTS simple + jieba |
| 4 | 同义词扩展缓存 key；短查询 min_similarity |
| 5 | tm_learn 解析与重试 |
| 5b | 管道三阶段文档 |
| 6 | Rules 四文件 + Server instructions |
| 7 | README 等 |

## 2 执行协议
Agent1 跑集 A并落日志 → Agent2 跑集 B → Agent3 两轮审稿 → 完成度报告 + 待确认清单。

## 3 单测映射
test_scoring_feedback、test_embedding_text、test_tokenizer、test_synonym_expansion 等（以原文列表为准）。

## 4 历史上下文
若复用本 Plan，请先将工具名与路径替换为当前 Lite/Harness 文档中的等价能力。
""",
    ),
    "workflow-optimization": _p(
        "workflow-optimization",
        "工作流：文档清理、英文 step id、步骤门控、Web 步骤聚合、oracle 审计格式、$ref",
        """## 阶段流
0 决策（英文语义 step id、是否门控）→ 1 清理过期 workflow 文档 → 2 统一 YAML step id 与 oracle → 3 Rules 门控（tm_workflow_next_step）→ 4 Web 步骤/组进度 → 5 审计单行格式 → 6 Step 独立文件 + $ref + actions。

## 后续调研
OpenMemory vs TM：metadata 过滤、Query 优化、Dashboard、批量归档等 P0～P2。""",
        """## 1 阶段一（摘）
删/归档与 TM+tm_message 矛盾的老文档；`workflow-scheme` 已删等。

## 2 阶段二
`task-execution-workflow.yaml` 等：`step-coldstart`、`step-claim`、`step-execute`…

## 3 阶段三
tm-core / execute-task-workflow-by-match：**进入下一步前须调用 next_step**；acceptance 自检。

## 4 阶段四
`parse_workflow_steps_from_messages`、`GET tasks/:id/workflow-steps`、组进度 API、组完成复盘提示。

## 5 阶段五
oracle **仅解析首行**审计；防污染单测。

## 6 阶段六
steps/*.yaml、`$ref` 解析、actions 库、timeout_hint 等扩展字段。

## 7 依赖
原文阶段有向图：0→1→2→3→4→5→6。
""",
    ),
    "workflow-visualization-v1": _p(
        "workflow-visualization-v1",
        "Web Cytoscape 只读工作流图：拖入 YAML/目录、递归 $ref、循环与深度上限",
        """## 目标
Web **只读**可视化：拖单文件或目录；js-yaml + **递归** step `$ref`（深度≤10、循环检测）；Cytoscape 节点边；缩放平移。

## 非目标
编辑、保存、n8n/Dify 导入。

## 解析
以 **`workflow_oracle._resolve_step_ref` 单层语义**为基线；前端 **扩展递归**且与单层场景一致。

## Task 1～5
示例 YAML → 页面路由 → 解析+$ref → 构图渲染 → 拖放与错误文案。""",
        """## 1 $ref 策略
单文件无法解析外链 → 提示拖文件夹；目录收集 path→content 映射；相对路径基于主 YAML 目录。

## 2 边规则
`allowed_next` 多叉；`when[].next` 带 condition label；`when` 优先于 `allowed_next`。

## 3 文件（摘）
`workflow-viewer.js`、`index.html`、`app.js`、`pages.js`；入口可达设置/高级 Modal（以落地为准）。

## 4 验收 AC
AC-1～AC-10：入口、单文件、$ref 提示、文件夹成功、循环/超深报错、浏览器 getAsEntry 降级等。

## 5 风险
循环、深度、CDN、浏览器兼容 —— 见原文 §八。
""",
    ),
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)  # .tmp/data/tm-remediation
    OUT.write_text(json.dumps(BODIES, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT, "keys:", len(BODIES))


if __name__ == "__main__":
    main()
