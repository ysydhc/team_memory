# 档案馆知识归档系统实施计划

> **状态**：✅ 已完成  
> **执行记录**：[execute.md](../2-plan/execute.md) · [渐进式披露执行记录](../2-plan/archives-progressive-disclosure-execute.md)

> **目标**：将档案馆从"文件上传仓库"转型为 **OpenViking 式知识归档系统**——用户一句"归档"或 `/archive`，Agent 自动提取当前会话中的决策、方案、结论，按 L0/L1/L2 三级渐进暴露存入档案馆，未来 Agent 以最小 Token 成本检索定位知识。

**前置阅读**：[archive-attachment-to-experience.md](../../../../design-docs/archive-attachment-to-experience.md)、[档案馆首版 Plan](../../archive-attachment/1-plan/plan.md)。

**取代**：[archive-file-upload-mvp Plan](../../archive-file-upload-mvp/1-plan/plan.md)（标记为 superseded）。

**评审修订**：综合 2026-04-01 五角色虚拟评审（架构师、技术主管、运维、QA、产品）的 Blocker/High-Risk 发现后修订，修订项标注 `[R]`。

**工程现状（2026-04-03 校对）**：`server_lite.py`、`test_server_lite.py` 已不存在；MCP 与档案检索以 **`server.py`**、**`test_server.py`** 及 `memory_save(scope=archive)` **已硬错误** 的代码为准。表格中仍出现旧文件名列，仅作 Plan 起草时的历史痕迹。

---

## Phase 0：动工前必须解决（Blockers + High-Risk 前置）

> **原则**：以下 6 项在 Phase 1 开工前必须全部落实到 plan 设计中，部分需要在 Phase 1 migration/代码中体现。

### P0.1 [Blocker] 去重竞态条件 → 数据库约束 + 原子 UPSERT

**问题**：`SELECT → 判断 → INSERT/UPDATE` 是经典 check-then-act 竞态——并发 `/archive` + Stop Hook 可能同时通过 SELECT 并双双 INSERT，产生 title+project 重复行，且无 DB 约束兜底。

**修复**（在 migration 006 中一并实施）：

1. 为 `archives` 表添加 **部分唯一索引**：
   ```sql
   CREATE UNIQUE INDEX uq_archives_title_project
   ON archives (title, project)
   WHERE title IS NOT NULL;
   ```
2. Repository 层使用 **`INSERT ... ON CONFLICT (title, project) DO UPDATE`** 原子 upsert，消除竞态窗口
3. Upsert 方法返回操作类型，供上层决策（见 P0.5）

**验收**：并发测试——两个协程同时 upsert 同 title+project → 表中仅 1 行。

### P0.2 [Blocker] Skill 不能依赖未交付的 tm-cli → 双通路架构

**问题**：`/archive` Skill 在 §3.4 中唯一调用路径是 `tm-cli archive`，但 CLI 在 Phase 4 才交付。首次使用 100% `command not found`。

**修复**：Skill 采用 **双通路架构**：

| 阶段 | 通路 | 适用场景 |
|------|------|----------|
| **Phase 1 起** | Agent 直接 `curl POST /api/v1/archives`（HTTP API） | Skill 默认通路；Agent 本身可执行 Bash |
| **Phase 4 起** | `tm-cli archive`（CLI 封装） | Hook 自动化、非 Agent 场景 |

Skill 文件中两种方式都列出，Agent 优先用 curl，检测到 tm-cli 可用时切换。

**验收**：Skill 在 tm-cli 未安装时可正常归档。

### P0.3 [High-Risk] scope=archive 删除需过渡期 → Deprecation Warning

**问题**：直接删除 `memory_save(scope=archive)` 会让旧客户端收到通用错误，无迁移指引。

**修复**：

1. **Phase 3 不直接删除**，改为 **deprecation 路由**：`scope=archive` 仍然工作，但返回结果中追加 `"deprecated": true, "migration": "Use /archive skill or POST /api/v1/archives instead"`
2. 日志输出 `WARNING: scope=archive is deprecated, will be removed in next major version`
3. **下一个 major 版本**再彻底删除

**验收**：`scope=archive` → 正常创建 + 响应含 `deprecated: true` + `migration` 字段。

### P0.4 [High-Risk] Skill 文件可跟踪 → 仓库内存放

**问题**：`.claude/skills/` 在 `.gitignore` 排除范围内，Skill 不进 git，团队成员无法获取。

**修复**：

1. Skill 源文件直接放在 `.claude/skills/archive/SKILL.md`（Claude Code 可直接识别）
2. 无需 symlink，`/archive` 命令开箱即用
3. `.gitignore` 添加 `!.claude/skills/` 排除例外（或 Phase 2 再决定）

**验收**：`git ls-files .claude/skills/archive/SKILL.md` 输出文件路径。

### P0.5 [High-Risk] 去重覆盖需可感知 → Upsert 返回 action 类型

**问题**：同名归档静默覆盖，用户不知道旧内容被替换，可能丢失有价值内容。

**修复**：

1. Repository upsert 方法返回结构化结果：
   ```python
   # 新建
   {"action": "created", "archive_id": "xxx"}
   # 覆盖
   {"action": "updated", "archive_id": "xxx", "previous_updated_at": "2026-03-30T..."}
   ```
2. `POST /api/v1/archives` 响应中包含 `action` 字段
3. Skill 流程中：当 `action == "updated"` 时，告知用户"检测到同名档案（上次更新于 X），已覆盖"
4. **Phase 2 可选**：添加版本历史表（`archive_versions`），覆盖前快照旧版

**验收**：二次归档同 title → 响应 `{"action": "updated", "previous_updated_at": "..."}` + Skill 输出覆盖提示。

### P0.6 [High-Risk] Embedding fallback → 保留 solution_doc 1000 字符

**问题**：移除 solution_doc 做 embedding 后，旧 archives（overview 为截断文本或空）的向量质量退化，搜索召回率下降。

**修复**：`_embedding_text_for_archive()` 保留 fallback 逻辑：

```python
def _embedding_text_for_archive(
    title: str,
    value_summary: str | None,
    overview: str | None,
    content_type: str | None,
    tags: list[str] | None,
    solution_doc: str | None = None,  # fallback 源
) -> str:
    parts = [title or ""]
    if value_summary:
        parts.append(value_summary.strip())
    if content_type:
        parts.append(f"type: {content_type}")
    if tags:
        parts.append(f"tags: {', '.join(tags)}")
    if overview and len(overview.strip()) >= 50:
        parts.append(overview.strip()[:2000])
    elif solution_doc:
        # Fallback: overview 为空或过短时，用 solution_doc 前 1000 字符补位
        parts.append(solution_doc.strip()[:1000])
    return "\n\n".join(p for p in parts if p)
```

**验收**：overview 为空的 archive → embedding 包含 solution_doc 内容（非零向量）。

---

## 一、核心理念

### 1.1 与 `memory_save` 的本质区别

| 维度 | `memory_save` | `/archive` |
|------|---------------|------------|
| **触发者** | Agent 主动 | 人工或 Hook |
| **粒度** | 原子经验（一条 bug、一个决策） | 会话全景（整个 session 的主题汇总） |
| **内容来源** | Agent 在上下文中识别 | Agent 按 Skill 模板提取+压缩 |
| **时机** | 随时 | 会话结束 / 人工喊停 |

### 1.2 渐进暴露（L0 → L1 → L2）

| 层级 | 用途 | Token 预算 | 内容 |
|------|------|-----------|------|
| **L0** | 搜索列表/快速过滤 | ~100 tok | title + content_type + value_summary + tags + score |
| **L1** | 判断是否值得深入 | ~500-2000 tok | 结构化 overview（决策表、结论清单、适用条件） |
| **L2** | 完整知识体 | 无上限 | solution_doc + conversation_summary + 附件文件 |

**关键原则**：L0/L1 由调用端 Agent 生成结构化内容（非后端 LLM），L2 含完整文件通过 HTTP 上传。

---

## 二、对齐共识清单

| # | 决策 | 结论 |
|---|------|------|
| 1 | 工具拆分 | `memory_save`（Agent 主动，原子）+ Skill + `tm-cli archive`（人/Hook 触发，全景）。不新增 MCP 工具。 |
| 2 | L0 字段 | Archive model 新增 `content_type` + `value_summary` + `tags` |
| 3 | L1 生成 | 调用端 Agent 按 Skill 模板生成结构化 overview，不再截断 |
| 4 | L2 组装 | conversation_summary（Agent 压缩）+ Plan 文件（附件上传）+ git refs |
| 5 | Token 成本 | Agent 输出 ~500-1000 tok 给 L0+L1+summary；文件直传不过 MCP |
| 6 | 触发方式 | 人工（"归档" / `/archive`）或 Stop Hook |
| 7 | CLI 工具 | `tm-cli archive` + `tm-cli upload`，Agent 和 Hook 均可调用 |
| 8 | 搜索默认 | `include_archives` 灰度：本地开发 `True`，生产保持 `False` 直到稳定 `[R]` |
| 9 | content_type | 预定义枚举 + 允许自定义 |
| 10 | tags 存储 | ARRAY(String)（与 Experience 一致） |
| 11 | 旧接口 | `memory_save(scope=archive)` 先 deprecation warning，下一 major 删除 `[R]` |
| 12 | 去重策略 | 按 title+project 匹配，DB 唯一约束 + 原子 UPSERT，覆盖时返回 action 类型 `[R]` |
| 13 | 经验关联 | 方案 A —— Agent 显式传递 `linked_experience_ids`，Skill 引导 |
| 14 | 反向查询 | 经验搜索结果含 `archive_id`（若有关联） |
| 15 | 本地处理 | Plan 文件归档后移至 `.harness/plans/completed/` |
| 16 | 旧计划 | `2026-03-30` 文件上传 MVP 标记 superseded |
| 17 | Skill 双通路 | MVP 用 curl HTTP API，Phase 4 加 tm-cli 通路 `[R]` |
| 18 | Skill 存放 | 源文件在 `.claude/skills/archive/SKILL.md`，可跟踪 `[R]` |
| 19 | Embedding fallback | overview 为空或 <50 字符时，用 solution_doc 前 1000 字符补位 `[R]` |

---

## 三、技术设计

### 3.1 Archive Model 变更

**新增列 + 唯一约束**（migration `006_archive_knowledge_fields.py`）：

```python
# Archive 表新增
content_type: Mapped[str] = mapped_column(
    String(50), nullable=False, server_default="session_archive"
)  # session_archive, tech_design, incident_review, decision_record, ...自定义
value_summary: Mapped[str | None] = mapped_column(
    String(500), nullable=True
)  # 一句话价值（L0 展示）
tags: Mapped[list[str] | None] = mapped_column(
    ARRAY(String), default=list
)  # 标签（与 Experience 同构）
```

```sql
-- [R] 去重唯一约束（P0.1）
CREATE UNIQUE INDEX uq_archives_title_project
ON archives (title, project)
WHERE title IS NOT NULL;
```

**语义变更**（不改列，改用途）：

| 列 | 旧用途 | 新用途 |
|----|--------|--------|
| `overview` | 截断 fallback | **L1 结构化内容**（Agent 按模板生成） |
| `solution_doc` | 原始文档 | **L2 主体**（Agent 压缩后的完整叙事 + 引用） |
| `conversation_summary` | 可选摘要 | **L2 对话压缩摘要** |

### 3.2 搜索改造

**`search_archives()` L0 返回增强**：

```python
{
    "id": "...",
    "title": "...",
    "content_type": "session_archive",        # 新
    "value_summary": "一句话价值描述",          # 新
    "tags": ["python", "refactoring"],         # 新
    "score": 0.85,
    "overview_preview": "...",                 # L1 预览
    "linked_experience_ids": [...],
    "attachment_count": 2,
    "type": "archive",
}
```

**`include_archives` 灰度策略** `[R]`：

- 本地开发环境（`TEAM_MEMORY_ENV=development`）：默认 `True`，方便调试
- 生产环境：保持默认 `False`，待稳定后全量放开
- MCP instructions 中建议 Agent 传 `include_archives=True`
- 现有测试保持 `include_archives=False` 的原逻辑，**新增** `include_archives=True` 的专项用例 `[R]`

**经验反向关联**：经验搜索结果增加 `archive_ids` 字段（通过 `ArchiveExperienceLink` 查询）。

### 3.3 去重策略 `[R]`

**原子 UPSERT**（依赖 P0.1 唯一约束）：

```python
# Repository 层伪代码
async def upsert_archive(self, ...) -> dict:
    """INSERT ... ON CONFLICT (title, project) DO UPDATE; 返回 action 类型。"""
    result = await session.execute(
        insert(Archive).values(...).on_conflict_do_update(
            index_elements=["title", "project"],
            set_={...覆盖字段...},
        ).returning(Archive.id, Archive.updated_at)
    )
    row = result.one()
    is_update = row.updated_at != row.created_at  # 近似判断
    return {
        "action": "updated" if is_update else "created",
        "archive_id": row.id,
        "previous_updated_at": row.updated_at.isoformat() if is_update else None,
    }
```

覆盖范围：`overview`, `solution_doc`, `conversation_summary`, `tags`, `content_type`, `value_summary`, `embedding`。旧 attachments **不自动删除**（附件是增量的），experience links 重新关联。

### 3.4 `/archive` Skill `[R]`

**源文件**：`.claude/skills/archive/SKILL.md`（纳入 git 跟踪）。使用时 symlink 或复制到 `.claude/skills/archive/SKILL.md`。

```markdown
---
name: archive
description: 归档当前会话的知识到团队档案馆
user_invocable: true
---

## 归档流程

1. **确定主题**：如果用户未指定，基于当前对话推断并询问确认
2. **提取 L0**：
   - title: 简洁标题（<100 字符）
   - content_type: session_archive | tech_design | incident_review | decision_record
   - value_summary: 一句话价值（<200 字符）
   - tags: 3-8 个标签
3. **生成 L1（overview）**：按模板生成结构化内容

   ```
   ## 核心决策
   | 决策 | 选定方案 | 理由 |
   |------|----------|------|
   ...

   ## 关键结论
   - ...

   ## 适用条件与边界
   - 适用于：...
   - 不适用于：...
   ```

4. **生成 L2**：
   - solution_doc: 完整叙述（问题背景 → 方案选型 → 实施要点 → 验证结果）
   - conversation_summary: 对话关键转折点摘要（500-1000 tok）
5. **关联经验**：列出本次 session 中 `memory_save` 过的 experience_ids
6. **检测去重**：告知用户如果同名档案已存在，将被覆盖 `[R]`
7. **调用 API 完成归档**（双通路）`[R]`：

   **通路 A（默认）：curl HTTP API**
   ```bash
   # 将 L1 overview 写入临时文件
   OVERVIEW=$(cat <<'OVERVIEW_EOF'
   ...Agent 生成的 L1 内容...
   OVERVIEW_EOF
   )

   curl -s -X POST "${TM_BASE_URL:-http://localhost:9111}/api/v1/archives" \
     -H "Authorization: Bearer ${TEAM_MEMORY_API_KEY}" \
     -H "Content-Type: application/json" \
     -d "$(jq -n \
       --arg title "..." \
       --arg content_type "session_archive" \
       --arg value_summary "..." \
       --arg overview "$OVERVIEW" \
       --arg solution_doc "..." \
       --arg summary "..." \
       --arg project "team_memory" \
       '{title: $title, content_type: $content_type,
         value_summary: $value_summary, overview: $overview,
         solution_doc: $solution_doc, conversation_summary: $summary,
         project: $project, tags: ["tag1","tag2"]}')"
   ```

   **通路 B（tm-cli 可用时）：**
   ```bash
   python -m team_memory.cli archive \
     --title "..." --content-type "session_archive" ...
   ```

8. **上传附件**（如有 Plan 文件）：
   ```bash
   curl -X POST "${TM_BASE_URL}/api/v1/archives/${ARCHIVE_ID}/attachments/upload" \
     -H "Authorization: Bearer ${TEAM_MEMORY_API_KEY}" \
     -F "file=@.harness/plans/active-plan.md" -F "kind=plan_doc"
   ```
9. **本地归档**：将 Plan 文件移至 `.harness/plans/completed/`
```

### 3.5 `tm-cli` 命令行工具

新建 `src/team_memory/cli.py`，注册为 `pyproject.toml` entry point：

```toml
[project.scripts]
tm-cli = "team_memory.cli:main"
```

**推荐调用方式** `[R]`：`python -m team_memory.cli`（更可靠，无 PATH 依赖）；`tm-cli` 作为可选快捷方式。

**子命令**：

| 命令 | 作用 | 核心参数 |
|------|------|----------|
| `tm-cli archive` | 创建/更新档案 | `--title`, `--content-type`, `--value-summary`, `--tags`, `--overview-file`, `--solution-file`, `--summary`, `--linked-experience-ids`, `--project` |
| `tm-cli upload` | 向档案追加附件 | `--archive-id`, `--file`, `--kind`, `--snippet` |

**实现**：CLI 通过 HTTP 调用 Web API（`POST /api/v1/archives`、`POST .../attachments/upload`），不直接访问 DB。

**认证**：读取 `TEAM_MEMORY_API_KEY` 环境变量（与现有 MCP/Web 认证一致），不另设配置文件 `[R]`。

### 3.6 Stop Hook 集成

`.harness/hooks/stop-check.sh` 或独立 hook 在会话结束时提示归档：

```bash
# 在 stop hook 中追加
echo "Session completed. Consider running /archive to save knowledge."
```

可选：自动触发归档（需要 Agent 已生成内容，Phase 2）。

### 3.7 `memory_save` 接口瘦身 `[R]`

**不直接删除 archive 分支**，改为 **deprecation 过渡**：

```python
# scope=archive 仍然工作，但响应追加 deprecation 警告
if scope == "archive":
    logger.warning("memory_save scope=archive is deprecated, use /archive skill or POST /api/v1/archives")
    result = await _save_archive(...)  # 现有逻辑保留
    result_dict = json.loads(result)
    result_dict["deprecated"] = True
    result_dict["migration"] = "Use /archive skill or POST /api/v1/archives instead. scope=archive will be removed in next major version."
    return json.dumps(result_dict, ensure_ascii=False)
```

**同步更新**：
- `server.py` 中 `tm_save` 的 archive 逻辑同样加 deprecation
- `memory_save` tool description 移除 `scope='archive'` 推荐，改为提及 `/archive`
- README / MCP 运维文档标注 scope=archive 为 deprecated（原 `mcp-patterns.md` 已删除）

### 3.8 Web API 新增/调整

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/archives` | **新增**：创建或更新档案（原子 UPSERT）`[R]` |
| `GET` | `/api/v1/archives/{id}` | 现有，L2 返回增加新字段 |
| `GET` | `/api/v1/archives` | 现有，列表返回增加 L0 新字段 |
| `POST` | `/api/v1/archives/{id}/attachments/upload` | 现有上传端点（保留） |
| `GET` | `/api/v1/archives/{id}/attachments/{att_id}/file` | 现有下载端点（保留） |

**`POST /api/v1/archives`** `[R]` 拆分设计：

1. **Pydantic Schema**（`schemas.py`）：
   ```python
   class ArchiveCreateRequest(BaseModel):
       title: str = Field(..., max_length=500)
       solution_doc: str
       content_type: str = "session_archive"
       value_summary: str | None = Field(None, max_length=500)
       tags: list[str] | None = None
       overview: str | None = None
       conversation_summary: str | None = None
       linked_experience_ids: list[str] | None = None
       project: str | None = None
       scope: str = "session"
       scope_ref: str | None = None
   ```
2. **Route handler**（`web/routes/archives.py`）：**强制认证**（非 `get_optional_user`），调用 service 层
3. **响应**：`{"action": "created"|"updated", "archive_id": "...", "previous_updated_at": ...}`

### 3.9 Embedding 改进 `[R]`

```python
def _embedding_text_for_archive(
    title: str,
    value_summary: str | None,
    overview: str | None,
    content_type: str | None,
    tags: list[str] | None,
    solution_doc: str | None = None,  # [R] fallback 源
) -> str:
    parts = [title or ""]
    if value_summary:
        parts.append(value_summary.strip())
    if content_type:
        parts.append(f"type: {content_type}")
    if tags:
        parts.append(f"tags: {', '.join(tags)}")
    # L1 优先；为空/过短时用 solution_doc 前 1000 字符补位 [R]
    if overview and len(overview.strip()) >= 50:
        parts.append(overview.strip()[:2000])
    elif solution_doc:
        parts.append(solution_doc.strip()[:1000])
    return "\n\n".join(p for p in parts if p)
```

### 3.10 Web 详情页 L0/L1/L2 三段式布局 `[R]`

档案详情页从 `<pre>` 平铺改为三段式：

| 区域 | 内容 | 渲染方式 |
|------|------|----------|
| **L0 头部** | 标题 + content_type 胶囊 + tags 标签行 + value_summary 副标题 | HTML 结构化 |
| **L1 可展开区** | overview（Markdown 结构化内容） | Markdown → HTML 渲染（至少支持表格、列表） |
| **L2 折叠区** | solution_doc + conversation_summary + 附件列表 | 默认折叠，点击展开 |

列表卡片增加：content_type 胶囊 + value_summary 副标题 + tags 标签行。

---

## 四、任务拆分与验收

### Phase 1：数据层 + API（基础能力）

| # | 任务 | 验收标准 | 估计文件 |
|---|------|----------|----------|
| 1.1 | Archive model 新增 `content_type`, `value_summary`, `tags` + **UNIQUE 约束** + migration `[R]` | `alembic upgrade head` 成功；字段存在；唯一索引存在 | `models.py`, `migrations/006_*` |
| 1.2 | `archive_repository.py`：**原子 upsert** 方法（返回 action 类型）`[R]` | 同 title 二次 → `{"action": "updated"}`；并发测试通过 | `archive_repository.py` |
| 1.3 | `archive_repository.py`：`search_archives` L0 返回新字段 | 单测验证新字段存在 | `archive_repository.py` |
| 1.4 | `archive_repository.py`：经验反向查询（experience → archive_ids） | 单测 | `archive_repository.py` |
| 1.5 | `archive.py`：embedding 改用 L0+L1+fallback `[R]` | overview 为空时 embedding 含 solution_doc | `archive.py` |
| 1.6a | `schemas.py`：`ArchiveCreateRequest` schema 定义 `[R]` | Pydantic 校验测试 | `schemas.py` |
| 1.6b | `web/routes/archives.py`：`POST /api/v1/archives` 路由（强制认证）`[R]` | TestClient 验证：创建 + 更新 + 未认证 401 | `web/routes/archives.py` |
| 1.6c | `archive.py`：`archive_upsert` service 方法（串联 repo upsert + embedding）`[R]` | service 层单测 | `archive.py` |
| 1.7 | L2 `_build_l2_dict` 返回新字段 | 快照测试 | `archive_repository.py` |

### Phase 2：搜索改造

| # | 任务 | 验收标准 | 估计文件 |
|---|------|----------|----------|
| 2.1 | `include_archives` 灰度：dev 默认 `True`，prod 默认 `False` `[R]` | dev 环境测试返回 archive；prod 测试不返回 | `server_lite.py`, `server.py`, `config.py` |
| 2.2 | 经验搜索结果含 `archive_ids` | 单测 | `search_pipeline.py`, `repository.py` |
| 2.3 | 新增 `include_archives=True` 专项测试用例 `[R]` | 测试覆盖：True 返回 archive + False 不返回 | `test_server_lite.py`, `test_server.py` |
| 2.4 | Web：详情页三段式布局 + 列表卡片新字段 + Markdown 渲染 `[R]` | `lint-js` + 手工验证 L0/L1/L2 分层展示 | `web/static/js/*.js` |

### Phase 3：`memory_save` deprecation `[R]`

| # | 任务 | 验收标准 | 估计文件 |
|---|------|----------|----------|
| 3.1 | `memory_save(scope=archive)` 改为 deprecation warning（仍工作）`[R]` | 响应含 `deprecated: true` + `migration` 字段 | `server_lite.py` |
| 3.2 | 同步 `server.py` 中 `tm_save` 的 archive deprecation `[R]` | 同上 | `server.py` |
| 3.3 | 更新 tool description：移除 scope=archive 推荐，提及 `/archive` | description 无 scope=archive 推荐 | `server_lite.py`, `server.py` |
| 3.4 | 更新 README / mcp-server：标注 scope=archive deprecated + 新归档方式 | 文档一致 | 根 README、`docs/design-docs/ops/mcp-server.md` |
| 3.5 | **重写** `test_archive_mode` 测试为 deprecation 断言 `[R]` | 测试断言 deprecated=true | `test_server_lite.py` |
| 3.6 | 排查 `scripts/smoke_archive_session.py` 等依赖 scope=archive 的脚本 `[R]` | 所有调用方适配 deprecation 或迁移 | `scripts/` |

### Phase 4：CLI + Skill

| # | 任务 | 验收标准 | 估计文件 |
|---|------|----------|----------|
| 4.1 | `tm-cli` 骨架 + `archive` 子命令 | `python -m team_memory.cli archive --help` 输出参数说明 | `cli.py`, `pyproject.toml` |
| 4.2 | `tm-cli upload` 子命令 | `python -m team_memory.cli upload --archive-id <id> --file <path>` 成功上传 | `cli.py` |
| 4.3 | `/archive` Skill 源文件（纳入 git）`[R]` | `git ls-files .claude/skills/archive/SKILL.md` 有输出 | `.claude/skills/archive/SKILL.md` |
| 4.4 | CLI 认证（`TEAM_MEMORY_API_KEY` 环境变量）`[R]` | 无 key → 友好报错 | `cli.py` |
| 4.5a | CLI 单元测试（mock HTTP）`[R]` | 参数解析 + auth 注入 + 错误处理 | `tests/test_cli.py` |
| 4.5b | `POST /api/v1/archives` 集成测试（TestClient）`[R]` | 创建 + upsert + 缺字段校验 + 未认证 | `tests/test_web.py` |

### Phase 5：收尾

| # | 任务 | 验收标准 | 估计文件 |
|---|------|----------|----------|
| 5.1 | 标记旧 plan superseded | 文件头部增加 superseded 声明 | `archive-file-upload-mvp/1-plan/plan.md` |
| 5.2 | 更新 `CLAUDE.md` 导航 | 含新 plan 链接 | `CLAUDE.md` |
| 5.3 | 更新 MCP instructions（server_lite.py） | 描述反映新架构 | `server_lite.py` |
| 5.4 | `make verify` 全绿 | lint + 全量测试通过 | — |

---

## 五、功能验证

### 5.1 自动化（门禁）

`make verify`（`ruff` + `pytest` + `lint-js`）。

最小用例集：

- **归档创建**：`POST /api/v1/archives` → DB 有记录，L0 字段齐全，响应含 `action: created`
- **去重覆盖**：同 title+project 二次 POST → 只有 1 条记录，响应含 `action: updated` + `previous_updated_at` `[R]`
- **并发 upsert**：两协程同时 upsert → 表中仅 1 行，无异常 `[R]`
- **去重边界**：大小写、前后空格、空 tags、超长 overview、Unicode title `[R]`
- **搜索含档案（dev 环境）**：`memory_recall(include_archives=True)` 返回 archive 类型结果，含 L0 新字段
- **搜索不含档案（prod 默认）**：`memory_recall()` 不返回 archive（保持现有行为） `[R]`
- **经验反向关联**：关联了 archive 的 experience 搜索结果含 `archive_ids`
- **L2 完整**：`GET /api/v1/archives/{id}` 返回所有新字段 + 附件列表
- **CLI archive**：`python -m team_memory.cli archive --title ...` 成功创建
- **CLI upload**：`python -m team_memory.cli upload --archive-id <id> --file <path>` 成功上传
- **CLI 无 key**：未设 `TEAM_MEMORY_API_KEY` → 友好报错 `[R]`
- **memory_save deprecation**：`scope=archive` → 正常创建 + `deprecated: true` + `migration` 字段 `[R]`
- **Embedding fallback**：overview 为空的 archive → embedding 非零向量 `[R]`

### 5.2 手工验收清单

| 步骤 | 操作 | 预期 |
|------|------|------|
| H1 | 在 Claude Code 中执行 `/archive` | Skill 引导提取 L0/L1/L2 |
| H2 | Agent 调用 curl API 完成归档 `[R]` | 档案创建成功，响应含 action |
| H3 | Web 打开档案馆，查看新建档案 | L0 头部 + L1 Markdown 渲染 + L2 折叠区 `[R]` |
| H4 | `memory_recall(query="...", include_archives=True)` | 搜索结果含 archive 条目，L0 字段齐全 |
| H5 | 再次对同主题执行 `/archive` | Skill 提示"检测到同名档案，将覆盖" + 覆盖成功 `[R]` |
| H6 | Plan 文件已移至 `.harness/plans/completed/` | 本地归档完成 |

### 5.3 回归

- `memory_save(title=..., problem=..., solution=...)` 正常工作（scope=project / personal）
- `memory_save(scope=archive)` 仍然工作但含 deprecation 警告 `[R]`
- `memory_recall` 无 archive 的旧项目不报错
- Web 档案馆列表向后兼容（旧档案无新字段显示为空/默认值）

---

## 六、风险与缓解

| 风险 | 缓解 |
|------|------|
| Agent 生成的 L1 质量不稳定 | Skill 模板强约束格式；MVP 后可加后端 LLM 后处理 |
| CLI 未安装时 Skill 失败 | **双通路架构**：MVP 用 curl HTTP API，CLI 为可选增强 `[R]` |
| 去重误覆盖（不同内容同标题） | DB 唯一约束 + 原子 upsert + Skill 提示覆盖确认 `[R]` |
| 旧 scope=archive 调用方未迁移 | Deprecation warning 过渡期，非直接删除 `[R]` |
| 搜索结果 archive 淹没 experience | 灰度发布：dev `True` / prod `False`，渐进放开 `[R]` |
| 旧 archives embedding 质量退化 | Embedding fallback：overview 过短时保留 solution_doc 前 1000 字符 `[R]` |
| 并发归档竞态 | DB 唯一约束 + ON CONFLICT UPSERT `[R]` |

---

## 七、建议执行顺序

```
Phase 0 (设计冻结) → Phase 1 (1.1-1.7) → Phase 2 (2.1-2.4) → Phase 3 (3.1-3.6) → Phase 4 (4.1-4.5) → Phase 5 (5.1-5.4)
```

- Phase 0 的修复点已内嵌到各 Phase 任务中（标注 `[R]`），无独立实施阶段
- Phase 1 和 Phase 3 之间无强依赖，可并行
- Phase 4 依赖 Phase 1（API 端点存在）

**DoD**：`make verify` 全绿 + 手工验收 H1-H6 通过。

---

## 八、引用

- `src/team_memory/storage/models.py` — Archive model
- `src/team_memory/services/archive.py` — ArchiveService
- `src/team_memory/storage/archive_repository.py` — ArchiveRepository
- `src/team_memory/server_lite.py` — memory_save（待 deprecation）
- `src/team_memory/server.py` — tm_save（待同步）
- `src/team_memory/services/search_pipeline.py` — 搜索管道
- [archive-file-upload-mvp/1-plan/plan.md](../../archive-file-upload-mvp/1-plan/plan.md) — 被取代的旧计划

---

## 九、评审修订记录

| 日期 | 来源 | 变更 |
|------|------|------|
| 2026-04-01 | 首席架构师 | P0.1 去重唯一约束 + 原子 upsert |
| 2026-04-01 | 首席架构师 | P0.6 embedding fallback 保留 solution_doc 1000 字符 |
| 2026-04-01 | 首席架构师 | §3.2 include_archives 灰度（dev True / prod False） |
| 2026-04-01 | 产品经理 | P0.2 Skill 双通路（curl + CLI），不依赖未交付 CLI |
| 2026-04-01 | 产品经理 | P0.5 upsert 返回 action 类型，覆盖可感知 |
| 2026-04-01 | 产品经理 | §3.10 Web 三段式布局 + Markdown 渲染 |
| 2026-04-01 | 技术主管 | P0.3 scope=archive deprecation 过渡，非直接删除 |
| 2026-04-01 | 技术主管 | P0.4 Skill 源文件放 docs/skills/，纳入 git |
| 2026-04-01 | 技术主管 | §3.8 POST /api/v1/archives 拆为 schema + route + service 三任务 |
| 2026-04-01 | QA | §5.1 补充并发、边界、deprecation、fallback 测试用例 |
| 2026-04-01 | QA | Phase 3 补充 test_archive_mode 重写 + 脚本排查任务 |
| 2026-04-01 | QA | Phase 4 CLI 测试拆为 unit（mock HTTP）+ integration（TestClient） |
| 2026-04-01 | 运维专家 | §3.5 CLI 推荐 `python -m` 调用；认证仅用 TEAM_MEMORY_API_KEY |
