# File + Line Association and Expiration for Experience/Memory

**Purpose:** Summarize how Copilot and similar tools associate context with file/line ranges and handle staleness; propose three options for a TeamMemory-style system.

---

## 1. How Copilot and Similar Tools Handle File/Line and Staleness

### GitHub Copilot (Agentic Memory)

- **Association:** Memories are stored with **citations** that are **file path + line references** (e.g. `src/client/sdk/constants.ts:12`). Commit SHA is **not** stored.
- **Staleness / line drift:** When a memory is considered for use, Copilot **validates citations against the current codebase** (and current branch). If the cited locations no longer exist or no longer match (e.g. lines shifted or code changed), the memory is **not used**; if the code contradicts the memory, Copilot may store a **corrected** memory.
- **Expiration:** Memories are **automatically deleted after 28 days**. If a memory is validated and used, a new memory with the same details may be stored, effectively **refreshing** the 28-day window so that frequently used memories persist longer.
- **Rationale (inferred):** Just-in-time validation avoids relying on commit history and handles line drift by “use only if still valid”; the fixed TTL limits impact of abandoned branches and closed PRs and keeps the memory set bounded.

### Cursor (Codebase Index)

- **Association:** The **codebase index** stores **file path + start/end line** (and chunk metadata) for each embedded code chunk; retrieval can target file and line ranges. **Memories** are workspace-scoped and documented as general persistent context—**no public description** of binding memories to specific file/line ranges.
- **Staleness:** Index freshness is handled by **periodic sync** (e.g. every ~5–10 minutes) and **Merkle-tree change detection**: only changed files are re-indexed; for modified files, **old embeddings are removed and new ones created**. So “line drift” is handled by **re-indexing**, not by validating stored line references at read time.
- **Expiration:** No **TTL for location-based context** is described in public docs; staleness is addressed by keeping the index in sync with the workspace.

### Codeium / Windsurf / Codium (PR-Agent)

- **Association:** Context is often **structure-based** (e.g. enclosing function/class) rather than fixed line ranges; Windsurf supports **@-mentions** of files, directories, and symbols. **Remote indexing** is branch-based with configurable re-indexing; **memories** are workspace-level.
- **Staleness:** Handled by **re-indexing** and **dynamic context** (e.g. “three lines before/after” or enclosing block) so that small edits do not break references. No explicit **path+line binding with validation** like Copilot in public docs.
- **Expiration:** No **explicit TTL** for location-based context in the material reviewed.

---

## 2. Expiration (TTL) in Practice

- **Copilot** is the only system reviewed that **explicitly exposes a fixed TTL** (28 days) for location-backed memory. Justifications implied in docs: avoid stale information affecting decisions; limit impact of unmerged/closed work; bound retention.
- **A configurable default (e.g. 30 days)** is easy to justify: (1) align with common “monthly” review cycles and branch life; (2) cap storage and relevance window; (3) reduce risk of applying outdated patterns after large refactors. Making it configurable allows stricter (e.g. 7 days) or looser (e.g. 90 days) policies per deployment.

---

## 3. Three Options for TeamMemory-Style Systems

**Option A — Path + line range, just-in-time validation, fixed TTL (e.g. 30 days)**  
Store bindings as `path`, `start_line`, `end_line`. On retrieval (e.g. when the user is editing that file/range), check the **current file**: if the line range still exists and (optionally) content is still consistent, use the experience and optionally refresh the TTL; otherwise skip or mark stale. Experiences not revalidated within the TTL are deleted or deprioritized.

- **Pros:** No dependency on commit SHA; validation naturally handles line drift and refactors; simple mental model (Copilot-style).
- **Cons:** Requires reading the current file at retrieval time; strict TTL can drop long-lived but rarely revalidated knowledge.

**Option B — Path-only (or path + fuzzy region), optional TTL**  
Store only **file path** (or path + coarse region, e.g. “top/middle/bottom”). Retrieval: when the user is in that file, boost experiences bound to that path; no precise line-range check. TTL (e.g. 30 days default) is configurable for cleanup.

- **Pros:** No line-drift problem; simple storage and retrieval; no file read at query time for validation.
- **Cons:** Less precise; many experiences per file can clutter ranking without line-range or symbol cues.

**Option C — Path + line range + optional content fingerprint, TTL + invalidation on change**  
Store `path`, `start_line`, `end_line`, and optionally a **content hash or short snippet** of the bound code. Apply a TTL as in A. On retrieval, if the file’s **modification time or content hash** indicates it changed since the binding was created, either (1) try to **re-anchor** (e.g. find the snippet elsewhere in the file) or (2) **downgrade** to path-only relevance or mark the binding stale.

- **Pros:** Can survive small edits by re-anchoring; clear invalidation when the bound region is removed or heavily changed.
- **Cons:** More implementation and storage (fingerprint/snippet); re-anchoring logic is non-trivial and may be unnecessary if just-in-time validation (Option A) is enough.

---

*Report generated for design discussion; no code changes. Sources: GitHub Copilot Memory docs and blog, Cursor codebase indexing docs, project survey `experience-commit-binding-survey.md`, and web search.*

---

## 实现：内容指纹与 location_weight

本小节描述当前实现方案，完整任务与约定见实现计划 [经验文件位置绑定 — 实现计划](../plans/2025-03-10-experience-file-location-binding.md)。

### 数据结构：file_locations 与 current_file_locations

**file_locations**（写入：`tm_save` / `tm_save_typed`、Web 创建/更新经验时可选传入）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 文件路径 |
| `start_line` | int | 否 | 起始行（从 1 计） |
| `end_line` | int | 否 | 结束行 |
| `snippet` | string | 否 | 绑定时的代码片段，用于内容指纹计算 |
| `file_mtime` | float | 否 | 绑定时文件 mtime，用于失效校验 |
| `file_content_hash` | string | 否 | 绑定时文件内容 hash，用于失效校验 |

**current_file_locations**（读取：`tm_search` / `tm_solve` 检索时可选传入，表示「当前编辑/关注的位置」）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 文件路径 |
| `start_line` | int | 否 | 起始行（从 1 计） |
| `end_line` | int | 否 | 结束行 |
| `snippet` | string | 否 | 可选，用于指纹重锚 |
| `file_mtime` | float | 否 | 可选，用于失效判断 |
| `file_content_hash` | string | 否 | 可选，用于失效判断 |

服务端据此做**批量查询**（按 path 拉取未过期绑定）、在管道**内存**中计算 `location_score`（0 / 0.7 / 1.0），并与 RRF 得分按 `location_weight`（默认 0.15）融合：`final_score = rrf_score + location_weight * location_score`。

### 过期绑定清理策略与配置

- **策略**：仅保留 `expires_at > now` 的绑定；过期记录通过**定时清理任务**物理删除，避免表膨胀。
- **配置项**（见 `config.py` 中文件位置绑定配置块）：
  - **cleanup_enabled**（`file_location_cleanup_enabled`）：是否启用定时清理，默认 `True`。
  - **interval_hours**（`file_location_cleanup_interval_hours`）：清理任务执行间隔（小时），默认 `24`。
- 清理逻辑：按间隔调用 `delete_expired_file_location_bindings(session, batch_size=500)`，直到单轮删除数小于 batch_size 或为 0。

### 检索路径可观测性

- **批量查询**：`list_bindings_by_paths(session, paths, ttl_days)` 的调用量、延迟、失败应打日志或指标。
- **location 步骤**：管道内对该步骤记录：`current_file_locations` 数量、批量查询返回的绑定数、参与 location 加分的候选数、本步耗时；若批量查询或后续刷新失败，记录错误并降级为 `location_score=0`。
- 便于生产排查排序异常与性能问题。
