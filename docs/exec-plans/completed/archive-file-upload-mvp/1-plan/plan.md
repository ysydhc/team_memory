# ~~档案馆文件上传 MVP 实施计划~~ [SUPERSEDED]

> **归档索引**：✅ 执行过程已并入 2026-03-31 总 Plan；指针见 [README.md](../README.md)

> **状态**：已被 [档案馆知识归档系统总 Plan](../../archive-knowledge-system/1-plan/plan.md) 取代。
> 以下内容仅作历史参考。

---

# 档案馆文件上传 MVP 实施计划（原文）

> **目标**：在**不引入对象存储**的前提下，用 **multipart 落盘 + 可配置本地目录**，实现与**档案馆（Archive）同一套数据模型与权限**的文件归档能力。大文件走磁盘引用；检索仍以 **title + overview + 可选小片段** 为主以节省 Token。**上传失败**时落库可查，Web 提示并给出 **本机可执行的 curl**，由人工补救（与 Agent 触发场景衔接）。

**前置阅读**：[archive-attachment-to-experience.md](../../../../design-docs/archive-attachment-to-experience.md)、[档案馆首版 Plan](../../archive-attachment/1-plan/plan.md)。

**修订说明**：综合评审结论、**人机补救（curl）** 与 **失败记录表** 后的统一版本；冲突巡检与验证方式见 **§七**、**§九**。

**建议阅读顺序**：§一（范围）→ §二（原则）→ §三（选型）→ §四（设计 4.1–4.12）→ §五（任务）→ **§六（功能验证）** → §七（风险）→ §八（引用）→ §九（评审）。

---

## 一、范围与非目标

### 1.1 MVP 包含

- **HTTP**：在已有 FastAPI、档案馆鉴权下，向**已存在**档案 **追加附件**（multipart 上传）与 **流式下载**。
- **数据**：复用 `archive_attachments`；失败场景追加 **`archive_upload_failures`**（见 **§4.11**）。
- **存储**：本地目录 / Docker volume；按 `archive_id` 分子目录。
- **权限**：与 `GET /api/v1/archives/{id}` 一致（**§4.5**）。
- **Web**：详情页 **附件列表 + 下载**；**失败横幅 + 可复制 curl**（**§4.11**）；其余交互收窄见 **§4.7**。
- **文档**：大文件走 HTTP；Agent 自动化写清 **API Key + curl/Web**；备份与 Helm 见 **§4.10**、**§七**。

### 1.2 MVP 明确不做

- 对象存储、预签名 URL、分片上传、病毒扫描。
- 客户端指定服务器绝对路径读文件；Agent **大体积 base64** 直塞 MCP。
- 上传进度条、多文件队列、拖拽文件夹；列表页 **不**设上传入口（**§4.7**）。

---

## 二、与档案馆合并使用的核心原则

| 原则 | 说明 |
|------|------|
| **附件属于档案** | `archive_attachments.archive_id` 非空；API 路径挂在 `.../archives/{archive_id}/...`，无长期孤儿文件。 |
| **档案先于附件** | `solution_doc` NOT NULL；先 `memory_save(scope=archive)` 等创建档案，再上传。**Web 冷启动**：不设「零档案直传」单页（Phase 2）。 |
| **L2 一致** | HTTP `GET /archives/{id}` 与 `memory_get_archive` / `tm_get_archive` 的 `attachments[]` **同形**（**§4.6**）；不含二进制本体。 |
| **检索不绑全文附件** | 向量/FTS 仍以 title、overview、`solution_doc` 等为主；可选 `content_snapshot` 小片段（**§4.2**）。 |
| **失败可被人接管** | 业务层上传失败后 **可持久化失败行**，Web 展示并生成 **curl 模板**（**§4.11**），与 §4.4 **写入顺序不冲突**（见 **§4.12**）。 |

---

## 三、方案对比（为何档案下挂上传）

| 方案 | MVP 结论 |
|------|----------|
| **A. `POST .../archives/{id}/attachments/...`** | **采用**——权限单一、无孤儿上传。 |
| B. 全局 `POST /uploads` 再关联 | 不采用。 |
| C. 单请求同时创建档案 + 多文件 | Phase 2。 |

---

## 四、技术设计摘要

### 4.1 配置

- `uploads.enabled`：`false` 时 **`POST upload` → 503**（推荐 **`GET` 已存在附件仍可下载**，避免历史断供；须在代码注释写死）。
- `uploads.root_dir`、`uploads.max_bytes`、可选 `uploads.allowed_extensions`；可选 `uploads.max_total_bytes_per_archive`（否则仅靠单文件上限 + 运维）。
- 环境变量：`TEAM_MEMORY_UPLOADS_*`。

### 4.2 路径与安全

- 上传流 → `{root}/{archive_id}/{attachment_id}.part` → `replace` 至终路径；`ArchiveAttachment.path` 为 **相对 `root_dir` 的 posix 路径**。
- 下载：`realpath` 约束在 `realpath(root)` 之下；**表驱动** pytest（`..`、非常规 Unicode 等）。
- 扩展名白名单；**幂等**：每次新 UUID，不覆盖同名。

### 4.3 REST 草案

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/archives/{archive_id}/attachments/upload` | multipart：`file`、`kind`、`snippet`；413 / 503。 |
| `GET` | `/api/v1/archives/{archive_id}/attachments/{attachment_id}/file` | 流式下载；`Content-Disposition: attachment`。 |
| `GET` | `/api/v1/archives/{archive_id}/upload-failures` | 见 **§4.11**。 |
| `PATCH` | `/api/v1/archives/{archive_id}/upload-failures/{failure_id}` | 可选：`resolved`（**§4.11**）。 |

### 4.4 落盘与 DB 顺序（冻结）

1. 生成 `attachment_id`。  
2. 写入 `.part`，超 `max_bytes` → 删 `.part`，**413**，并按 **§4.12** 记失败（若已进入业务处理）。  
3. `os.replace` → 最终文件。  
4. **仅 3 成功后** `INSERT archive_attachments` + `commit`。  
5. `commit` 失败 → 删最终文件（best-effort）、error 日志；**§4.12** 记失败。  
6. 2–3 失败 → 不 INSERT；清理 `.part`。

**禁止**：先 INSERT 再异步落盘；禁止客户端写绝对路径入 `path`。

### 4.5 鉴权与错误语义（冻结）

- 横向越权 / 资源不可见：**HTTP 404**（与现有 `get_archive_detail` 一致）。
- 未登录行为与 `get_optional_user` **现状对齐**；实现前核对 `test_web.py`。
- **404 / 401（鉴权与档案不可见）**：**不写** `archive_upload_failures`（**§4.12**）。

### 4.6 L2 JSON 契约（HTTP + MCP 共用）

`attachments[]` 增量字段示例：

```json
{
  "id": "…",
  "kind": "file",
  "path": "…/…",
  "content_snapshot": null,
  "snippet": null,
  "storage": "local",
  "download_api_path": "/api/v1/archives/{archive_id}/attachments/{attachment_id}/file"
}
```

- `memory_get_archive` / `tm_get_archive` **同形**；无二进制。

### 4.7 Web SPA（MVP）

- 详情：附件下载链接（`aria-label`）、**可选**简易上传表单；无进度条。
- **失败 UI**：**§4.11**。

### 4.8 MCP / Lite

- 可不新增工具；大附件 **HTTP**；L2 附件 JSON **Lite/Full 一致**。

### 4.9 可观测性与备份

- 结构化日志（成功/失败、`archive_id`、字节数、`attachment_id`、原因）。
- DB 与 `uploads.root_dir` **同周期备份**（运维文档）。

### 4.10 Kubernetes / Helm

- **单磁盘**与无共享存储的 **多副本** 不兼容：`replicas: 1` + 卷，或 **RWX PVC** / 对象存储（详见 `helm/` 与 ops 文档）。

### 4.11 上传失败 · Web 提示 · 本机 curl

**目标**：Agent 或 API 上传失败后，**有权限用户**打开 Web 同一档案，看到记录并 **复制命令** 在本机重传。

**表 `archive_upload_failures`**（名可微调）：`id`、`archive_id` FK、`created_at`、`created_by`（可空）、`source`（`agent`|`web`|`api`）、`error_code`、`error_message`（≤500）、`client_filename_hint`（可空）、`resolved_at`（可空）。

**API**：`GET .../upload-failures?limit=&include_resolved=`；可选 `PATCH .../resolved`。

**写入时机（与 §4.5、§4.12 一致）**：在 **已确认用户对 `archive_id` 具备与 L2 相同的访问资格**（通过可见性校验）之后，若 **`POST upload` 未形成成功的 `archive_attachments` 行**（含校验错误、413、503 `uploads.disabled`、磁盘错误、`commit` 失败等），**插入一行**。  
**不写记录**：请求在「档案不可见 / 未授权」路径即以 404/401 结束。

**Web**：详情页可折叠横幅「有 N 次上传未成功」→ 时间、来源、错误摘要、文件名提示 → **「复制上传命令」**。  
命令由前端用 `window.location.origin`、当前 `archive_id`、**占位符** `YOUR_API_KEY` 与 `@/path/to/your/file` 生成；**页面永不持久化、不展示真实 Key**。若生产环境 **仅 cookie、无 API Key**，须在根目录 **README** / Web 文档说明 **curl 的认证替代方案**（与现网一致）。

**Agent**：失败响应可提示用户到 Web 该档案详情使用 curl。

### 4.12 失败记录与 §4.4 的衔接（无冲突声明）

- 失败行 **不参与**旧附件路径；**不插入** `archive_attachments` 时仍可插入 `archive_upload_failures`。  
- 推荐在 **返回错误 HTTP 响应之前** 同一请求内 **best-effort 提交**失败行；若写失败日志仍记 `error`，**不**掩盖原始上传错误。  
- **resolved**：用户确认已用 curl 补传成功后，可手点「已处理」；**不**自动与「新附件出现」强绑定（简化 MVP）。

---

## 五、任务拆分与验收

| # | 任务 | 验收标准 |
|---|------|----------|
| 1 | 配置 + `uploads.enabled` + 目录初始化 | `test_config` 等 |
| 2 | Repository：`add_attachment`（或等价） | 单测/仓储测 |
| 3 | `POST upload` + `GET file` + **§4.4** | `TestClient` 流式字节 + `Content-Disposition`；404 与现网 archive 一致 |
| 4 | 路径安全 | 表驱动 pytest；413 |
| 5 | `uploads.disabled` | POST 503；GET 行为与 **§4.1** 一致且测试固定 |
| 6 | Web：下载 + 可选上传入口 | `aria-label`、`lint-js`、**非手测唯一依据** |
| 7 | L2：`storage`、`download_api_path` | `_build_l2_dict`/快照；Lite 与 Full 若测 JSON 则同步 |
| 8 | 文档 | 根 `README.md`、`docs/design-docs/ops/mcp-server.md` 或 `web-server.md` |
| 9 | **§4.11** 迁移 + 失败写入 + `GET`（+ 可选 `PATCH`） | 权限同 L2；**鉴权失败路径 0 行** |
| 10 | Web 失败横幅 + 复制 curl | `lint-js`；可用 mock JSON 单测渲染/剪贴板逻辑（若有） |
| 11 | （可选）列表 `has_upload_failure` | Phase 2 |

**DoD**：`make verify`；运维文档含卷与 **DB+磁盘** 备份。

---

## 六、功能验证

### 6.1 自动化（门禁）

- `make verify`（`ruff`、`pytest`、`lint-js` 以 Makefile/CI 为准）。
- 最小用例集合建议固定为：  
  - **上传成功**：`POST` → DB 有 `archive_attachments`、磁盘有对应文件、`GET file` 字节与 Content-Type/Disposition 预期一致。  
  - **权限**：无权用户 `POST`/`GET file`/`GET upload-failures` **404**（或项目统一语义）。  
  - **413 / 503**：超限与 `uploads.enabled=false`。  
  - **路径**：越狱路径被拒绝。  
  - **失败记录**：制造业务失败 → `archive_upload_failures` 增 1；**鉴权前失败** → 表无新增。

### 6.2 手工验收清单（预发/本地）

| 步骤 | 操作 | 预期 |
|------|------|------|
| H1 | 创建档案（MCP `memory_save` scope=archive 或现有流程） | 存在 `archive_id` |
| H2 | Web 登录，打开档案馆 → 该档案详情 | 页可访问 |
| H3 | 上传允许扩展名的小文件 | 附件列表出现；下载内容与本地一致 |
| H4 | 故意触发失败（如关停 `uploads.enabled`、超限、非法扩展名） | 详情出现 **失败横幅**；展开有 **curl**，`archive_id` 正确；复制后在 shell 填入 Key 与本地路径可 **200** 成功 |
| H5 | （若实现 PATCH）点「已处理」 | `resolved_at` 非空；默认列表可筛掉 |
| H6 | Helm/多副本环境（若使用） | 与 **§4.10** 一致：要么单副本+卷，要么 RWX |

### 6.3 回归已有能力

- 档案馆列表/详情 **无上传功能时** 仍能浏览历史档案；`memory_get_archive` JSON **向后兼容**（旧客户端忽略新字段）。

---

## 七、风险与后续（Phase 2）

- **备份/RPO**：仅恢复 DB 或仅恢复盘均可能导致断链。  
- **多副本 + 本地盘**：见 **§4.10**。  
- **列表角标** `has_upload_failure`、**零档案 + 首文件**单页、速率限制、magic byte、上传进度、对象存储。

---

## 八、引用

- `src/team_memory/storage/models.py` — `Archive`, `ArchiveAttachment`；**新增** `ArchiveUploadFailure`（实现时）。
- `src/team_memory/web/routes/archives.py`（及将实现的上传/失败路由）。
- `helm/team-memory/`。

---

## 九、评审结论摘要

| 维度 | 结论 |
|------|------|
| **健康度** | **§4.4 / §4.5 / §4.6 / §4.11–§4.12** 冻结后可开工。 |
| **冲突消解** | §1.1 的 Web 收窄指向 **§4.7**；失败记录仅在 **过 L2  visibility 后** 写入，与 404 语义一致（**§4.12**）。 |
| **QA** | **§六.1** 为门禁；禁止仅以手工代替自动化。 |
| **产品** | 冷启动在 **§二**；Agent + 人工 curl 在 **§4.11**。 |

---

## 附录：文档内一致性巡检（自洽检查表）

| 检查项 | 状态 |
|--------|------|
| 上传入口仅在档案存在之后 | §二、§三 与 §4.4 一致 |
| `uploads.disabled` 仅影响 POST（推荐） | §4.1 与 §六 H4 可测 |
| 失败记录不误写在 404 鉴权路径 | §4.5、§4.11、§4.12、§六.1 |
| L2 与 MCP 附件 JSON 同形 | §4.6、§4.8 |
| curl 不含真 Key | §4.11 |
| Helm 与本地盘 | §4.10、§六 H6 |
