# 文档维护规范

> **目的**：防止文档腐化，保持准确性、时效性与可维护性。  
> **Agent 读取**：规范比照 → 第一章；扫描规则 → 第二章。

---

# 第一章：规范

## 一、同步约定

1. **功能或代码变更后**，须同步更新 README 与 docs/design-docs 对应文档（若有）；文档不同步不得视为任务完成。
2. **根 README**：面向用户与贡献者；涉及具体运维步骤或设计细节时，链接到 `docs/design-docs/` 下对应文档。
3. **docs/design-docs/**：设计文档目录；新增或修改时，在 `docs/design-docs/README.md` 目录中登记或更新链接。

## 二、技术术语

- 统一使用 `tm_search`、`experience_type`、`min_similarity` 等
- 避免中英混用歧义

## 三、API 与配置文档

- 描述 API 时使用 `/api/v1/` 路径（旧路径 `/api/` 可兼容，但文档以 v1 为准）
- 配置项与 `config.development.yaml` / `config.production.yaml` 对应段保持一致；变更 config 时同步更新 docs

## 四、归档与废弃

- 过时、已合并或不再引用的文档移入 `docs/exec-plans/completed/{主题}/`
- **archive 目录已废弃**：不再使用 `docs/exec-plans/archive/` 或 `docs/design-docs/archive/`（根级 archive 已废弃）
- **completed/archive/**：历史归档子目录，仍保留；其内文档互相引用可豁免（白名单）；新文档不得引用该路径下文件（扫描规则见第二章 DOC_DEPRECATED_REF）
- 移入时在目标目录中简要记录

## 五、结构变更检查清单

路径或目录重组时，**必须**同步更新以下位置：

- [ ] `docs/design-docs/README.md` 中的索引
- [ ] `AGENTS.md` 知识库导航链接

## 六、Plan 生命周期

| 状态 | 位置 |
|------|------|
| **等待执行** | `docs/exec-plans/wait/{主题}/` |
| **执行中** | `docs/exec-plans/executing/{主题}/` |
| **已完成** | `docs/exec-plans/completed/{主题}/` |

实现完成的 plan 移入 `completed/`；不确定是否已实现的保留在 `wait/` 或根级。Plan 文档结构规范（调研、计划、复盘、扫描规则）详见 [plan-document-structure](plan-document-structure.md)。

**Plan 整理时的信息保留**：doc-admin-organize 整理 Plan 类文档时，须按 [plan-document-structure](plan-document-structure.md) 第五节（合并时的信息保留原则）、第四节（调研阶段合并原则）执行，逐项核对复盘必需保留清单与 assessment 合并要求。

## 七、治理周期建议

- **季度**：运行 `make harness-doc-check` 扫描链接有效性，修正失效链接（扫描规则见第二章）；复审白名单（completed/archive 内互相引用、路径白名单），清理已修复条目
- **半年**：评估 design-docs、exec-plans 是否有新冗余，按治理流程合并或归档
- **功能大改**：对照 config、server 校验 API/配置描述

---

# 第二章：扫描设计

> 定期扫描文档健康度，检出 deprecated 引用、断裂链接、过时标注。输出可被脚本与 Agent 解析。

## 一、设计目标

| 目标 | 说明 |
|------|------|
| **可脚本化** | 输出格式与 harness_import_check 一致，可被 CI、Makefile、Agent 解析 |
| **范围可控** | 首版与 CI 统一为 docs/design-docs、docs/exec-plans，避免拖慢 CI |
| **白名单可配置** | completed/archive、deprecated 内互相引用豁免；白名单格式与复审时机见第一章第七节 |
| **规则衔接** | 扫描结果可驱动人工或 Agent 修复 |

## 二、扫描范围

### 2.1 首版与 CI 统一范围

| 范围 | 首版 | CI | 说明 |
|------|------|-----|------|
| `docs/design-docs` | ✅ | ✅ | 设计文档、架构、方案 |
| `docs/exec-plans` | ✅ | ✅ | 执行计划、归档、executing |
| `README.md`、`AGENTS.md` | 可选 | ❌ | 由本地 `make harness-doc-check` 或定期任务扫描 |
| `.cursor/` | ❌ | ❌ | 若被 git 追踪可后续扩展 |

**CI 策略**：只跑 `docs/design-docs`、`docs/exec-plans`，避免拖慢流水线。README/AGENTS 由本地或定期任务扫描。

### 2.2 文件类型

- 扫描 `.md` 文件
- 解析 Markdown 链接：`[text](url)`、`<url>`

## 三、扫描项与 rule_id

| rule_id | 说明 | 示例 |
|---------|------|------|
| `DOC_LINK_404` | 内部链接指向不存在文件 | `[foo](docs/design-docs/missing.md)` 指向的文件不存在 |
| `DOC_LINK_BROKEN` | 断裂路径（相对路径解析失败、锚点不存在等） | 路径拼写错误、锚点 `#section` 不存在 |
| `DOC_DEPRECATED_REF` | 新文档引用 deprecated/archive 内文档 | 非 archive 文档引用 `docs/exec-plans/completed/archive/xxx.md` 或 `deprecated/` 下文档 |
| `DOC_STALE_MARKER` | 过时标注（可选） | 含「部分内容已过时」「已废弃」等标注，产出待办清单 |

**说明**：

- `DOC_LINK_404`、`DOC_LINK_BROKEN`：内部相对链接（以 `./`、`../` 或 `docs/` 开头）需解析为实际路径并校验文件存在。
- `DOC_DEPRECATED_REF`：deprecated/archive 内文档**不应**被新文档引用；completed/archive、deprecated 内互相引用由白名单豁免。
- `DOC_STALE_MARKER`：首版可先做简单关键词匹配，后续可扩展为正则或配置。

## 四、输出格式

与 `harness_import_check` 一致：

```
file:line: rule_id: message
```

**示例**：

```
docs/design-docs/foo.md:12: DOC_LINK_404: Link target does not exist: docs/design-docs/missing.md
docs/exec-plans/index.md:5: DOC_DEPRECATED_REF: Should not reference archived doc: docs/exec-plans/completed/archive/old-plan.md
docs/design-docs/bar.md:3: DOC_STALE_MARKER: Contains stale marker: 部分内容已过时
```

**Exit code**：`0` = 无问题，非 `0` = 有问题（便于 CI 阻断）。

## 五、白名单规则

### 5.1 豁免场景

| 场景 | 豁免条件 | 说明 |
|------|----------|------|
| completed/archive 内互相引用 | 引用方与被引用方均在 `docs/exec-plans/completed/archive/` 下 | 历史归档文档之间引用可保留，不强制更新 |
| deprecated 内互相引用 | 引用方与被引用方均在 `deprecated/` 下 | 同上 |
| 路径白名单 | 在配置中显式列出的 `file:line` 或 `file:rule_id` | 用于临时豁免误报 |

### 5.2 白名单格式

**路径白名单**（供脚本读取）：

```
# 每行一个条目，支持：
# 1. 完整豁免：path/to/file.md
# 2. 行级豁免：path/to/file.md:42
# 3. 规则豁免：path/to/file.md:DOC_STALE_MARKER
```

**建议路径**：`scripts/doc-gardening-whitelist.txt` 或与脚本同目录的 `doc-gardening-whitelist.txt`。若文件不存在，则无路径白名单。

## 六、脚本引用说明

实现脚本（如 `scripts/harness_doc_gardening.py`）应：

1. 读取本设计文档或等价配置，获取扫描范围、rule_id、白名单路径；
2. 解析 `docs/design-docs`、`docs/exec-plans` 下 `.md` 的链接；
3. 按第三节规则校验；
4. 应用第五节白名单豁免；
5. 按第四节格式输出，exit 0/非 0 表示通过/失败。

**Golden Set**：在 `tests/fixtures/doc-gardening/expected.txt` 维护预期检出列表，用于自动化回归。

## 七、相关文档

| 文档 | 说明 |
|------|------|
| [harness-spec](harness-spec.md) | Plan 执行流程、摸底、反馈回路 |
| [project-extension](project-extension.md) | 架构分层、文档迁移收尾清单、harness_import_check 参考 |
| [doc-admin-organize](../../../.cursor/agents/doc-admin-organize.md) | 文档整理 Agent（按规范实际整理） |
