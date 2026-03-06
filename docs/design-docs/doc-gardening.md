# doc-gardening 设计文档

> 定期扫描文档健康度，检出 deprecated 引用、断裂链接、过时标注。纯 Harness 优先，输出可被脚本与 Agent 解析。

---

## 一、设计目标

| 目标 | 说明 |
|------|------|
| **可脚本化** | 输出格式与 harness_import_check 一致，可被 CI、Makefile、Agent 解析 |
| **范围可控** | 首版与 CI 统一为 docs/design-docs、docs/exec-plans，避免拖慢 CI |
| **白名单可配置** | archive/ 内互相引用豁免；白名单格式与复审时机在文档中声明 |
| **规则衔接** | 与 tm-doc-maintenance 规则衔接：扫描结果可驱动人工或 Agent 修复 |

---

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

---

## 三、扫描项与 rule_id

| rule_id | 说明 | 示例 |
|---------|------|------|
| `DOC_LINK_404` | 内部链接指向不存在文件 | `[foo](docs/design-docs/missing.md)` 指向的文件不存在 |
| `DOC_LINK_BROKEN` | 断裂路径（相对路径解析失败、锚点不存在等） | 路径拼写错误、锚点 `#section` 不存在 |
| `DOC_DEPRECATED_REF` | 新文档引用 deprecated/archive 内文档 | 非 archive 文档引用 `docs/exec-plans/archive/xxx.md` 或 `deprecated/` 下文档 |
| `DOC_STALE_MARKER` | 过时标注（可选） | 含「部分内容已过时」「已废弃」等标注，产出待办清单 |

**说明**：

- `DOC_LINK_404`、`DOC_LINK_BROKEN`：内部相对链接（以 `./`、`../` 或 `docs/` 开头）需解析为实际路径并校验文件存在。
- `DOC_DEPRECATED_REF`：deprecated/archive 内文档**不应**被新文档引用；archive 内互相引用由白名单豁免。
- `DOC_STALE_MARKER`：首版可先做简单关键词匹配，后续可扩展为正则或配置。

---

## 四、输出格式

与 `harness_import_check` 一致：

```
file:line: rule_id: message
```

**示例**：

```
docs/design-docs/foo.md:12: DOC_LINK_404: Link target does not exist: docs/design-docs/missing.md
docs/exec-plans/index.md:5: DOC_DEPRECATED_REF: Should not reference archived doc: docs/exec-plans/archive/old-plan.md
docs/design-docs/bar.md:3: DOC_STALE_MARKER: Contains stale marker: 部分内容已过时
```

**Exit code**：`0` = 无问题，非 `0` = 有问题（便于 CI 阻断）。

---

## 五、白名单规则

### 5.1 豁免场景

| 场景 | 豁免条件 | 说明 |
|------|----------|------|
| archive 内互相引用 | 引用方与被引用方均在 `docs/exec-plans/archive/` 或 `docs/design-docs/archive/` 下 | 归档文档之间引用可保留，不强制更新 |
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

### 5.3 复审时机

- **archive/deprecated 豁免**：每季度复审一次，确认 archive 内文档是否仍需互相引用；若已无价值，可移出或删除。
- **路径白名单**：每次新增白名单条目时，在 PR 或 commit 中注明原因；每季度清理已修复的条目。

---

## 六、与 tm-doc-maintenance 规则衔接

当项目启用 **tm (team_memory)** 时，可能存在 `tm-doc-maintenance` 规则，负责：

- 结构变更联动（docs 目录变更时同步规则）
- 数据准确性、plan 归档
- deprecated 不引用

**衔接方式**：

| 衔接点 | 说明 |
|--------|------|
| **扫描结果驱动修复** | doc-gardening 输出可被人工或 Agent 解析，用于生成修复任务（如更新链接、移出 deprecated 引用） |
| **输出可被 Agent 解析** | `file:line: rule_id: message` 格式便于 Agent 定位并修复 |
| **白名单可被脚本读取** | 白名单文件路径固定，脚本按约定读取；tm-doc-maintenance 可引用本设计，约定「deprecated 不引用」由 doc-gardening 脚本执行 |
| **纯 Harness 优先** | 本设计不依赖 tm；若启用 tm，tm-doc-maintenance 可叠加调用 doc-gardening 作为子步骤 |

---

## 七、脚本引用说明

实现脚本（如 `scripts/harness_doc_gardening.py`）应：

1. 读取本设计文档或等价配置，获取扫描范围、rule_id、白名单路径；
2. 解析 `docs/design-docs`、`docs/exec-plans` 下 `.md` 的链接；
3. 按第三节规则校验；
4. 应用第五节白名单豁免；
5. 按第四节格式输出，exit 0/非 0 表示通过/失败。

**Golden Set**：在 `tests/fixtures/doc-gardening/expected.txt` 维护预期检出列表，用于自动化回归。

---

## 八、相关文档

| 文档 | 说明 |
|------|------|
| [harness-workflow-execution](harness-workflow-execution.md) | Plan 执行流程、摸底 |
| [harness-vs-tm-boundary](harness-vs-tm-boundary.md) | 纯 Harness 与 tm 叠加的分离说明 |
| [architecture-layers](architecture-layers.md) | harness_import_check 输出格式参考 |
