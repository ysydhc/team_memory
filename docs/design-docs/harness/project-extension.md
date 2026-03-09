# team_doc 项目扩展

> 本项目的 Harness 扩展：架构分层、质量门禁、收尾清单、tm 边界。见 [harness-spec](harness-spec.md)。

---

## 一、架构分层（L0～L3）

| 层 | 模块 | 路径 | 可依赖 | 禁止依赖 |
|----|------|------|--------|----------|
| **L0** | schemas, config | `src/team_memory/schemas.py`, `config.py` | 无 | - |
| **L1** | storage | `src/team_memory/storage/` | L0 | L2, L3 |
| **L2** | services, auth, embedding, reranker, architecture | `src/team_memory/` 对应目录 | L0, L1 | web, server, bootstrap |
| **L3** | web, server, bootstrap, workflow_oracle | `src/team_memory/web/` 等 | L0, L1, L2 | - |

**依赖方向**：只能向前依赖，禁止反向。bootstrap、server 禁止被 L0–L2 引用。

**豁免**：`# noqa: layer-check` 单行豁免；`if TYPE_CHECKING:` 块豁免；白名单 `exclude_paths`。

**脚本**：`scripts/harness_import_check.py` 读取分层表与依赖矩阵，校验 `src/team_memory/**/*.py`。

---

## 二、质量门禁（harness-check）

```bash
python scripts/harness_import_check.py
ruff check src/
./scripts/harness_ref_verify.sh
```

等价于 `make harness-check`。CI 无 Makefile 时可直接调用。

---

## 三、文档迁移收尾清单

当 Plan 涉及文档/路径迁移时，收尾须按本清单勾选：

- [ ] `./scripts/harness_ref_scan.sh` 已执行，输出无异常
- [ ] `./scripts/harness_ref_verify.sh` 已执行，exit 0
- [ ] 无 404 或指向已移动路径的引用
- [ ] **Tool usage 基线**（可选，仅 tm 项目）：若 team_memory 服务可用，执行 `scripts/harness_tool_usage_baseline.sh`；否则标注「待服务就绪后补跑」或「非 tm 项目可跳过」
- [ ] 文档维护规则已同步 harness 文档结构
- [ ] README、AGENTS.md 等主入口已更新（见 [doc-maintenance-guide](doc-maintenance-guide.md) 第五章）
- [ ] `ruff check src/`、`pytest tests/ -q` 通过

---

## 四、tm (team_memory) 边界

### 纯 Harness 与 tm 叠加

| 环节 | 纯 Harness | tm 叠加（可选） |
|------|------------|-----------------|
| 反馈回路 | 更新 rules 或 docs | 可额外 tm_save / tm_learn |
| Subagent 审计 | 在回复中记录 `[subagent] task-N:` | 若与 tm_task 关联，可同时 tm_message |
| 收尾清单 | 引用校验、质量门禁；不含 tool_usage | tool_usage 作为 tm 项目可选 |
| 任务预检 | 无 | tm_preflight 触发经验检索 |

### tm 特有能力

| 能力 | 用途 | Harness 关系 |
|------|------|--------------|
| tm_task | 任务编排、workflow 门控 | 可选 |
| tm_message | 任务审计日志（存 DB） | 可选 |
| tm_save / tm_learn | 经验沉淀到经验库 | 可选 |
| harness_tool_usage_baseline.sh | 调用 team_memory API | tm 特有，依赖服务 |

**原则**：Harness 规则不依赖 tm_*；tm 规则可引用 Harness 原则，在启用 tm 时叠加能力。
