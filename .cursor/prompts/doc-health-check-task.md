# 文档健康巡检任务

**先理解规范，再执行校对**。按 `docs/design-docs/harness/doc-maintenance-guide.md` 与 `docs/design-docs/harness/plan-document-structure.md` 对项目文档进行健康度检查。

## 步骤

### 0. 前置：读取并理解规范

**必须**先读取以下两个文档的完整内容，理解其中的规则：

- `docs/design-docs/harness/doc-maintenance-guide.md`：同步约定、归档废弃、结构变更、Plan 生命周期、扫描规则（rule_id）、白名单
- `docs/design-docs/harness/plan-document-structure.md`：结构规范、合并原则、复盘必需保留清单、信息完整性检查清单、扫描规则（rule_id）

### 1. 技术文档健康度

- 执行 `make harness-doc-check`，解析脚本输出
- **基于 doc-maintenance-guide 理解**：对脚本未覆盖的规范（如同步约定、结构变更清单）做补充检查

### 2. Plan 文档结构

- 执行 `make harness-plan-check`，解析脚本输出
- **基于 plan-document-structure 理解**：对脚本未覆盖的规范（如合并原则、复盘必需保留清单、信息完整性检查清单）做补充检查

### 3. 输出报告

- 标题：文档健康巡检报告（日期）
- **脚本检出**：技术文档、Plan 文档的 rule_id 违规
- **规范理解补充**：基于两文档第一章的额外发现（若有）
- 路径变更提醒、总结

## 约束

- 只读：不修改文件，仅输出报告与建议
- 白名单：脚本已应用，报告仅包含未豁免问题
