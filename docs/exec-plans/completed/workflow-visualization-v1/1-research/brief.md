# Cytoscape 工作流可视化 — 调研任务书

## 目标

在 Web 端提供工作流只读可视化，降低理解门槛（相比 YAML/MD 直观）。

## 范围

- 新增入口：顶部导航「工作流可视化」（后调整为设置-高级配置 Modal）
- 拖入 workflow 文件（单文件或文件夹）
- 解析 YAML 与 step 级 `$ref`，展示完整流程
- 节点+边图，Cytoscape 渲染，支持缩放、平移

## 不包含

编辑、保存、n8n/Dify 格式导入（后续可扩展）。

## 解析规范

以 `src/team_memory/workflow_oracle.py` 的 `_resolve_step_ref` 语义为准。前端实现需与之一致，并在此基础上扩展递归解析（循环检测、深度上限 10）。
