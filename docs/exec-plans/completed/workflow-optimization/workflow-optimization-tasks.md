# workflow-optimization

任务详细说明（目的、验收标准、是否可能已过时/可废弃）已写在 TM 每条任务的 description 中，便于判断保留或取消。本文仅作清单勾选用。

- [x] [一] step-complete 后显式禁止跳步（已做）
- [x] [一] 冷启动只在一个入口（仅 step-coldstart）
- [x] [一] group_completed 后禁止先选活
- [x] [一] 步骤审计格式固定（tm_message 含 step-X 前缀）
- [x] [一] step-verify 验收 subagent 推荐启动方式写清
- [x] [二] 新增「执行任务工作流」入口 Prompt（按任务匹配选择工作流）
- [ ] [二] 关键节点自动汇报进度
- [ ] [三] 步骤预言 MCP
- [ ] [三] 冷启动 description 由 LLM 生成模板
- [ ] [三] 验收入参结构化
- [ ] [四] step-verify 允许同会话内验收 checklist
- [ ] [四] 任务级验收 checklist 字段
- [ ] [四] 验收 checklist 模板
- [ ] [五] 组完成时强提示组复盘
- [ ] [五] 任务消息按 step 聚合/筛选
- [ ] [五] 任务看板按状态分列
- [ ] [五] 当前 step 与验收标准展示
- [ ] [五] 组/项目进度摘要
- [ ] [六] 任务完成时可选 Webhook
- [ ] [六] 步骤级完成率/耗时指标
- [ ] [七] 维护「断点恢复与组复盘」索引
- [ ] [八] 步骤预言 MCP 落地（基于 tm_message 解析下一步）
