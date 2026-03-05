# TeamMemory 扩展功能

任务管理、Skills 管理、工作流等扩展能力。核心经验记忆见 [README](../README.md)。

## 任务管理

面向 Agent（AI 优先、人工辅助）的任务协作能力：

- **Kanban 看板**：五列流转（等待/计划/进行中/已完成/已取消），WIP 限制，任务卡片
- **任务组与归档**：100% 完成的任务组可归档，从主视图隐藏
- **AI 执行**：Web 生成 Prompt、MCP execute_task、resume_project 三种方式
- **任务依赖**：blocks/related/discovered_from 类型，`tm_ready` 查询就绪任务

### 任务相关 MCP 工具

| 工具 | 功能 | 输入 |
|------|------|------|
| `tm_task` | 任务 CRUD | action、title、description、project、status |
| `tm_task_claim` | 认领任务 | 任务 ID |
| `tm_ready` | 就绪任务 | project（可选） |
| `tm_message` | 任务消息 | 任务 ID |
| `tm_dependency` | 任务依赖 | 任务 ID、依赖类型 |
| `tm_workflow_next_step` | 工作流步骤门控 | workflow_id、task_id |
| `tm_preflight` | 任务预检 | 任务描述、当前文件 |

### 任务相关 MCP Prompts

| Prompt | 说明 |
|--------|------|
| `execute_task` | 执行任务工作流 |
| `resume_project` | 恢复项目上下文 |

## Skills 与规则管理

- **可安装规则/技能**：从本地或远程安装 Cursor 规则、Skills，按项目启用或禁用
- **使用统计**：MCP 工具调用、按用户/API Key 分组、技能与规则使用情况

### 相关 MCP 工具

| 工具 | 功能 | 输入 |
|------|------|------|
| `tm_skill_manage` | 管理可安装规则/技能 | action、path、project |
| `tm_track` | 上报外部使用 | tool_name、tool_type、api_key_name |
| `tm_analyze_patterns` | 提取用户指令模式 | 对话或指令列表 |

## Web 扩展页面

| 页面 | 功能 |
|------|------|
| 任务 Kanban | 五列看板、任务组、任务侧边面板、AI Prompt、消息线程 |
| 归档管理 | 归档任务组、设置页折叠、取消归档 |
| 使用统计 | MCP 工具调用、按用户/API Key 分组、技能与规则使用 |

## Git Hooks 与任务关联

执行 `make hooks-install` 安装 Git hooks，commit 含 `[TM-xxx]` 时自动更新对应任务状态。
