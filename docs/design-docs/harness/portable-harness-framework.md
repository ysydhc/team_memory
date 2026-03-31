# Portable Harness Framework 设计文档

> **状态**：Draft — 设计讨论中，尚未实施
> **日期**：2026-03-31
> **背景**：将当前 team_memory 项目的 Harness 体系重构为可跨平台、跨项目移植的通用框架

---

## 1. 设计动机

当前 Harness 体系存在以下结构性问题：

1. **平台绑定**：编排逻辑散布在 `.claude/rules/` 和 `.cursor/rules/` 中，无法移植
2. **过度文档化**：用 exec-plans 三级目录管理任务状态，维护成本高于收益
3. **缺少强制执行**：质量门禁靠文字提醒，无自动化保障
4. **错误处理缺失**：没有失败分类和恢复策略
5. **契约缺失**：Plan/Task 启动无标准化检查清单
6. **项目耦合**：Harness 规则与 team_memory 业务逻辑混在一起
7. **无 Context 管理**：长任务 context rot 导致 Agent 质量下降，缺少拆分和刷新机制

## 2. 设计原则

- **关注点分离**：Harness Core（通用）vs Runtime Adapter（平台）vs Project Extension（项目）
- **能强制不建议**：Hook 自动执行 > 文字提醒（Mechanical Enforcement 共识）
- **文件即状态**：用 `progress.md` 替代复杂的文档目录管理
- **最小适配**：平台间共享尽可能多的配置，只在必须时双写
- **Context 保鲜**：长任务拆分，子任务独立 context window，防止 context rot
- **接口即 Harness**：工具接口设计 > 提示词工程（ACI 原则）

## 3. 三层架构

```
┌─────────────────────────────────────────────────────────┐
│                    Project Extension                     │
│         harness-config.yaml 中的项目特有配置              │
│    (质量门禁命令、分层规则、MCP 工具规范、业务约束)         │
├─────────────────────────────────────────────────────────┤
│                    Runtime Adapter                        │
│              平台特有的翻译 / 配置映射                     │
│                                                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │ Claude Code   │  │   Cursor     │  │  Future...   │  │
│   │ .claude/      │  │  .cursor/    │  │              │  │
│   │ settings.json │  │  hooks.json  │  │              │  │
│   │ rules/*.md    │  │  rules/*.mdc │  │              │  │
│   │ CLAUDE.md     │  │  AGENTS.md   │  │              │  │
│   └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│                     Harness Core                         │
│              .harness/ — 跨平台、可移植                   │
│                                                          │
│   编排流程 · 契约模板 · 失败分类 · Hook 脚本 · 计划管理    │
│   Context 管理 · 超时策略 · 行为防护                      │
└─────────────────────────────────────────────────────────┘
```

### 3.1 Harness Core（`.harness/`）

平台无关的核心逻辑，迁移新项目时整体复制。

### 3.2 Runtime Adapter（`.claude/` / `.cursor/`）

将 Harness Core 的通用定义翻译为平台能理解的格式。

### 3.3 Project Extension（`harness-config.yaml`）

项目特有的质量门禁、分层约束、归档策略等，通过配置而非硬编码。

## 4. 平台兼容性调研结论

> 基于 2026-03-31 对 Cursor 官方文档的调研

### 4.1 Cursor 对 Claude Code 的兼容情况

| Claude Code 功能 | Cursor 支持 | 机制 |
|---|---|---|
| **Hooks** (Pre/PostToolUse, Stop) | ✅ 完全兼容 | 自动读取 `.claude/settings.json`，自动映射事件名 |
| **Skills** (`.claude/skills/`) | ✅ 兼容 | 自动从 `.claude/skills/` 加载 |
| **Agents** (`.claude/agents/`) | ✅ 兼容 | 自动加载，`.cursor/agents/` 优先 |
| **CLAUDE.md** | ❌ 不读取 | Cursor 只读 `AGENTS.md` + `.cursor/rules/` |
| **`.claude/rules/`** | ❌ 不读取 | Cursor 只读 `.cursor/rules/` |
| **Permissions** (allow/deny) | ❌ 不支持 | Cursor 有独立权限模型 |
| **MCP 配置** | ⚠️ 各自独立 | `.mcp.json` vs `.cursor/mcp.json`，格式兼容 |
| **Memory** (auto memory) | ❌ 不支持 | Claude Code 特有 |
| **`@import` 语法** | ❌ 不支持 | Claude Code 特有 |

### 4.2 Cursor 独有 / 更强的功能

| 功能 | Claude Code | Cursor |
|---|---|---|
| Hook 事件数量 | 5 种 | 15+ 种（多了 subagentStart/Stop、beforeShellExecution 等） |
| Prompt-based Hooks | ❌ | ✅（用 LLM 判断是否放行） |
| Hook 配置层级 | project + user | enterprise → team → project → user → third-party |
| Cloud Agent | ❌ | ✅ |

### 4.3 共享 vs 需适配

```
完全共享（放 .claude/ 即可，Cursor 自动读取）：
├── Hook 脚本           → Cursor third-party hooks 兼容
├── Skills 定义         → .claude/skills/ 两边都读
├── Agents 定义         → .claude/agents/ 两边都读
└── MCP Server 代码     → 相同

需要双写 / 适配：
├── CLAUDE.md vs AGENTS.md     → 入口指令文件
├── .claude/rules/ vs .cursor/rules/ → 规则目录
├── MCP 配置路径               → .mcp.json vs .cursor/mcp.json
└── Agent frontmatter          → 字段定义有差异
```

### 4.4 可移植性设计

**移植到新项目的操作**：

| 层级 | 移植方式 | 工作量 |
|------|---------|--------|
| `.harness/`（Core） | 整体复制 | 零修改 |
| `.harness/harness-config.yaml` | 修改项目配置 | 改质量门禁命令、分层规则 |
| `.claude/hooks/` | 复制（或由 settings.json 引用 .harness/hooks/） | 零修改 |
| `.claude/settings.json` | 复制 | 调整 permissions 白名单 |
| `.claude/agents/` `.claude/skills/` | 复制通用的，删项目特有的 | 少量调整 |
| `.claude/rules/` `.cursor/rules/` | 复制通用 harness 规则，重写项目特有规则 | 中等 |
| `CLAUDE.md` / `AGENTS.md` | 重写项目说明部分 | 必须重写 |

**关键设计决策**：Hook 脚本放 `.harness/hooks/`，`.claude/settings.json` 通过相对路径引用。这样移植时 Hook 逻辑随 Core 一起走，适配层只需改引用路径。

**与 Trellis 对比**：

| 维度 | Trellis (`.trellis/`) | 我们 (`.harness/`) |
|------|----------------------|-------------------|
| 跨平台覆盖 | 10+ 平台 | Claude Code + Cursor（可扩展） |
| 编排深度 | spec + tasks，偏轻量 | Phase 0-4 + 契约 + 失败分类，更完整 |
| 状态管理 | tasks/ + memory/ + journals/ | progress.md + context 管理 |
| Hook 管理 | 无内置 | 内置 Hook 脚本 + 超时 + 安全防护 |
| 移植方式 | 复制 `.trellis/` | 复制 `.harness/` + 适配层骨架 |
| 平台适配 | 依赖 Agent 被指示去读 `.trellis/` | 原生目录 + 引用，无需额外指示 |

我们的优势在于**编排深度**和 **Hook 强制执行**；Trellis 的优势在于**平台覆盖广度**（10+ 平台经验验证）。两者的 `.harness/` vs `.trellis/` 模式本质相同：一个平台无关的核心目录。

## 5. 任务编排流程

```
Phase 0: Context         Phase 1: Planning       Phase 2: Execution
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ 加载 memory  │───────▶│ 生成 Plan    │───────▶│ 逐步执行     │
│ 读取 rules   │        │ 签署契约     │        │ 更新 progress│
│ 识别平台     │        │ → progress.md│        │ Hook 自动检查│
│ context 评估 │        │ 拆分 wave    │        │ context 监控 │
└──────────────┘        └──────┬───────┘        └──────┬───────┘
                               │                       │
                        ┌──────▼───────┐        ┌──────▼───────┐
                        │ 人工确认?    │        │ 失败恢复?    │
                        │ (scope>阈值) │        │ (3 级策略)   │
                        └──────────────┘        └──────────────┘

Phase 3: Verification   Phase 4: Closure
┌──────────────┐        ┌──────────────┐
│ 质量门禁     │───────▶│ 更新 progress│
│ lint/test/   │        │ 归档 plan    │
│ harness-check│        │ memory_save  │
└──────────────┘        └──────────────┘
```

### 5.1 编排扩展机制

核心流程定义在 `.harness/orchestration/task-flow.md`，每个 Phase 结尾包含扩展点：

```markdown
> 执行本阶段后，检查 `extensions/after-phase-{N}.md`，若存在则执行其中指令。
```

**扩展能力边界**：

| 操作 | 可靠性 | 说明 |
|------|--------|------|
| 新增步骤（additive） | ✅ 高 | 在 `extensions/` 放新文件即可 |
| 覆盖步骤（override） | ⚠️ 中 | 可能与原始指令冲突，需谨慎 |
| 条件分支 | ❌ 低 | 应放 `harness-config.yaml` 用配置控制 |

### 5.2 progress.md — 轻量状态文件

替代 exec-plans 三级目录结构，单文件追踪所有活跃计划：

```markdown
# Active Plans

## [Plan Name]
- **status**: in_progress | blocked | done
- **phase**: 0-4
- **current_task**: 当前执行的任务描述
- **blockers**: （如有）
- **started**: 2026-03-31
- **updated**: 2026-03-31
```

归档策略通过 `harness-config.yaml` 配置：

```yaml
plans:
  archive_completed: true
  archive_path: .harness/plans/completed/
```

### 5.3 Context 管理策略（借鉴 GSD）

> 来源：[GSD](https://github.com/gsd-build/get-shit-done) — Context rot 是长任务质量下降的主因

**核心问题**：Agent 在单一 context window 中累积过多历史后，输出质量持续下降（"context rot"）。

**解法 — 三层 context 管理**：

#### Layer 1：Context 监控与预警

| Context 使用率 | 状态 | 动作 |
|---|---|---|
| < 65% | 正常 | 继续执行 |
| 65%-80% | WARNING | 提醒尽快完成当前任务，准备保存状态 |
| > 80% | CRITICAL | 立即保存状态到 progress.md，完成当前步骤后停止 |

#### Layer 2：任务拆分原则

在 Phase 1（Planning）中判断是否需要拆分：

```
触发拆分的条件（满足任一）：
├── 预估涉及文件 > 5 个
├── 涉及多个子系统 / 模块
├── 预估 context 消耗 > 50%
└── 任务包含独立的并行子任务
```

拆分后的子任务遵循 **wave-based 并行**模型：

```
Wave 1（并行）：无依赖的子任务同时执行
  ├── Task A: 修改 models
  └── Task B: 修改 schemas

Wave 2（依赖 Wave 1）：
  └── Task C: 修改 services（依赖 A、B 的产出）

Wave 3（依赖 Wave 2）：
  └── Task D: 修改 API routes
```

每个 wave 内的子任务可以用**独立 subagent**（fresh context window）执行。

#### Layer 3：Subagent Context 隔离

当使用 subagent 执行子任务时，每个 subagent 只加载：
- 当前子任务的 Plan 描述
- progress.md（当前状态）
- 项目入口指令（CLAUDE.md / AGENTS.md）
- 必要的代码文件（通过 `read_first` 清单指定）

**不加载**：主 session 的对话历史、其他子任务的执行记录。

#### Layer 4：Analysis Paralysis 防护（借鉴 ECC/GSD）

如果 Agent 连续执行 5+ 次读/搜索操作而未产出任何写入，触发防护：

```
检测条件：连续 5 次 Read/Grep/Glob 无 Edit/Write
动作：
  1. 输出当前已收集的信息摘要
  2. 提示 "请基于已有信息开始行动，或说明为何需要继续搜索"
  3. 若再连续 3 次仍无写入 → 建议停下询问用户
```

## 6. 契约模板

### 6.1 Plan Contract（计划启动前签署）

```markdown
## Plan Contract: {plan_name}

### Scope
- [ ] 明确的交付物清单
- [ ] 影响范围（哪些文件/模块）
- [ ] 预估 context 消耗（小/中/大）→ 判断是否需要 wave 拆分

### Verification
- [ ] 完成标准（可机器验证的）
- [ ] 质量门禁命令

### Constraints
- [ ] 不做什么（scope 边界）
- [ ] 依赖 / 前置条件

### Human Decision Points
- [ ] 需要人工确认的节点

### Context Plan
- [ ] read_first 清单（子任务启动时必须读取的文件）
- [ ] wave 拆分方案（如需要）
```

### 6.2 Task Contract（任务开始前检查）

```markdown
## Task: {task_name}

- **输入**：需要什么信息/文件
- **输出**：产出什么
- **验证**：如何确认完成
- **回退**：失败时如何恢复
- **超时**：预期执行时间（用于超时判断）
```

## 7. 失败分类与恢复

### 7.1 三级恢复策略

```
Level 1: 自动恢复（Hook 处理）
├── lint 报错       → PostToolUse Hook 自动 format
├── import 方向违规  → Hook 检测并提示修正
├── 测试编译失败    → 自动重试一次
└── 命令超时        → 按超时策略自动重试或跳过

Level 2: Agent 判断恢复
├── 测试逻辑失败    → 分析失败原因，调整实现
├── 类型错误        → 读取上下文，修复类型
├── 依赖缺失        → 检查 requirements，安装
└── 偏差修复        → 自动修正 bug/缺失校验/阻塞依赖（最多 3 次）

Level 3: 人工介入
├── 架构决策冲突    → 停下询问
├── 外部服务不可用  → 报告并等待
├── scope 超出契约  → 提示需要新 Plan
└── 连续 3 次 Level 2 失败 → 升级
```

### 7.2 Agent 行为异常检测（借鉴 ECC/GSD）

除了代码执行错误，Agent 本身也可能出现行为异常：

| 异常类型 | 检测条件 | 恢复动作 |
|----------|---------|----------|
| **Observer Loop** | 连续 5+ 次 Read/Search 无 Write | 提示基于已有信息行动 |
| **重复操作** | 同一命令/编辑执行 3+ 次 | 停止并分析根因 |
| **Context 耗尽** | 使用率 > 80% | 保存状态，拆分子任务 |
| **无限回退** | 同一文件 undo/redo 3+ 次 | 回退到最近的 git commit，重新规划 |
| **偏离契约** | 编辑了 Plan Contract scope 外的文件 | 警告并确认是否扩展 scope |

### 7.3 错误分类维度

| 维度 | 分类 | 恢复策略 |
|------|------|----------|
| **可逆性** | 可逆 / 不可逆 | 不可逆操作前必须确认 |
| **自动化** | 可自动 / 需判断 / 需人工 | 对应 Level 1/2/3 |
| **影响范围** | 局部 / 跨模块 / 外部 | 范围越大，越倾向高 Level |
| **来源** | 代码错误 / 环境问题 / Agent 行为异常 | 行为异常需特殊处理 |

### 7.4 超时策略（借鉴 SWE-agent）

三级超时防止 Agent 或命令无限挂起：

```yaml
# harness-config.yaml
timeouts:
  command: 30        # 单条命令超时（秒），如 ruff、grep
  install: 300       # 安装/构建类命令超时（秒），如 pip install、make build
  task: 1800         # 单个 Task 总超时（秒），超时后保存状态并报告
```

超时后的恢复：
- **command 超时** → Level 1：自动 kill，记录警告，继续下一步
- **install 超时** → Level 2：Agent 判断是否换源/换方案
- **task 超时** → Level 3：保存 progress，提示人工介入

### 7.5 安全分级门控（借鉴 OpenHands）

命令执行前按风险分级：

| 风险等级 | 示例 | 处理 |
|----------|------|------|
| **Low** | `ruff check`, `pytest`, `git status` | 直接执行 |
| **Medium** | `git commit`, `pip install`, `alembic upgrade` | 记录日志 |
| **High** | `rm -rf`, `git push --force`, `DROP DATABASE` | Hook 阻止（exit 2） |
| **Interactive** | `vim`, `nano`, `python -i`, `psql` | Hook 阻止 + 提示替代方案 |

## 8. 目录结构

```
project-root/
│
├── .harness/                           # Harness Core（跨平台、可移植）
│   ├── harness-config.yaml             #   项目级配置（质量门禁、超时、归档等）
│   ├── orchestration/                  #   编排定义
│   │   ├── task-flow.md                #   核心流程 Phase 0-4
│   │   ├── context-management.md       #   Context 管理策略
│   │   ├── contracts/                  #   契约模板
│   │   │   ├── plan-contract.md
│   │   │   └── task-contract.md
│   │   └── extensions/                 #   扩展点（不改原文件加步骤）
│   │       └── README.md
│   ├── failure/                        #   失败分类与恢复策略
│   │   └── failure-taxonomy.md         #   含三级恢复 + 行为异常 + 超时 + 安全分级
│   ├── hooks/                          #   共享 Hook 脚本（两个平台引用同一份）
│   │   ├── pre-bash-guard.sh           #   安全防护（危险命令 + 交互命令阻止）
│   │   ├── post-edit-format.sh         #   编辑后自动格式化
│   │   └── stop-check.sh              #   完成标准检查
│   ├── plans/                          #   计划管理
│   │   ├── progress.md                 #   当前活跃计划状态（< 100 行）
│   │   └── completed/                  #   归档（可配置关闭）
│   └── docs/                           #   Harness 自身文档（随框架移植）
│       ├── harness-spec.md             #   框架规范
│       └── migration-guide.md          #   迁移到新项目指南
│
├── .claude/                            # Claude Code 适配层
│   ├── settings.json                   #   Hooks 引用 .harness/hooks/ + Permissions
│   ├── rules/                          #   Claude Code 规则（引用 .harness/ 内容）
│   │   ├── harness.md                  #   → @import .harness/ 核心编排
│   │   └── {project-specific}.md       #   项目特有规则
│   ├── agents/                         #   Agent 定义（Cursor 也自动读取）
│   │   ├── plan-implementer.md
│   │   └── code-reviewer.md
│   └── skills/                         #   Skills（Cursor 也自动读取）
│       ├── harness-check/SKILL.md
│       └── save-experience/SKILL.md
│
├── .cursor/                            # Cursor 适配层（仅放 Cursor 独有需求）
│   ├── hooks.json                      #   可选：Cursor 若不开 third-party hooks 时使用
│   ├── rules/                          #   Cursor 规则（引用 .harness/ 内容）
│   │   └── harness.mdc
│   └── mcp.json                        #   MCP 配置（Cursor 路径要求）
│
├── CLAUDE.md                           #   Claude Code 入口（< 200 行）
├── AGENTS.md                           #   Cursor 入口（内容与 CLAUDE.md 基本一致）
├── .mcp.json                           #   Claude Code MCP 配置
│
└── docs/                               #   项目文档（非 Harness）
```

### 8.1 harness-config.yaml 完整示例

```yaml
# .harness/harness-config.yaml
version: 1

project:
  name: team-memory
  description: MCP-based team experience library

# 质量门禁
quality_gates:
  lint: "ruff check src/"
  format: "ruff format --check src/"
  test: "pytest tests/ -v"
  verify: "make verify"
  # 项目可自定义任意门禁命令

# 分层约束
layering:
  forbidden_imports:
    - from: "storage"
      to: "services"
      direction: "禁止 storage → services"
    - from: "services"
      to: "web"
      direction: "禁止 services → web/server"

# 计划管理
plans:
  archive_completed: true
  archive_path: .harness/plans/completed/
  progress_max_lines: 100      # progress.md 硬限制

# 超时策略
timeouts:
  command: 30         # 单条命令（秒）
  install: 300        # 安装/构建命令（秒）
  task: 1800          # 单个 Task 总时长（秒）

# Context 管理
context:
  warning_threshold: 65        # context 使用率警告阈值（%）
  critical_threshold: 80       # context 使用率危险阈值（%）
  max_consecutive_reads: 5     # 连续只读操作上限（触发 analysis paralysis 防护）
  split_file_threshold: 5      # 涉及文件数超过此值建议拆分

# 失败恢复
failure:
  auto_retry_limit: 1           # Level 1 自动重试次数
  escalate_after_failures: 3    # 连续失败 N 次后升级到人工
  max_deviation_fixes: 3        # Level 2 偏差自动修复最大次数

# Hook 脚本路径（供适配层引用）
hooks:
  pre_bash_guard: .harness/hooks/pre-bash-guard.sh
  post_edit_format: .harness/hooks/post-edit-format.sh
  stop_check: .harness/hooks/stop-check.sh

# 安全防护
security:
  blocked_commands:
    - "rm -rf /"
    - "sudo"
    - "curl|sh"
    - "DROP DATABASE"
    - "git push --force main"
    - "git push --force master"
  blocked_interactive:
    - "vim"
    - "nano"
    - "emacs"
    - "python -i"
    - "psql"
    - "mysql"

# 平台适配
adapters:
  claude_code:
    instructions_file: CLAUDE.md
    rules_dir: .claude/rules/
    settings: .claude/settings.json
  cursor:
    instructions_file: AGENTS.md
    rules_dir: .cursor/rules/
    hooks_config: .cursor/hooks.json
```

## 9. 迁移到新项目

```bash
# Step 1: 复制 Harness Core（零修改即可工作）
cp -r .harness/ /path/to/new-project/.harness/

# Step 2: 复制适配层骨架
cp .claude/settings.json /path/to/new-project/.claude/settings.json
cp -r .claude/agents/ /path/to/new-project/.claude/agents/
cp -r .claude/skills/ /path/to/new-project/.claude/skills/

# Step 3: 修改项目配置
#   .harness/harness-config.yaml → 改 project 信息、质量门禁命令、分层规则
#   CLAUDE.md / AGENTS.md → 重写项目说明部分
#   .claude/rules/ → 删除项目特有规则，保留通用 harness 规则
#   .claude/settings.json → 调整 permissions 白名单

# Step 4: 如需 Cursor 支持
cp -r .cursor/rules/ /path/to/new-project/.cursor/rules/
# 或开启 Cursor third-party hooks，直接读 .claude/ 配置
```

**移植检查清单**：

- [ ] `.harness/` 整体复制
- [ ] `harness-config.yaml` 中 project/quality_gates/layering 已改为新项目值
- [ ] CLAUDE.md / AGENTS.md 已重写项目说明
- [ ] `.claude/settings.json` permissions 适配新项目
- [ ] 项目特有的 rules 已清理
- [ ] `make verify`（或新项目等效命令）通过

## 10. 社区调研 & 借鉴

> 基于 2026-03-31 对 GitHub 生态的全面调研

### 10.1 直接借鉴

| 来源项目 | Stars | 借鉴内容 | 融入位置 |
|----------|-------|---------|---------|
| [GSD](https://github.com/gsd-build/get-shit-done) | 45k | Context rot 管理：监控阈值 + 任务拆分 + subagent 隔离 + analysis paralysis 防护 | §5.3 Context 管理策略 |
| [SWE-agent](https://github.com/SWE-agent/SWE-agent) | 19k | 三级超时策略 + 交互命令阻止 + ACI 原则 | §7.4 超时策略, §7.5 安全分级 |
| [ECC](https://github.com/affaan-m/everything-claude-code) | 120k | Observer loop 防护 + 行为异常检测 | §7.2 Agent 行为异常检测 |
| [OpenHands](https://github.com/All-Hands-AI/OpenHands) | 70k | 两级错误分类（可恢复 vs 致命）+ 安全分级门控 | §7.1 + §7.5 |
| [Trellis](https://github.com/mindfold-ai/Trellis) | 4.4k | 统一目录跨平台模式验证（`.trellis/` ≈ `.harness/`） | §4.4 可移植性设计 |

### 10.2 参考但未直接借鉴

| 来源项目 | Stars | 可选借鉴 | 暂不引入原因 |
|----------|-------|---------|-------------|
| [Aider](https://github.com/Aider-AI/aider) | 43k | Architect/Editor 双模型分离 | 需要平台支持模型路由，当前 Claude Code/Cursor 不支持任务级模型切换 |
| [Cline](https://github.com/cline/cline) | 60k | Step-level checkpoint + 动态 MCP 创建 | Checkpoint 依赖 VS Code 扩展，非文件级可移植 |
| [rtk](https://github.com/rtk-ai/rtk) | 16k | 工具输出 token 压缩（60-90%） | 外部依赖，可作为可选插件后续引入 |
| [Superpowers](https://github.com/obra/superpowers) | 126k | 大规模 Skills 库 | 已有 Skills 机制，按需从社区安装 |
| [BMAD-METHOD](https://github.com/bmadcode/BMAD-METHOD) | 43k | 12+ Agent 角色定义 | 角色过多增加复杂度，保持精简 |

### 10.3 Mechanical Enforcement 共识

多个项目（harness-engineering、SWE-agent、ECC、我们）独立得出相同结论：

> **Agent 不会可靠地遵循文字指令。能用代码强制的，不要用 prompt 建议。**

优先级：Hook 自动执行 > Permission deny > Rule 文件 > CLAUDE.md 文字提醒

## 11. 理论依据

### 11.1 NLAH 论文（清华 / 哈工大，arXiv 2603.25723）

- **File-Backed State** 是最有效的 Harness 模块（OSWorld +5.5%）→ progress.md
- **6 核心组件**：Contracts, Roles, Stage Structure, Adapters & Scripts, State Semantics, Failure Taxonomy → 全部对应
- **IHR 模式**（In-loop LLM + Runtime）→ Agent 判断恢复（Level 2）

### 11.2 Anthropic 官方实践

- **CLAUDE.md < 200 行**，advisory 不如 Hook 强制 → Hook 体系
- **Context 是最重要的资源**，最小化高信号 token → progress.md + context 管理
- **Subagent 隔离**，只读 agent 不给写权限 → code-reviewer readonly

### 11.3 Claude Code + Cursor 兼容性调研

- Hooks/Skills/Agents 可共享，Rules/入口文件需双写 → 最小适配策略
- Hook 脚本放 `.harness/hooks/` 统一管理 → 两个平台引用同一份

## 12. 待办

- [x] 调研 GitHub 上已有的 Harness 框架设计，吸收最佳实践
- [ ] 细化 `task-flow.md` 编排内容
- [ ] 细化 `context-management.md` 内容
- [ ] 细化 `failure-taxonomy.md` 内容
- [ ] 实施：创建 `.harness/` 目录结构
- [ ] 实施：迁移现有 Hook 脚本到 `.harness/hooks/`
- [ ] 实施：更新 `.claude/settings.json` 引用路径
- [ ] 实施：增强 `pre-bash-guard.sh`（交互命令阻止 + 安全分级）
- [ ] 实施：创建 `AGENTS.md`
- [ ] 实施：精简现有 Harness 设计文档（合并冗余）

## 13. 开放问题

1. **Agent frontmatter 交集**：Claude Code 和 Cursor 的 agent 字段不完全相同（Claude Code: tools/disallowedTools/permissionMode/maxTurns/hooks/skills/isolation; Cursor: name/description/model/readonly/is_background），共享 agent 定义应取哪些字段的交集？
2. **Rules 同步**：`.claude/rules/` 和 `.cursor/rules/` 内容基本一致但格式不同（`.md` vs `.mdc`），是否需要同步脚本（`make harness-sync`）？
3. **Context 监控实现**：GSD 通过 statusline hook 写入 `/tmp/claude-ctx-{session_id}.json` 实现 context 监控，我们是否采用相同机制，还是依赖平台内置的 context 指示？
4. **Wave 并行调度**：当前 Claude Code 的 subagent 通过 `Agent` tool + `isolation: "worktree"` 支持并行，Cursor 通过 `is_background` 支持，是否在 `task-flow.md` 中统一抽象？
