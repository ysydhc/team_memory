# TeamMemory

mcp-name: io.github.ysydhc/team-memory

**让 AI 拥有团队记忆 -- 跨会话积累经验，像资深成员一样理解你的项目。**

基于 MCP 协议的团队经验数据库。AI 在开发过程中自动提取、存储和检索团队的历史经验，解决 AI 编程助手"每次对话都从零开始"的核心痛点。

## AI 编程的三个盲区

当你用 Cursor、Claude Desktop 等 AI 助手参与大型项目开发时，AI 面临三个根本性限制：

| 盲区 | 现象 | 影响 |
|------|------|------|
| **无记忆** | 上周 AI 帮你解决了某个 Bug，这周遇到类似问题它完全不记得 | 重复排查，浪费时间和 Token |
| **只见代码，不懂决策** | AI 能读懂代码"是什么"，但不知道"为什么这样写"，更不知道"上次那样写踩了什么坑" | 可能重蹈覆辙，或推翻合理的历史决策 |
| **静态知识不够用** | Rules 定义规范，Skills 封装流程，但团队每天产生的大量经验（某个接口的隐藏坑、某次故障的根因、某个方案被否决的原因）无法被它们覆盖 | AI 缺乏团队的"隐性知识"，始终像个新人 |

**TeamMemory 解决的就是这三个问题。**

它通过 MCP 协议将一个语义可搜索的经验数据库接入 AI，让 AI 能够：
- 遇到问题时，自动查询团队是否曾经解决过类似问题
- 解决问题后，自动从对话中提取关键经验并存储
- 下次任何团队成员遇到同类问题，AI 直接命中历史方案

适用场景：3-10 人技术团队 + Cursor / Claude Desktop。

**安装与获取**

- **PyPI**：`pip install team_memory`（推荐用于部署或本地 MCP 客户端）。
- **MCP 官方注册表**：在 [MCP Registry](https://registry.modelcontextprotocol.io) 或 Cursor / Claude Desktop 的 MCP 市场中搜索「TeamMemory」或「team-memory」，可一键发现并安装（安装后仍需配置数据库连接与 API Key，见下文）。

## TeamMemory 在 AI 知识体系中的位置

AI 助手在项目中可用的知识分为三层，每一层解决不同的问题：

```
┌─────────────────────────────────────────────────────────┐
│  Rules / Skills（静态层）                                │
│  已确定的规范和成熟的工作流，手动维护，变化频率低            │
│  例：代码风格规范、部署流程、API 用法                      │
├─────────────────────────────────────────────────────────┤
│  TeamMemory（动态层）          ← 本项目                  │
│  从日常开发中自动积累的团队经验，语义可搜索，持续演化         │
│  例：Bug 根因分析、架构决策背景、接口的隐藏坑               │
├─────────────────────────────────────────────────────────┤
│  代码 + 文档（基础层）                                    │
│  项目源码和文档，AI 可直接读取                              │
│  例：函数定义、README、注释                                │
└─────────────────────────────────────────────────────────┘
```

三层之间存在自然的知识生命周期：

```
日常开发会话
    │
    ▼
AI 自动提取经验 ──→ TeamMemory（动态积累）
    │                      │
    │                      ▼ 模式稳定后
    │               固化为 Rules / Skills
    │                      │
    └──── 新问题 / 新变化 ←─┘
```

**Rules/Skills 无法覆盖的知识，正是 TeamMemory 的价值所在**：那些太碎、太多、变化太快，不适合写成规则，但恰恰是团队"老手"和"新手"之间最大差距的经验。

## 功能概览

### 自动学习

AI 从对话和文档中自动提取结构化经验，无需手动录入：

- **对话提取**（`tm_learn`）：LLM 自动从开发对话中识别问题、方案、标签，提取为结构化经验；prompt 含 few-shot 示例与质量门控（score&lt;2 重试）
- **文档解析**：上传文档或输入 URL，AI 自动解析为标题、问题描述、解决方案、标签
- **经验组**（`tm_save_group`）：将一次完整的问题解决过程（从发现到排查到修复）作为父子经验组保存；三阶段管道方案见 `.debug/10-extraction-pipeline.md`
- **默认草稿模式**：AI 提取的内容默认为草稿，经人工审核后发布，保证质量

### 智能检索

多层检索管线，确保 AI 找到最相关的历史方案：

- **语义搜索**：基于向量嵌入（Ollama / OpenAI / 本地模型），理解查询意图
- **混合检索**：向量搜索 + 全文检索 + RRF 融合排序
- **查询优化**：同义词扩展（`config.tag_synonyms`）、短查询自动降低 min_similarity（0.45）；FTS 使用 simple 分词器并支持 jieba 中文分词
- **Reranker**：支持服务端 LLM 精排，或客户端 AI 自行判断结果相关性
- **Token 预算控制**：自动裁剪输出长度，避免经验库增大后撑爆 AI 上下文
- **PageIndex-Lite**：长文档自动分块建立节点索引，支持节点级精准检索

### 任务管理

面向 Agent（AI 优先、人工辅助）的任务协作能力：

- **Kanban 看板**：五列流转（等待/计划/进行中/已完成/已取消），WIP 限制，任务卡片
- **任务组与归档**：100% 完成的任务组可归档，从主视图隐藏
- **AI 执行**：Web 生成 Prompt、MCP execute_task、resume_project 三种方式
- **任务依赖**：blocks/related/discovered_from 类型，`tm_ready` 查询就绪任务

### 三层作用域

- **global**：全局共享知识
- **team**：团队/项目级经验
- **personal**：个人笔记与草稿

### 经验类型自动分类

保存时可根据内容自动推荐经验类型（general/feature/bugfix/tech_design/incident/best_practice/learning），减少手动选择。

### 任务预检机制

创建任务或 Plan 前调用 `tm_preflight`，根据任务复杂度返回搜索深度建议（skip/light/full），避免重复工作。

### 经验质量打分系统

自动评估经验的活跃度和价值，让高质量经验脱颖而出：

- **阶梯衰减**：新经验 100 分起步，10 天保护期后未被引用每天 -1 分（低于 50 分后 -0.5/天）
- **引用加分**：每次被 `tm_search`/`tm_solve` 命中 +2 分，获 4 星以上评价 +1 分
- **质量等级**：Gold (≥120) / Silver (≥60) / Bronze (≥20) / Outdated (≤0)
- **置顶免衰**：手动置顶的经验永不衰减（年度发布流程等长期有效经验）
- **Outdated 管理**：分值归零的经验仍可搜索到，但在管理面板提示处理（恢复/删除/置顶）
- **规则可配**：初始分值、衰减速率、加分幅度、等级阈值均可在设置页调整

### 结构化管理

不是随意堆放的笔记，而是有类型、有层级、有评分的经验体系：

- **7 种经验类型**：通用、需求、Bug、技术方案、故障、最佳实践、学习笔记，每种类型有专属结构化字段
- **父子层级存储**：支持将相关经验组织为经验组（如：一个需求从评审到上线的全过程）
- **完整度评分**：0-100 分自动评分，鼓励团队逐步完善经验
- **生命周期管理**：草稿 → 审核 → 发布 → 过期检测 → 去重合并

### 团队协作

多人共建共享的团队知识库：

- **RBAC 权限**：admin / editor / viewer 三级角色
- **反馈评分**：1-5 星评分影响搜索排序，低分经验自动降权
- **版本历史**：就地编辑 + 版本快照，支持回滚
- **去重检测**：保存前自动检测相似经验，避免重复录入
- **合并建议**：设置页标记建议合并的相似经验对，三栏 Git-Conflict 风格 diff 对比 + 合并预览
- **多项目隔离**：通过 `project` 参数隔离不同项目的经验，避免跨项目污染

### 无缝集成

原生 MCP 协议支持，AI 助手零配置调用：

- **13 个 MCP 工具**：覆盖搜索、保存、反馈、协作全场景
- **3 个 MCP 资源**：最近经验、统计数据、过期经验
- **5 个 MCP Prompt**：对话摘要、文档提交、质量审核、故障排查、长文档分析
- **Web 管理界面**：浏览、搜索、审核、配置一站式管理

### 一键部署

最小化上手成本：

- `make setup` 一键完成全部安装
- 最小配置模板只需改 2 项即可启动
- Docker Compose 自动拉取所需模型
- 统一端口 9111，健康检查一键诊断

## 快速开始

以下从**初次部署者**（把 TeamMemory 跑起来的人）和**初级使用者**（在 Cursor/Claude 里连上已有服务的人）两种角色，按最小步骤说明。

### 前提条件

| 依赖 | 确认方式 |
|------|----------|
| Docker Desktop | `docker --version` |
| Python 3.11+ | `python3 --version` |
| Make | `make --version`（macOS/Linux 一般自带） |
| Ollama | `ollama --version`（首次 `make web` 前拉取模型用，见下） |

### 一、初次部署者：一键部署

你是第一次在团队里部署，目标是：跑起 Web、拿到 API Key、并交给同事用。

**1. 一键初始化**

克隆仓库后，在项目根目录执行：

```bash
make setup
```

会完成：启动 Docker（PostgreSQL+pgvector、Ollama、Redis）、安装 Python 依赖、执行数据库迁移。

**2. 改一处配置**

最简配置：

设置环境变量作为管理员引导密钥（首次部署必须）：

```bash
export TEAM_MEMORY_API_KEY=$(openssl rand -hex 16)
echo "你的管理员 API Key: $TEAM_MEMORY_API_KEY"
```

**首次登录**

- 打开 http://localhost:9111
- 点击「使用 API Key 登录」
- 将上面的 TEAM_MEMORY_API_KEY 粘贴到输入框，点击「登录」
- 登录成功后即为 admin，右上角显示当前用户名（默认来自 TEAM_MEMORY_USER，未设置时为 admin）

该 Key 仅在内存中生效（与 config 中的 auth.api_key 一样），重启服务后仍使用同一环境变量即可再次用该 Key 登录。若希望用「用户名 + 密码」登录并长期使用，见下文「可选：为自己创建持久 Admin 账户」。

### 方式二：配置文件

在 config.yaml 或 config.minimal.yaml 中设置：
```
auth:
  type: db_api_key
  api_key: "${TEAM_MEMORY_API_KEY}"   # 从环境变量读取
  user: admin
```

或直接写明文（仅限本地/测试，勿提交到 Git）：
```
auth:
  type: db_api_key
  api_key: "你的引导Key"
  user: admin
```
启动服务后，用该 api_key 在 Web 登录页选择「使用 API Key 登录」即可获得 admin。

### 可选：为自己创建持久 Admin 账户
若希望用「用户名 + 密码」登录、且不依赖引导 Key：
1. 已用引导 Key 以 admin 身份登录 Web。
2. 进入 设置 → 用户管理，点击「添加用户」。
3. 填写：
    - 用户名：如 admin 或你的英文名
    - 角色：admin
    - 初始密码：设置一个强密码（用于 Web 登录）
4. 点击「创建」。系统会生成并展示一条 API Key（用于 MCP 客户端），请复制保存。
5. 之后可：
    - Web：用刚设置的用户名 + 密码登录。
    - MCP：在 Cursor/Claude 的 mcp.json 中使用刚生成的 API Key 作为 TEAM_MEMORY_API_KEY。
引导 Key（环境变量或 config 中的 key）与数据库中的用户彼此独立：引导 Key 仅用于首次拿到 admin 权限；后续日常可使用数据库里的 admin 账号（用户名+密码 + 自己的 API Key）。

若使用默认 Docker 数据库，`database.url` 无需改；否则改为你的 PostgreSQL 连接串。

**3. 拉取 Embedding 模型（仅首次）**

```bash
ollama pull nomic-embed-text
```



**4. 启动 Web**

```bash
make web
```

浏览器打开 **http://localhost:9111**。

**首次登录**：切换到「使用 API Key 登录」，输入上一步的 `TEAM_MEMORY_API_KEY`，即以 admin 身份进入。

**多人使用**：
- 团队成员在登录页点击「注册」，填写用户名和密码
- Admin 在「设置 > 用户管理」中审批注册申请，系统自动生成 API Key
- Admin 将 API Key 分发给成员，用于 MCP 客户端（Cursor/Claude Desktop）接入
- 成员后续 Web 登录使用用户名 + 密码，MCP 使用 API Key

**5. 可选：健康检查**

```bash
make health
```

日常再次启动只需：`make dev`（或先 `docker compose up -d` 再 `make web`）。更多命令见下方「运维」。

---

### 二、初级使用者：在 Cursor / Claude 里接入

你已经从部署者拿到 **API Key**，TeamMemory 的数据库（和 Web）已就绪。只需在本机配置 MCP 并验证。

**1. 安装（本机跑 MCP 时）**

```bash
pip install team_memory
```

若通过 Cursor/Claude 的「MCP 市场」安装 TeamMemory，按客户端说明即可，可能无需本机再装。

**2. 拿到 API Key 与数据库连接**

- 向部署者索取 **TEAM_MEMORY_API_KEY**。
- 若你**本机直连**团队 PostgreSQL（即 MCP 进程自己连库），还需 **TEAM_MEMORY_DB_URL**（如 `postgresql+asyncpg://用户:密码@主机:5432/team_memory`）。若你使用「从源码运行」且项目目录下已有正确配置，可不设 DB_URL，由 config 提供。

**3. 配置 MCP**

在 **Cursor** 中编辑项目或用户下的 `.cursor/mcp.json`；在 **Claude Desktop** 中编辑 MCP Servers 对应配置。

- **从 pip 安装、无项目目录**（推荐）：用本机 Python + 环境变量，无需 `cwd`：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "python3",
      "args": ["-m", "team_memory.server"],
      "env": {
        "TEAM_MEMORY_DB_URL": "postgresql+asyncpg://用户:密码@主机:5432/team_memory",
        "TEAM_MEMORY_API_KEY": "你的API密钥",
        "TEAM_MEMORY_USER": "你的名字"
      }
    }
  }
}
```

- **从源码运行**（本地有仓库、用项目 venv 和 config）：替换路径为你的项目根目录：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/team_memory/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_memory",
      "env": {
        "TEAM_MEMORY_API_KEY": "你的API密钥",
        "TEAM_MEMORY_USER": "你的名字"
      }
    }
  }
}
```

**4. 验证**

重启 Cursor 或 Claude Desktop，在对话里输入：「请搜索经验库中关于 Docker 的经验」。若配置正确，AI 会调用 `tm_search` 并返回结果。

---

### 其他安装方式

- **仅 Docker、不克隆源码**：若你只想跑 Web 不跑本地 MCP，可用 `docker compose up -d` 启动（含数据库迁移）。在 `.env` 中设置 `TEAM_MEMORY_API_KEY`（默认 `changeme` 请修改），访问 http://localhost:9111 。
- **从 PyPI 部署、无 Make**：`pip install team_memory` 后，需自备 PostgreSQL（pgvector）并从本仓库克隆后在项目根目录执行 `alembic upgrade head`，再通过 `TEAM_MEMORY_CONFIG_PATH` 或 `TEAM_MEMORY_DB_URL` 启动 `team-memory-web` 或 `python -m team_memory.server`。

---

## MCP 接入指南（配置参考）

**Cursor**（`.cursor/mcp.json`）— 从 pip 安装、无项目目录：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "python3",
      "args": ["-m", "team_memory.server"],
      "env": {
        "TEAM_MEMORY_DB_URL": "postgresql+asyncpg://用户:密码@主机:5432/team_memory",
        "TEAM_MEMORY_API_KEY": "你的API密钥",
        "TEAM_MEMORY_USER": "你的名字"
      }
    }
  }
}
```

**从源码目录运行**（使用项目 venv 与 config）：将 `command` 改为 `/path/to/team_memory/.venv/bin/python`，增加 `"cwd": "/path/to/team_memory"`，`env` 中至少保留 `TEAM_MEMORY_API_KEY` 与 `TEAM_MEMORY_USER`（数据库由项目内 config 提供）。

**Claude Desktop**：在 MCP 设置中添加同名 `team_memory` 条目，`command` / `args` / `env` 与上一致即可。

| 变量 | 必填 | 说明 |
|------|------|------|
| `TEAM_MEMORY_DB_URL` | 是（或由 config 提供） | PostgreSQL 连接串，`postgresql+asyncpg://...`，库需启用 pgvector |
| `TEAM_MEMORY_API_KEY` | 是 | 与部署者在 auth 中配置一致 |
| `TEAM_MEMORY_USER` | 否 | 当前用户标识，默认 `anonymous` |
| `TEAM_MEMORY_CONFIG_PATH` | 否 | 配置文件路径，设置则优先从该文件加载 |

### 实际场景：AI 如何使用 TeamMemory

**场景 1：遇到问题，AI 自动查找历史方案**

```
你：这个 Docker 容器的网络一直不通，帮我看看

AI 的行为（自动触发，无需手动操作）：
  1. 调用 tm_solve(problem="Docker容器网络不通", framework="docker")
  2. 命中团队经验：「上次是 bridge 网络 DNS 解析问题，需要指定 --dns」
  3. AI 直接基于历史方案给出解决步骤，并标注"参考了团队经验库"
```

**场景 2：问题解决后，AI 自动保存经验**

```
你和 AI 经过一番排查，最终解决了问题

AI 的行为（自动触发）：
  1. 调用 tm_learn(conversation="...", tags=["docker","network"])
  2. LLM 自动提取：标题、问题描述、根因、解决方案、标签
  3. 保存为草稿，等待人工审核确认后发布
  4. 下次团队任何人遇到类似问题，AI 直接命中这条经验
```

**场景 3：基于上下文主动推荐**

```
你正在编辑一个 Kubernetes 部署配置文件

AI 的行为（主动推荐）：
  1. 调用 tm_suggest(file_path="k8s/deployment.yaml", framework="kubernetes")
  2. 返回团队在 K8s 部署方面积累的经验和注意事项
  3. 避免重复踩坑
```

### MCP 工具列表

| 工具 | 功能 | 输入 | 典型场景 |
|------|------|------|----------|
| `tm_solve` | 智能问题求解 | 问题描述 + 上下文 | 遇到技术问题时优先调用 |
| `tm_search` | 语义搜索 | 自然语言查询 | 通用经验检索 |
| `tm_suggest` | 上下文推荐 | 文件路径/语言/框架/错误信息 | 主动推荐相关经验 |
| `tm_learn` | 对话学习 | 对话或文档文本 | 自动从对话中提取经验 |
| `tm_save` | 快速保存 | title + problem | 手动保存单条经验 |
| `tm_save_typed` | 完整保存 | 全量结构化字段 | 保存带类型和专属字段的经验 |
| `tm_save_group` | 保存经验组 | 父经验 + 子步骤 | 保存一组关联经验 |
| `tm_claim` | 认领经验 | 经验 ID | 声明正在处理某条经验 |
| `tm_notify` | 通知团队 | 经验 ID | 通过 Webhook 通知 |
| `tm_feedback` | 反馈评分 | 经验 ID + 1-5 分 | 对搜索结果评分 |
| `tm_update` | 更新经验 | 经验 ID + 字段 | 追加方案或标签 |
| `tm_config` | 查看配置 | 无 | 查看运行时配置快照 |
| `tm_status` | 系统状态 | 无 | 查看健康状态和 Pipeline 信息 |
| `tm_preflight` | 任务预检 | 任务描述、当前文件 | 分析复杂度并返回搜索建议 |
| `tm_task_claim` | 认领任务 | 任务 ID | 原子认领/释放，防并发冲突 |
| `tm_ready` | 就绪任务 | project（可选） | 查询依赖已满足的可执行任务 |
| `tm_message` | 任务消息 | 任务 ID | 发送/查看任务消息线程 |
| `tm_dependency` | 任务依赖 | 任务 ID、依赖类型 | 管理任务间 blocks/related 依赖 |
| `tm_doc_sync` | 文档同步 | .debug 文档路径 | 将文档幂等同步到经验库 |

### MCP 资源

| 资源 URI | 说明 |
|----------|------|
| `experiences://recent` | 最近创建的经验 |
| `experiences://stats` | 经验库统计 |
| `experiences://stale` | 疑似过时的经验 |

### MCP Prompts

| Prompt | 说明 |
|--------|------|
| `summarize_experience` | 引导从对话中提取经验 |
| `submit_doc_experience` | 提交文档作为经验 |
| `review_experience` | 审核经验质量 |
| `troubleshoot` | 系统化故障排查 |
| `analyze_long_document` | 长文档分段分析 |

## Web 管理界面

启动 Web 服务后访问 `http://localhost:9111`，提供经验的可视化管理：

| 页面 | 功能 |
|------|------|
| 仪表盘 | 经验总量、近期趋势、热门标签、类型分布 |
| 经验列表 | 浏览全部经验，按类型/标签/项目/进度多维筛选 |
| 草稿箱 | 查看 AI 自动提取的待审核草稿 |
| 审核队列 | 审核团队成员提交的经验 |
| 去重检测 | 发现和合并相似经验 |
| 任务 Kanban | 五列看板、任务组、任务侧边面板、AI Prompt、消息线程 |
| 归档管理 | 归档任务组、设置页折叠、取消归档 |
| 系统设置 | 检索参数、搜索配置、缓存、Webhook、Schema 配置 |

创建经验支持三种模式：
- **手动填写**：逐字段填写标题、问题、方案
- **文档解析**：粘贴文档或 Markdown，AI 自动提取字段
- **URL 导入**：输入链接，AI 自动抓取并解析内容

API 参考：
- **Swagger UI**: `http://localhost:9111/docs`
- **ReDoc**: `http://localhost:9111/redoc`

## 配置说明

配置分层加载（后者覆盖前者）：

| 层级 | 文件 | 用途 |
|------|------|------|
| 1 | `config.yaml` | 全量默认配置（按 `[必改]`/`[可选]`/`[高级]` 分级标注） |
| 2 | `config.minimal.yaml` | 用户简化配置（只需改 2 项） |
| 3 | `config.local.yaml` | 开发者高级覆盖 |
| 4 | `config.{env}.yaml` | 多环境叠加 |
| 5 | 环境变量 | 最高优先级 |

### 多环境配置

```bash
# 开发环境（默认）
TEAM_MEMORY_ENV=development  # 使用 config.yaml

# 生产环境
TEAM_MEMORY_ENV=production   # 叠加 config.production.yaml

# 测试环境
TEAM_MEMORY_ENV=test         # 叠加 config.test.yaml
```

### RBAC 角色权限

| 角色 | 权限 |
|------|------|
| admin | 全部操作（用户管理、配置修改、审计日志） |
| editor | 创建、编辑、删除、审核经验 |
| viewer | 只读（搜索、浏览、反馈） |

管理员通过 Web UI 设置页面管理 API Key 和角色分配。

### Embedding 配置

```yaml
embedding:
  provider: ollama  # ollama / openai / local
```

支持 Ollama（默认，本地运行，无需 API Key）、OpenAI API、本地 sentence-transformers 模型。

## 运维

### 常用命令

```bash
make setup         # 首次安装
make dev           # 启动全部服务
make web           # 仅启动 Web（9111）
make health        # 一键健康检查
make backup        # 备份数据库
make test          # 运行测试
make lint          # 代码检查（ruff）
make hooks-install # 安装 Git hooks（commit 含 [TM-xxx] 时自动更新任务）
```

### Make 命令说明（等价手动命令）

便于问题定位和手动分步启动时参考：

| 命令 | 含义 | 等价手动步骤 |
|------|------|----------------|
| `make help` | 列出所有 make 目标 | `grep -E '^[a-zA-Z_-]+:.*?## ' Makefile \| awk ...` |
| `make release-9111` | 释放 9111 端口 | 停掉占用 9111 的 Docker 容器（如 `team-memory-web`）和本机进程：<br>`docker compose stop team-memory-web`<br>`docker ps -q --filter "publish=9111" \| xargs docker stop`<br>`lsof -i :9111 -t \| xargs kill -9` |
| `make setup` | 首次安装 | `docker compose up -d` → 等 PG 就绪 → 建库（若无）→ 按需启动 Ollama 容器 → `pip install -e ".[dev]"` → `alembic upgrade head` |
| `make dev` | 启动全部服务 | 先执行 `make release-9111`（避免 9111 被占）→ `docker compose up -d postgres redis` → 若 11434 未被占用则 `docker compose --profile ollama up -d` → 前台运行 `python -m team_memory.web.app` |
| `make web` | 仅启动 Web | 先执行 `make release-9111` → 前台运行 `python -m team_memory.web.app`（默认 http://0.0.0.0:9111） |
| `make mcp` | 仅启动 MCP 服务 | `python -m team_memory.server`（stdio，供 Cursor/Claude Desktop 调用） |
| `make health` | 健康检查 | `./scripts/healthcheck.sh`（检测 DB、Web、Ollama 等） |
| `make migrate` | 数据库迁移 | `alembic upgrade head` |
| `make migrate-fts` | 补齐经验表 FTS 字段（存量迁移） | `python scripts/migrate_fts.py`；可用 `--dry-run` 预览待更新条数 |
| `make hooks-install` | 安装 Git hooks | 复制 `scripts/post-commit-hook.sh` 到 `.git/hooks/post-commit`，commit 含 [TM-xxx] 时自动更新任务 |

**说明**：`make dev` 与 `make web` 会在启动前自动释放 9111 端口（停止占用该端口的 Docker 容器或本机进程），因此可重复执行而不会出现「address already in use」。

### 仪表盘报「加载仪表盘失败」时

1. **先看健康检查**：`make health` 或 `GET http://localhost:9111/health`。
   - 若输出中有 **`dashboard_stats: FAIL`** 及后面的 `error` / `ops_hint`，按提示排查（常见原因：数据库未启动、未执行迁移、或 config 中 `database.url` 错误）。
   - 若 **database 为 FAIL**：先启动数据库（如 `docker compose up -d postgres`），执行 `alembic upgrade head` 后再访问仪表盘。

2. **API 错误会返回 JSON**：若后端报 500，接口会返回 `detail`、`ops_error_id`、`ops_hint`。
   - 前端会把这些信息拼进错误提示；
   - 在服务端日志中按 `ops_error_id` 搜索可定位对应异常。

3. **本地快速诊断**：`python scripts/smoke_web_dashboard.py [--api-key KEY]` 会请求 `/health` 和 `/api/v1/stats` 并打印结果，便于确认是数据库、配置还是鉴权问题。

### 备份恢复

```bash
# 备份
make backup
# 或: ./scripts/backup.sh [output_dir]

# 恢复
./scripts/restore.sh backups/team_memory_20260209_120000.sql.gz
```

### Docker 部署

```bash
# 自动等待 PG、运行迁移、拉取 Ollama 模型、生成 admin key
docker compose up -d
# 统一端口: 9111
```

### 监控

- 内置仪表盘: Web UI 首页
- 健康检查: `make health` 或 `GET /health`
- 就绪探针: `GET /ready`

## 技术栈

| 层面 | 技术选型 |
|------|----------|
| MCP | FastMCP |
| Web | FastAPI + Uvicorn |
| 数据库 | PostgreSQL + pgvector |
| ORM | SQLAlchemy 2.0 (async) |
| 嵌入 | Ollama / OpenAI / 本地模型 |
| 搜索 | 向量 + 全文检索 + RRF 混合 + Reranker |
| 缓存 | 内存 LRU / Redis |
| 配置 | Pydantic Settings + YAML 分层 |
| 迁移 | Alembic |

## FAQ

**Q: TeamMemory 和 Cursor Rules 有什么区别？**

A: Rules 是静态的、手动维护的规范文件，适合"已经确定的"知识（代码风格、架构约束）。TeamMemory 是动态的、AI 自动积累的经验库，适合"还在演化的"知识（Bug 根因、方案权衡、实际踩坑）。两者互补：经验在 TeamMemory 中积累，当某个模式足够稳定后，可以将其固化为 Rule。

**Q: 经验库越来越大，AI 上下文会不会爆？**

A: 不会。TeamMemory 有多层机制控制输出体量：语义搜索本身只返回最相关的 Top-N 结果；Token 预算控制会自动裁剪过长的输出；Reranker 会过滤低相关度结果；PageIndex-Lite 对长文档做节点级检索而非全文返回。经验库从 100 条增长到 10000 条，AI 每次实际读取的内容量不会显著增加。

**Q: 团队成员不想手动录入经验怎么办？**

A: 这正是 TeamMemory 的设计重点。`tm_learn` 工具让 AI 自动从开发对话中提取经验，团队成员只需要正常和 AI 对话解决问题，经验会自动被捕获并保存为草稿。审核后发布即可。Web 界面也支持粘贴文档或输入 URL 来自动生成经验。

**Q: 切换 Embedding 模型后需要做什么？**

A: 需要重新生成所有 embedding。使用 `scripts/migrate_embeddings.py`。

**Q: 存量数据如何支持全文检索（FTS）？**

A: 若经验表存在 `fts` 为空的记录，可执行 `make migrate-fts` 或 `python scripts/migrate_fts.py` 回填；先用 `--dry-run` 可预览待更新条数。

**Q: 没有 Ollama 可以使用吗？**

A: 可以。将 `embedding.provider` 改为 `openai` 并配置 API Key，或使用 `local` 加载本地 sentence-transformers 模型。

**Q: 多个项目的经验会混在一起吗？**

A: 不会。通过 `project` 参数或环境变量 `TEAM_MEMORY_PROJECT` 实现项目级隔离。每个项目的经验独立存储和检索，不会互相干扰。

## 开发

### 运行测试

```bash
# 全部测试
pytest -v

# 带覆盖率
pytest --cov=team_memory
```

### 代码检查

```bash
# 检查
ruff check src/

# 自动修复
ruff check src/ --fix
```

### CI/CD

推送至 `main` / `develop` 或向 `main` 提 PR 时，GitHub Actions 会执行 lint、测试与 Docker 构建。触发条件与各 job 说明见 [.debug/25-CI/CD 流水线](.debug/25-ci-cd.md)。
