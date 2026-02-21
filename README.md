# TeamMemory

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

- **对话提取**（`tm_learn`）：LLM 自动从开发对话中识别问题、方案、标签，提取为结构化经验
- **文档解析**：上传文档或输入 URL，AI 自动解析为标题、问题描述、解决方案、标签
- **经验组**（`tm_save_group`）：将一次完整的问题解决过程（从发现到排查到修复）作为父子经验组保存
- **默认草稿模式**：AI 提取的内容默认为草稿，经人工审核后发布，保证质量

### 智能检索

多层检索管线，确保 AI 找到最相关的历史方案：

- **语义搜索**：基于向量嵌入（Ollama / OpenAI / 本地模型），理解查询意图
- **混合检索**：向量搜索 + 全文检索 + RRF 融合排序
- **Reranker**：支持服务端 LLM 精排，或客户端 AI 自行判断结果相关性
- **Token 预算控制**：自动裁剪输出长度，避免经验库增大后撑爆 AI 上下文
- **PageIndex-Lite**：长文档自动分块建立节点索引，支持节点级精准检索

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

### 方式一：使用 Makefile（推荐）

```bash
# 首次安装：启动 Docker + 安装依赖 + 初始化数据库
make setup

# 启动 Web 管理界面（默认端口 9111）
make web

# 查看所有可用命令
make help
```

### 方式二：手动安装

```bash
# 1. 启动基础设施（PostgreSQL + Ollama + Redis）
docker compose up -d

# 2. 安装 Python 依赖
pip install -e ".[dev]"

# 3. 初始化数据库
alembic upgrade head

# 4. 准备 Embedding 模型（仅首次需要）
ollama pull nomic-embed-text

# 5. 启动服务
python -m team_memory.web.app    # Web 管理界面（http://localhost:9111）
python -m team_memory.server     # MCP Server（供 Cursor / Claude Desktop 使用）
```

### 配置

- **最小配置**: 修改 `config.minimal.yaml` 中的 `auth.api_key` 即可启动
- **完整配置**: `config.yaml` 包含所有选项，按 `[必改]` / `[可选]` / `[高级]` 分级标注
- **健康检查**: `make health` 或 `./scripts/healthcheck.sh`

## MCP 接入指南

### Cursor 配置

在项目的 `.cursor/mcp.json` 中添加：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/team_memory/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_memory",
      "env": {
        "TEAM_MEMORY_API_KEY": "your-api-key",
        "TEAM_MEMORY_USER": "your-name"
      }
    }
  }
}
```

### Claude Desktop 配置

在 Claude Desktop 设置中添加 MCP Server：

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/path/to/team_memory/.venv/bin/python",
      "args": ["-m", "team_memory.server"],
      "cwd": "/path/to/team_memory",
      "env": {
        "TEAM_MEMORY_API_KEY": "your-api-key"
      }
    }
  }
}
```

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
make setup     # 首次安装
make dev       # 启动全部服务
make health    # 一键健康检查
make backup    # 备份数据库
make test      # 运行测试
make lint      # 代码检查（ruff）
```

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
