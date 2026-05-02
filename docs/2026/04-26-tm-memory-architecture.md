# TM Memory System v9 — 完整架构图

> 生成时间: 2026-04-26
> 覆盖: Hermes / Cursor / Claude Code 三端接入的记忆体系

---

## 1. 总体架构

```mermaid
graph TB
    subgraph Clients["💻 AI Agent Clients — 三平台接入"]
        direction LR
        H[Hermes<br/>Python API 直调]
        C[Cursor<br/>hooks.json 外部 Hook]
        CC[Claude Code<br/>settings.json 外部 Hook]
    end

    subgraph Daemon["🔥 TM Daemon — FastAPI localhost:3901"]
        direction TB
        API["<b>HTTP API Endpoints</b>
        /hooks/before_prompt
        /hooks/after_response
        /hooks/session_start
        /hooks/session_end
        /draft/save & /publish
        /recall & /search/stats"]

        subgraph Pipeline["⚡ Pipeline Layer"]
            P1["process_before_prompt()"]
            P2["process_after_response()"]
            P3["process_session_start()"]
            P4["process_session_end()"]
        end

        subgraph CoreComponents["🛠️ Core Components"]
            SINK["TMSink<br/>(Abstract)
            ├─ LocalTMSink: Python 直调
            └─ RemoteTMSink: HTTP 远程"]
            BUF["DraftBuffer<br/>(SQLite 本地缓存)
            ├─ create/update_draft
            ├─ get_pending
            └─ mark_published"]
            LOG["SearchLogWriter<br/>(PostgreSQL)
            ├─ log_search
            ├─ mark_used [mem:xxx]
            └─ get_stats"]
            DET["ConvergenceDetector<br/>(收敛检测)"]
            REF["DraftRefiner<br/>(LLM 精修 + 发布)"]
        end
    end

    subgraph Storage["💾 Storage Layer"]
        direction TB
        PG[(PostgreSQL
        — experiences 表
        — search_logs 表
        — personal_memory 表)]
        CH[(Chroma
        — experience_embeddings
        向量检索)]
        SL[(SQLite
        — drafts 草稿缓存)]
    end

    H -->|HTTP POST| API
    C -->|HTTP POST| API
    CC -->|HTTP POST| API

    API --> Pipeline
    Pipeline --> SINK
    Pipeline --> BUF
    Pipeline --> LOG
    Pipeline --> DET
    Pipeline --> REF

    SINK -->|read/write| PG
    SINK -->|vector search| CH
    BUF -->|draft CRUD| SL
    LOG -->|log stats| PG
```

---

## 2. 一次完整的对话循环

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant Client as AI Client
    participant Hook as Hook Script
    participant Daemon as TM Daemon
    participant Search as Search Pipeline
    participant DB as PostgreSQL/Chroma
    participant Draft as DraftBuffer SQLite

    rect rgb(230, 245, 255)
        Note over U,DB: 🔍 Phase 1: 会话开始
        U->>Client: 新建对话
        Client->>Hook: session_start / SessionStart
        Hook->>Daemon: POST /hooks/session_start
        Daemon->>DB: 检索项目上下文
        DB-->>Daemon: 项目级 experience
        Daemon-->>Hook: {additional_context: "..."}
        Hook-->>Client: 注入项目背景
    end

    rect rgb(255, 245, 230)
        Note over U,DB: 💬 Phase 2: 用户提问
        U->>Client: "之前遇到的crash问题怎么解决的？"
        Client->>Hook: before_prompt / UserPromptSubmit
        Hook->>Daemon: POST /hooks/before_prompt

        Daemon->>Search: 启动检索流水线
        Search->>DB: 1. 缓存检查
        Search->>DB: 2. Embedding 编码
        Search->>DB: 3. 向量检索 + FTS 全文检索
        Search->>Search: 4. RRF 融合
        Search->>Search: 5. 精确匹配加权
        Search->>Search: 6. 自适应过滤
        Search->>Search: 7. 置信度标注
        DB-->>Search: 检索结果
        Search-->>Daemon: Top-5 experiences

        Daemon->>DB: 写入 search_log (评估采集)
        Daemon-->>Hook: {results: [...]}
        Hook-->>Client: additionalContext [mem:xxx]
        Client->>U: Agent 回复 (带记忆上下文)
    end

    rect rgb(230, 255, 230)
        Note over U,Draft: ✅ Phase 3: 回复收集
        Client->>Hook: after_response / Stop
        Hook->>Daemon: POST /hooks/after_response
        Daemon->>Draft: upsert_draft(累积回复)
        Draft-->>Daemon: draft_id
        Daemon->>DET: detect_convergence()

        alt 收敛检测通过
            Daemon->>REF: refine_and_publish()
            REF->>DB: LLM 精修 + 保存 experience
            Daemon->>Draft: mark_published(draft_id)
        else 未收敛
            Daemon->>Draft: 继续累积
        end

        Daemon->>DB: scan [mem:xxx] 标记
        Daemon->>DB: 更新 was_used (命中评估)
    end
```

---

## 3. 检索流水线 (Search Pipeline)

```mermaid
flowchart LR
    Q[用户 Query] --> Cache{Cache<br/>命中?}
    Cache -->|是| R1[返回缓存]
    Cache -->|否| EMB[Embedding<br/>编码]

    EMB --> VEC[向量检索<br/>Chroma]
    EMB --> FTS[FTS 全文检索<br/>PostgreSQL]

    VEC --> RRF[RRF 融合]
    FTS --> RRF

    RRF --> BOOST[精确匹配<br/>加权]
    BOOST --> FILTER[自适应<br/>过滤]
    FILTER --> RERANK{重排序<br/>配置?}

    RERANK -->|是| RER[LLM/交叉编码器<br/>重排序]
    RERANK -->|否| CONF[置信度<br/>标注]
    RER --> CONF

    CONF --> ARCHIVE{Archive<br/>包含?}
    ARCHIVE -->|是| MERGE[合并 Archive]
    ARCHIVE -->|否| CACHE2[写入缓存]
    MERGE --> CACHE2
    CACHE2 --> OUT[返回 Top-N<br/>Results]
```

---

## 4. 草稿管理流程

```mermaid
stateDiagram-v2
    [*] --> Pending: create_draft

    Pending --> Pending: update_draft
    Note right of Pending
        多轮对话累积内容
        每轮 after_response
        都 append 到草稿
    End note

    Pending --> ReadyToPublish: detect_convergence
    Note right of ReadyToPublish
        收敛检测触发:
        - 关键词匹配 (解决/总结/...)
        - 工具执行模式稳定
        - 任务路径不再变化
    End note

    ReadyToPublish --> Published: refine_and_publish
    Note right of Published
        LLM 精修草稿:
        - 提取 Problem/Solution
        - 生成 title + tags
        - 保存到 experiences 表
        - 写入 Chroma 向量库
    End note

    Pending --> [*]: session_end
    Note right of [*]
        会话结束时:
        - 未发布的草稿
        - 刷新为已发布
    End note
```

---

## 5. 三平台 Hook 对比

| 平台 | 配置文件 | Hook 类型 | 对应事件 | Hook 脚本 |
|------|---------|-----------|---------|-----------|
| **Hermes** | 无 (内置) | Python API | `on_turn_start` | `hermes_pipeline.py` |
| | | | `on_turn_end` | |
| **Cursor** | `~/.cursor/hooks.json` | 外部进程 | `sessionStart` | `cursor_session_start.py` |
| | | | `beforeSubmitPrompt` | `cursor_before_prompt.py` |
| | | | `afterAgentResponse` | `cursor_after_response.py` |
| **Claude Code** | `~/.claude/settings.json` | 外部进程 | `SessionStart` | `claude_session_start.py` |
| | | | `UserPromptSubmit` | `claude_user_prompt_submit.py` |
| | | | `Stop` | `claude_stop.py` |

### 通用 Hook 脚本 (`scripts/hooks/`)

```
hooks/
├── tm_hook.py              # CLI 工具: tm-hook session-start / before-prompt / after-response
├── hermes_pipeline.py      # Hermes 内置 pipeline (异步 HTTP 调用 Daemon)
├── cursor_session_start.py     # → POST /hooks/session_start
├── cursor_before_prompt.py     # → POST /hooks/before_prompt
├── cursor_after_response.py    # → POST /hooks/after_response
├── claude_session_start.py     # → POST /hooks/session_start
├── claude_user_prompt_submit.py# → POST /hooks/before_prompt
└── claude_stop.py              # → POST /hooks/after_response
```

---

## 6. 数据模型

```mermaid
erDiagram
    experiences ||--o{ experience_embeddings : "vectorized"
    experiences ||--o{ search_logs : "retrieved_by"

    experiences {
        uuid id PK
        string title
        text problem
        text solution
        text content
        string project
        string[] tags
        string scope
        datetime created_at
        datetime updated_at
    }

    experience_embeddings {
        string id PK
        vector embedding
        json metadata
    }

    search_logs {
        uuid id PK
        string query
        string project
        string source
        json result_ids
        boolean was_used
        datetime created_at
    }

    drafts {
        string id PK
        string project
        string conversation_id
        text content
        string status
        datetime created_at
    }

    personal_memory {
        uuid id PK
        string user_name
        string kind
        json data
        datetime created_at
    }
```

---

## 7. 配置文件一览

```
├── ~/.claude/settings.json          # Claude Code hooks + model 配置
├── ~/.cursor/hooks.json             # Cursor hooks 配置
├── ~/.hermes/skills/tm-memory.md    # Hermes skill (自动加载)
├── ~/Work/agent/team_doc/
│   ├── scripts/daemon/app.py              # FastAPI 应用
│   ├── scripts/daemon/pipeline.py         # 业务流水线
│   ├── scripts/daemon/tm_sink.py          # 存储抽象层
│   ├── scripts/daemon/draft_buffer.py     # SQLite 草稿缓存
│   ├── scripts/daemon/search_log_writer.py # 搜索日志 + 评估
│   ├── scripts/daemon/convergence_detector.py
│   ├── scripts/daemon/draft_refiner.py
│   ├── scripts/daemon/markdown_indexer.py # Obsidian 文档索引
│   ├── scripts/daemon/watcher.py          # 文件监控
│   ├── scripts/hooks/                     # 三平台 hook 脚本
│   ├── src/team_memory/
│   │   ├── services/search_pipeline.py      # 检索引擎
│   │   ├── services/memory_operations.py  # CRUD 操作
│   │   ├── storage/repository.py          # SQL 仓库
│   │   ├── embedding/                     # Embedding 提供商
│   │   ├── reranker/                      # 重排序提供商
│   │   └── web/                           # Web API / MCP 端点
│   └── tests/                             # 测试集
├── ~/Obsidian/TeamMemory/           # Obsidian vault (标准化文档)
└── ~/Work/.../.claude/settings.json  # 项目级 Claude Code 配置
```

---

## 8. 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| `use_rate` | 检索结果被 Agent 引用的比例 | `was_used=True / 有结果的检索总数` |
| `hit_rate` | 检索命中率 (有无返回结果) | `有结果的检索总数 / 检索总数` |
| `query_top10` | 最常检索的问题 | `GROUP BY query ORDER BY count DESC` |

**CLI 查看:**
```bash
tm-hook stats          # 本周统计
tm-hook stats --days 3 # 近3天统计
```

---

## 9. 启动顺序

```mermaid
sequenceDiagram
    participant U as User
    participant D as TM Daemon
    participant PG as PostgreSQL
    participant CH as Chroma
    participant W as Watcher

    U->>PG: 启动 PostgreSQL (如未运行)
    U->>CH: 启动 Chroma (如未运行)
    U->>D: python -m scripts.daemon.app
    D->>PG: 连接数据库 + 创建表
    D->>CH: 连接向量库
    D->>D: 初始化 DraftBuffer (SQLite)
    D->>D: 初始化 ConvergenceDetector
    D->>D: 初始化 DraftRefiner
    D->>D: 初始化 SearchLogWriter
    D->>W: 启动 Obsidian Watcher
    W->>D: 监控 vault 变化
    D-->>U: FastAPI 启动完成 localhost:3901
```

---

## 10. 关键设计决策

1. **为什么用 Daemon 而不是每个 Client 直接调 TM?**
   - 统一接口: 三个平台走同一套 pipeline
   - 状态管理: DraftBuffer 需要跨轮对话的状态
   - 性能: 检索引擎初始化成本高，不应每次 hook 都重新初始
   - 评估: SearchLog 需要单个位置统一写入

2. **为什么 Draft 用 SQLite 而非 PostgreSQL?**
   - 草稿是临时状态，不需要共享
   - SQLite 零配置，便于本地运行
   - 与 PostgreSQL 解耦，daemon 可独立启停

3. **为什么检索走向量+FTS Hybrid?**
   - 向量: 语义匹配 ("之前的问题" → 相关实体)
   - FTS: 关键词匹配 (crash / 上报 / 解决)
   - RRF 融合: 取两者之长

4. **为什么需要收敛检测，而不是每次都保存?**
   - 避免粒度过细: 单轮对话通常只是片段信息
   - 跨轮累积: 多轮后形成完整问题+解决方案
   - 自动发布: 减少用户手动操作
