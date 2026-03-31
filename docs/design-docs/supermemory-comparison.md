# Supermemory vs TeamMemory 对比调研

> 调研日期：2026-03-27

## 1. 项目概述

### Supermemory
- **定位**：AI 的记忆和上下文引擎（State-of-the-art memory and context engine for AI）
- **官网**：https://supermemory.ai
- **GitHub**：https://github.com/supermemoryai/supermemory
- **Benchmark**：LongMemEval、LoCoMo、ConvoMem 三大 AI 记忆基准测试均排名第一

### TeamMemory
- **定位**：团队经验数据库，让 AI 拥有跨会话记忆
- **目标**：解决 AI 编程助手「每次对话都从零开始」的痛点

## 2. 核心功能对比

| 功能 | Supermemory | TeamMemory |
|------|-------------|------------|
| **记忆提取** | ✅ 自动从对话提取事实 | ✅ tm_learn 从对话提取经验 |
| **用户画像** | ✅ 自动维护 User Profile（静态+动态） | ⚠️ 部分支持：`memory_context` / `memory_recall(include_user_profile)` 返回 `profile.static` / `profile.dynamic`（自建库，无 Supermemory 云端 profile API） |
| **混合搜索** | ✅ RAG + Memory 一体化 | ✅ 向量 + 全文 + RRF 融合 |
| **知识更新/矛盾处理** | ✅ 自动处理时间变化、矛盾 | ⚠️ 需手动更新 |
| **自动遗忘** | ✅ 过期信息自动遗忘 | ❌ 无自动遗忘 |
| **连接器** | ✅ Google Drive/Gmail/Notion/GitHub | ❌ 仅本地文档同步 |
| **文件处理** | ✅ PDF/图片/视频/代码 | ⚠️ 仅文本 |
| **团队协作** | ⚠️ 以个人为中心 | ✅ 团队共享经验库 |
| **经验类型** | ❌ 通用记忆 | ✅ 结构化类型（bugfix/feature/incident...） |
| **架构可视化** | ❌ 无 | ✅ GitNexus 集成 |
| **MCP 支持** | ✅ 官方 MCP Server | ✅ MCP Server |

## 3. 技术架构对比

| 维度 | Supermemory | TeamMemory |
|------|-------------|------------|
| **部署模式** | SaaS 为主，API 调用 | 自托管为主 |
| **协议** | REST API + MCP | MCP + Web API |
| **存储** | 云端托管 | PostgreSQL + pgvector |
| **SDK** | npm/pip 包 | pip 包 |
| **定价** | 免费额度 + 付费 | 开源免费 |

## 4. Supermemory 核心能力详解

### 4.1 Memory Engine（记忆引擎）
- 自动从对话中提取事实
- 处理时间变化（"我刚搬到 SF" 会取代 "我住在 NYC"）
- 解决矛盾
- 自动遗忘过期信息（"我明天有考试" 在日期过后失效）

### 4.2 User Profiles（用户画像）
```typescript
const { profile } = await client.profile({ containerTag: "user_123" });
// profile.static  → ["Senior engineer at Acme", "Prefers dark mode", "Uses Vim"]
// profile.dynamic → ["Working on auth migration", "Debugging rate limits"]
```
- 一次调用，~50ms
- 静态事实 + 动态上下文
- 注入 system prompt 即可让 AI 了解用户

### 4.3 Hybrid Search（混合搜索）
- RAG（文档检索）+ Memory（用户记忆）在一个查询中
- 返回知识库文档 + 个性化上下文

### 4.4 Connectors（连接器）
支持的数据源：
- Google Drive
- Gmail
- Notion
- OneDrive
- GitHub
- Web Crawler

实时 Webhook 同步，自动处理、分块、可搜索。

### 4.5 File Processing（文件处理）
- PDF（提取文本）
- 图片（OCR）
- 视频（转录）
- 代码（AST 感知分块）

## 5. Supermemory MCP 工具

| 工具 | 功能 |
|------|------|
| `memory` | 保存或遗忘信息，AI 自动调用 |
| `recall` | 按查询搜索记忆，返回相关记忆 + 用户画像摘要 |
| `context` | 注入完整用户画像（偏好、近期活动） |

## 6. 各自优势

### Supermemory 优势
1. **智能记忆管理**：自动提取、更新、遗忘，无需手动维护
2. **用户画像**：一次调用获取完整用户上下文
3. **多数据源连接器**：自动同步 Google Drive、Notion 等
4. **Benchmark 领先**：三大基准测试均第一
5. **框架集成**：Vercel AI SDK、LangChain、Mastra 等开箱即用

### TeamMemory 优势
1. **团队导向**：专为技术团队设计，支持经验共享
2. **结构化经验**：类型化的经验（bugfix/incident/tech_design）
3. **完全自托管**：数据在自己的服务器，无外部依赖
4. **架构感知**：GitNexus 集成，经验与代码位置绑定
5. **开源免费**：无使用限制

## 7. 适用场景建议

| 场景 | 推荐 |
|------|------|
| 个人 AI 助手记忆 | Supermemory |
| 技术团队知识沉淀 | TeamMemory |
| 快速原型/小项目 | Supermemory（SaaS 快速接入） |
| 企业内网部署 | TeamMemory（自托管） |
| 需要连接 Google Drive/Notion | Supermemory |
| 需要与代码架构关联 | TeamMemory |

## 8. 可借鉴功能

从 Supermemory 可以借鉴到 TeamMemory 的功能：

### 8.1 用户画像（User Profile）
- 自动维护静态事实 + 动态上下文
- 一次调用获取完整上下文

### 8.2 自动遗忘机制
- 过期经验自动失效
- 基于时间的记忆管理

### 8.3 矛盾检测与处理
- 新经验与旧经验冲突时自动更新
- 时间感知的知识更新

### 8.4 更丰富的连接器
- Google Drive、Notion 等外部数据源
- 实时 Webhook 同步

### 8.5 Benchmark 框架
- 标准化的记忆能力评测
- MemoryBench 开源评测工具

## 9. 参考链接

- [Supermemory 文档](https://supermemory.ai/docs)
- [Supermemory Quickstart](https://supermemory.ai/docs/quickstart)
- [MemoryBench](https://supermemory.ai/docs/memorybench/overview)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
- [LoCoMo](https://github.com/snap-research/locomo)
- [ConvoMem](https://github.com/Salesforce/ConvoMem)
