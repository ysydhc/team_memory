# Agent 记忆项目调研

> 调研主流 Agent 记忆开源项目，重点标注「档案馆式」存储思路。  
> 日期：2025-03-11

---

## 〇、OpenViking（用户提及的 OpenVxxx）

**GitHub**: [volcengine/OpenViking](https://github.com/volcengine/OpenViking) · 字节跳动火山引擎 · 5.7k+ stars · Apache 2.0

**核心思路**：用「**文件系统范式**」重构 Agent 记忆，将所有上下文统一组织在虚拟文件系统中，通过 `viking://` 协议为每个条目赋予唯一 URI。Agent 可像操作文件一样使用 `ls`、`find`、`glob`、`read` 精确定位信息，而非依赖模糊语义匹配。

### viking:// 虚拟文件系统（档案馆式目录）

```
viking://
├── resources/          # 外部资源（文档、代码库等）
├── user/              # 用户记忆（偏好、习惯）
└── agent/             # Agent 技能和经验
    ├── skills/
    ├── memories/
    └── instructions/
```

### L0/L1/L2 分层上下文

| 层级 | Token 量 | 用途 |
|------|----------|------|
| **L0（摘要）** | ~100 | 一句话概括，快速筛选 |
| **L1（概览）** | ~2000 | 核心信息，规划决策 |
| **L2（全文）** | 完整 | 按需加载 |

每个目录拥有 `.abstract` 和 `.overview` 文件，形成结构化档案馆。**目录递归检索**：向量定位高分目录 → 目录内二次检索 → 递归子目录。

**可借鉴点**：文件系统式组织、分层按需加载、目录级检索、Token 成本降低 90%+。

---

## 一、主流 Agent 记忆项目一览

| 项目 | 机构/作者 | Stars | 核心思路 | 是否档案馆式 |
|------|-----------|-------|----------|--------------|
| **OpenViking** | 字节火山引擎 | 5.7k+ | 文件系统范式，viking:// 虚拟目录，L0/L1/L2 分层 | ✅ **是** |
| **MemGPT / Letta** | Letta (原 MemGPT 团队) | 21k+ | 虚拟上下文管理，主上下文 + 外部存储（Recall + Archival） | ✅ **是** |
| **Mem0** | mem0ai | 50k+ | 知识图谱 + 向量 + 键值，托管/自托管 | 否 |
| **Zep** | getzep | - | 时间性知识图谱，多级内存 | 否 |
| **Memory Palace** | AGI-is-going-to-arrive | - | SQLite + 混合检索，MCP 集成 | 否 |
| **OpenClaw Memory** | codesfly / OpenClaw | - | 三层：短期日志 → 中期会话 → 长期 MEMORY.md | ✅ **是** |
| **ReMe** | ModelScope / AgentScope | 2k+ | 文件型内存，MEMORY.md + 日日志 | ✅ **是** |
| **Engram** | zanfiel | - | 语义+全文搜索，知识图谱，版本控制 | 否 |
| **MemoryKit** | 0j | - | 轻量语义检索，自动压缩 | 否 |
| **Always On Memory** | Google PM | - | 弃用向量库，LLM 直接组织记忆，SQLite | 否 |

---

## 二、档案馆式存储项目详解

### 2.1 MemGPT / Letta — Archival Storage（归档存储）

**GitHub**: [letta-ai/letta](https://github.com/letta-ai/letta)

**核心思路**：借鉴操作系统虚拟内存，将记忆分为「主上下文」与「外部上下文」。外部上下文包含两类：

| 类型 | 英文 | 用途 |
|------|------|------|
| **回溯存储** | Recall Storage | 完整对话历史，可检索特定时间段 |
| **归档存储** | **Archival Storage** | 文档/大数据集（如 Wikipedia、PDF），主上下文的**溢出空间** |

当主上下文空间不足时，LLM 通过函数调用（如 `archival_memory_search()`）将重要信息保存到归档存储，需要时再按页取回。**档案馆 = 磁盘式冷存储**，与主内存形成分层。

**可借鉴点**：
- 主上下文 ↔ 归档的「换页」机制
- 归档作为文档/大数据的持久化池，按需检索注入

---

### 2.2 OpenClaw Memory — 三层档案式精炼

**GitHub**: [codesfly/openclaw-memory-final](https://github.com/codesfly/openclaw-memory-final)

**核心思路**：短期 → 中期 → 长期，逐层精炼，长期记忆相当于「档案馆」。

| 层级 | 存储形式 | 说明 |
|------|----------|------|
| **短期** | `memory/YYYY-MM-DD.md` | 每日 append-only 日志，48 小时上下文 |
| **中期** | Sessions | 完整会话存档，压缩时关键信息冲刷到此 |
| **长期** | `MEMORY.md` | 精选持久知识：用户偏好、技术决策、最佳实践；**周精炼**整合 |

**精炼流程**：日增量 → 周精炼 → 幂等去重 → QMD 索引。长期记忆 = 经过筛选的「档案库」。

**可借鉴点**：
- 时间维度分层（日/周）
- 精炼（Refinement）将碎片整合为结构化知识
- Markdown 文件可读、可审计、git 友好

---

### 2.3 ReMe — 文件型档案

**GitHub**: [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe)（Apache 2.0，2k+ stars）

**核心思路**：将记忆存储为可读写的 Markdown 文件，而非数据库。

| 组件 | 路径 | 说明 |
|------|------|------|
| **长期记忆** | `MEMORY.md` | 持久知识 |
| **日日志** | `memory/YYYY-MM-DD.md` | 按日归档 |
| **工具输出** | `tool_result/` | 工具结果缓存 |

**可借鉴点**：
- 文件即档案，结构简单
- 上下文检查、历史压缩、内存总结

---

## 三、档案馆式 vs 向量库式

| 维度 | 档案馆式（MemGPT/OpenClaw/ReMe） | 向量库式（Mem0/Zep） |
|------|----------------------------------|----------------------|
| 存储 | 文件/分层冷热 | 向量 + 图 + 键值 |
| 检索 | 按需取页/按路径/按时间 | 语义相似度 + 图遍历 |
| 精炼 | 显式压缩、整合、归档 | 隐式去重、合并 |
| 可解释性 | 高（Markdown 可读） | 中（需解析） |
| 适用 | 文档、日志、决策记录 | 语义检索、知识图谱 |

---

## 四、TM 可借鉴的档案馆思路

| 借鉴点 | 来源 | 落地建议 |
|--------|------|----------|
| **分层冷热** | MemGPT Archival | Experience 可增加「活跃/归档」状态，低访问频率自动归档 |
| **精炼流程** | OpenClaw 周精炼 | 定期将碎片经验整合为 MEMORY.md 式结构化知识 |
| **文件可审计** | ReMe / OpenClaw | 导出为 Markdown 档案，支持 git 版本管理 |
| **按时间分层** | OpenClaw 日/周 | 短期（最近 N 天）优先检索，长期按需加载 |

---

## 五、你记得的「档案馆」项目可能是？

根据「档案馆方式存储资料」的描述，最可能对应：

1. **MemGPT / Letta** — 明确使用 **Archival Storage（归档存储）** 作为主上下文的溢出空间，文档/大数据集存入「档案馆」，按需检索。
2. **OpenClaw Memory** — 长期记忆 `MEMORY.md` 相当于精选档案库，经周精炼整合。
3. **ReMe** — 文件型记忆，`MEMORY.md` + 日日志，结构类似档案分层。

若强调「档案馆」一词，**MemGPT 的 Archival Storage** 最贴切；若强调「文件 + 精炼」，则 **OpenClaw** 或 **ReMe** 更接近。

---

## 六、OpenViking 与 TM 可借鉴点

| 借鉴点 | OpenViking 设计 | TM 落地建议 |
|--------|-----------------|-------------|
| **文件系统式组织** | viking:// 虚拟目录，resources/user/agent 分层 | Experience 可按 project/type/category 组织为虚拟路径，支持 `ls`/`find` 式查询 |
| **L0/L1/L2 分层** | 摘要 ~100 token → 概览 ~2000 → 全文按需 | Experience 已有 summary，可扩展 overview（中粒度）字段，检索时按需加载 |
| **目录级检索** | 先向量定位目录，再目录内精确检索 | 可先按 category/type 筛目录，再在目录内做向量/FTS |
| **Token 成本** | 分层加载，节省 90%+ | 检索结果可只返 L0/L1，需要时再拉 L2 |
