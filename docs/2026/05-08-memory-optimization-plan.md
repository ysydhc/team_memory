# TM 记忆系统优化方案

> 日期: 2026-05-08
> 状态: 规划中

## 一、全链路优化路线

### 1. 质量门控：拉取→加工→推送→清理

| 阶段 | 现状 | 优化 |
|------|------|------|
| 拉取(采集) | ConversationBuffer 存储 user+assistant messages | ✅ 已完成 |
| 加工(精炼) | RefinementWorker LLM 精炼，prompt 泛用，无最低质量门槛 | 重写精炼 prompt：要求 problem→solution 结构，拒绝纯描述性内容，加反例引导，输入加入 project 上下文 |
| 推送(入库) | 精炼后直接写入 PG，无验证步骤 | 增加 LLM 验证步骤：quality_score < 60 标记 draft 不入库；title/description/solution 任一为空或 solution < 50 字则丢弃 |
| 检索(查询) | 知识卡片和经验同等权重 | experience_type 加权：experience 正常，knowledge_card 降权或默认不返回 |
| 评估(质量) | faithfulness + quality_score 评估结果未驱动任何动作 | Janitor 定时任务：Bronze + recall_count=0 超过30天→标记待审核；quality_score < 50→自动软删除；was_used=true→quality_score 加分 |
| 清理(淘汰) | Janitor 手动触发，软删除无自动触发 | Janitor Worker：每24h 扫描；Bronze + recall=0 + 超过30天→draft；矛盾检测；近重组合并 (cosine>0.95→保留 quality 高的) |

### 2. 去重/矛盾处理：全链路

| 阶段 | 现状 | 优化 |
|------|------|------|
| 拉取(采集) | 无去重，同一消息可能重复写入 | ConversationBuffer 消息级去重 (session_id + role + content hash) |
| 加工(精炼) | 精炼时不检查已有经验 | 精炼前检索相似经验：cosine>0.9→合并而非新建；精炼 prompt 加入已有经验摘要作为上下文 |
| 推送(入库) | PersonalMemoryService 有 cosine 去重，经验没有 | ExperienceService.write() 加去重：cosine>0.95→更新而非新建，保留 quality_score 更高的版本 |
| 检索(查询) | 无去重，可能返回重复结果 | 检索后按 group_key 去重，同一主题只返回最佳匹配 |
| 评估(质量) | 无矛盾检测 | Janitor 矛盾扫描：title 相似 + description 冲突→标记；定期用 LLM 判断是否矛盾 |
| 清理(淘汰) | 无自动合并 | Auto Dream（参考 Claude Code）：每24h + 至少5次会话后触发；Phase 1 定向→Phase 2 收集→Phase 3 合并→Phase 4 裁剪 |

### 3. 经验 vs 知识分离：全链路

| 阶段 | 现状 | 优化 |
|------|------|------|
| 拉取(采集) | wiki 编译的知识卡片和对话提取的经验同一入口 | 知识卡片不入 experiences 表，走 LLMWiki 独立通道 |
| 加工(精炼) | 同一套精炼逻辑 | 经验精炼：problem→solution 结构；知识精炼：概念→要点 结构 (不同 prompt) |
| 推送(入库) | 都写 experiences 表 | 经验：experiences 表 (PG)；知识：llmwiki 表 (本地 SQLite) |
| 检索(查询) | 混合检索，知识卡片噪声大 | 经验检索走 experiences；知识检索走 llmwiki (本地)；before_prompt 两路并行 |
| 评估(质量) | 知识卡片无 quality_score | 知识卡片用不同评估标准：完整性+时效性 |
| 清理(淘汰) | 知识卡片永不过期 | 知识卡片标记 stale，wiki 重新编译时覆盖 |

## 二、LLMWiki 方案

### 设计目标

知识不放进经验里，在 daemon 本地实现 LLMWiki，只提供本地搜索。

### 数据模型

```sql
-- 本地 SQLite
CREATE TABLE wiki (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_url TEXT,
    project TEXT NOT NULL DEFAULT 'default',
    tags TEXT,  -- JSON array
    fts FTS5(title, content),  -- 全文搜索
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 架构

- wiki_compiler 编译后写入本地 wiki 表而非 PG experiences
- daemon 新增 `/wiki/search?q=xxx` 端点
- before_prompt 时两路并行检索：经验(PG) + 知识(本地)
- 结果合并注入 prompt：经验 `[mem:xxx]`，知识 `[wiki:xxx]`

### 优势

- 经验和知识彻底分离，检索噪声降低
- 知识只存本地，不上云，隐私更好
- daemon 重启/迁移只需拷贝 SQLite 文件
- 可增量编译（只重新编译变化的源文件）

## 三、个人记忆(用户画像)模块化方案

### 设计目标

设置人设模块 (写作/调研/写代码等)，每个模块最多 100 条，根据用户意图加载对应人设到 prompt。

### 数据模型

```sql
ALTER TABLE personal_memories ADD COLUMN persona TEXT DEFAULT 'general';
ALTER TABLE personal_memories ADD COLUMN priority INTEGER DEFAULT 50;

-- persona 类型:
--   coding    : 写代码/调试/部署/测试
--   writing   : 写文档/文章/翻译
--   research  : 搜索/调研/对比/分析
--   debugging : 排障/日志分析/性能优化
--   general   : 通用偏好

-- 限制: 每个 (user_id, persona) 最多 100 条
```

### 意图识别

before_prompt 时：
1. 分析用户 prompt 关键词/意图
2. 映射到 persona (关键词匹配，后续可升级 LLM 判断)
3. 从 personal_memories 查询 (user_id, persona)
4. 按 priority 排序，取 top 100 注入 prompt

### 注入格式

```
[PERSONA: coding]
- 偏好中英双语输出
- 代码注释用中英双语
- 不喜欢太长的回复
- 偏好 pytest 而非 unittest
```

### 优势

- 精准：只加载当前场景相关的偏好
- 省 token：不加载无关人设
- 有上限：100 条限制防膨胀
- 可演进：LLM 定期整理/去重/提升 priority

## 四、多设备同步方案

### 选型

**方案 A (已选): Daemon-to-Daemon + Tailscale**

- 两台机器装 Tailscale 组网 (100.x.x.x 虚拟 IP)
- daemon 加 `/sync/changes?since=<ts>` 端点
- 定时 (每10分钟) pull 对端变更
- 本地对比 updated_at，最新的写入

### 同步端点

```
GET  /sync/status                → 返回本机最新 updated_at (各表)
GET  /sync/changes?since=<ts>    → 返回 since 之后变更的记录
POST /sync/apply                 → 接收变更记录，本地写入 (时间戳优先)
```

### 配置

```yaml
sync:
  enabled: true
  peers:
    - name: home
      url: http://100.x.x.x:3901
  interval_minutes: 10
```

### 备选方案 B: Git 仓库同步 (未选，备查)

- 私有 git 仓库 (tm-sync-data)
- daemon 定时导出变更数据为 JSON
- git add + commit + push / pull
- 优势: 不需要同时在线，天然版本历史
- 劣势: 数据过第三方，API key 需加密

## 五、实施计划

### Phase 1: 质量门控 — 精炼 prompt + 推送验证 + 数据清理

1. 重写精炼 prompt (针对技术对话场景)
2. 推送前 LLM 验证 (quality_score 门槛)
3. 清理现有垃圾数据 (测试数据/低质量 Bronze)

### Phase 2: 去重/矛盾处理

1. 精炼前去重 (检查已有经验，cosine>0.9 合并)
2. 消息级去重 (ConversationBuffer)
3. Auto Dream 定时合并/裁剪

### Phase 3: LLMWiki — 知识分离

1. 本地 SQLite wiki 表
2. wiki_compiler 改写本地
3. daemon 搜索端点
4. 现有知识卡片从 experiences 迁出

### Phase 4: 人设模块化

1. personal_memories schema 改造 (persona + priority)
2. 意图识别 (关键词→persona 映射)
3. before_prompt 按人设加载偏好
4. 提取 prompt 适配模块化

### Phase 5: 多设备同步

1. daemon sync 端点
2. Tailscale 组网
3. 定时 pull/push 逻辑
