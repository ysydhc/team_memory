# 阶段 4：提升 + 评估闭环

## 项目背景

阶段 1-3 完成了：
- TM 支持管线写入（draft → published → promoted）
- Hook 脚本跑通草稿→写入全流程
- Obsidian 文件通过 Git 索引进入 TM

阶段 4 要完成闭环：
1. **L2→L3 自动提升**：痕迹被反复使用后，Janitor 自动编译为 Markdown 写入 Obsidian
2. **评估生效**：[mem:xxx] 标记系统自动判定 was_used，周报告诉你系统有没有用
3. **Hermes 侧管线**：Hermes 不走 Hook，走内部逻辑

## 预计改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `services/janitor.py` | 修改 | promotion 任务完善：编译 Markdown + 写文件 |
| `services/promotion_compiler.py` | 新建 | L2 痕迹编译为结构化 Markdown |
| `services/evaluation.py` | 修改 | was_used 自动判定逻辑 |
| `scripts/hooks/weekly_report.py` | 新建 | 评估周报生成脚本 |
| `scripts/hooks/hermes_pipeline.py` | 新建 | Hermes 侧管线逻辑 |
| `~/.hermes/skills/` | 修改 | Hermes skill 集成管线 |

## 架构图

### L2→L3 提升完整流程

```
┌────────────────────────────────────────────────────────────┐
│  Janitor promotion 任务（每 24h）                            │
│                                                            │
│  1. 查询符合条件的 L2 Experience：                          │
│     · use_count >= 3                                       │
│     · 或 同 group_key >= 5 条                              │
│     · 且 exp_status = published                            │
│                                                            │
│  2. 按 group_key 聚合                                      │
│     · 同一组的 Experience 一起编译                          │
│     · 生成一篇结构化 Markdown                               │
│                                                            │
│  3. PromotionCompiler 编译                                 │
│     · LLM 把多条碎片合成为一篇知识文档                       │
│     · 模板：                                               │
│       ---                                                  │
│       title: 从经验提炼的知识                               │
│       tags: [自动提取]                                     │
│       source: tm-promotion                                 │
│       promoted_from: [exp-id1, exp-id2, ...]               │
│       ---                                                  │
│       ## 问题描述                                          │
│       ...                                                  │
│       ## 解决方案                                          │
│       ...                                                  │
│       ## 经验来源                                          │
│       - 2026-04-15: [原 Experience 摘要]                    │
│                                                            │
│  4. 写入 Obsidian 目录                                     │
│     · 根据 project 确定写入哪个仓库                         │
│     · 文件名: promoted-YYYY-MM-DD-group_key.md             │
│     · git add + git commit                                 │
│                                                            │
│  5. 标记 L2 为 promoted                                    │
│     · exp_status = "promoted"                              │
│     · 不再参与默认检索                                     │
│     · 但 full_document 在 L3 的 Obsidian 中可查             │
└────────────────────────────────────────────────────────────┘
```

### 评估闭环

```
┌────────────────────────────────────────────────────────────┐
│  自动评估流程                                               │
│                                                            │
│  检索时：                                                   │
│    SearchOrchestrator → 注入 [mem:exp-xxx] → 返回 Agent    │
│    同时写入 SearchLog（query, intent, result_ids）          │
│                                                            │
│  Agent 回复后（afterAgentResponse Hook）：                   │
│    扫描回复文本：                                           │
│    ├── 包含 [mem:exp-xxx] → was_used = True（精确匹配）    │
│    └── 不包含 → 嵌入相似度 > 0.8 → was_used = True（模糊）│
│    更新 SearchLog.was_used                                 │
│                                                            │
│  每周：                                                     │
│    weekly_report.py 生成周报：                               │
│    · 检索次数：47                                          │
│    · 命中次数：38                                          │
│    · 使用次数：22                                          │
│    · 使用率：46.8%                                         │
│    · Top 5 被使用的记忆                                     │
│    · 本周新增/提升/衰减                                     │
│    · was_used 使用率趋势                                    │
└────────────────────────────────────────────────────────────┘
```

### Hermes 侧管线

```
┌────────────────────────────────────────────────────────────┐
│  Hermes（不走 Hook，走内部逻辑）                             │
│                                                            │
│  每轮对话开始：                                             │
│    1. 自动检索 TM（调 MCP memory_recall）                   │
│    2. 检索结果注入上下文（和 memory 一样）                   │
│                                                            │
│  每轮对话结束：                                             │
│    1. 解析本轮对话（user + assistant + 工具调用）            │
│    2. 更新草稿缓冲区                                        │
│    3. 检测收敛信号                                          │
│    4. 收敛 → 调 MCP memory_draft_publish                   │
│                                                            │
│  使用记忆时：                                               │
│    明确标注"根据你之前的经验 [mem:exp-xxx]"                  │
│    → 评估系统自动标记 was_used                              │
└────────────────────────────────────────────────────────────┘
```

## 子任务拆分

---

### 任务 4-1：PromotionCompiler — L2 编译为 Markdown

**描述**：实现把多条 L2 Experience 编译为一篇结构化 Markdown 的逻辑。

**TDD 流程**：
1. 写测试：`tests/test_promotion_compiler.py`
   - 测试单条 Experience → 简单 Markdown
   - 测试多条同 group_key Experience → 合成一篇 Markdown
   - 测试编译后 Markdown 包含 frontmatter（title/tags/source/promoted_from）
   - 测试编译后 Markdown 包含原 Experience 的摘要
2. 新建 `services/promotion_compiler.py`
3. 验证测试通过

**伪代码**：
```python
PROMOTION_TEMPLATE = """---
title: {title}
tags: [{tags}]
source: tm-promotion
promoted_from: [{promoted_from}]
promoted_at: {date}
---

## 问题描述
{problem}

## 解决方案
{solution}

## 经验来源
{sources}
"""

class PromotionCompiler:
    def __init__(self, llm_config=None):
        self._llm_config = llm_config
    
    async def compile(self, experiences: list[dict], group_key: str = None) -> str:
        """编译多条 Experience 为一篇 Markdown"""
        if len(experiences) == 1:
            return self._compile_single(experiences[0])
        else:
            return await self._compile_multi(experiences, group_key)
    
    def _compile_single(self, exp: dict) -> str:
        """单条编译"""
        return PROMOTION_TEMPLATE.format(
            title=exp["title"],
            tags=", ".join(exp.get("tags", [])),
            promoted_from=exp["id"],
            date=datetime.now().strftime("%Y-%m-%d"),
            problem=exp.get("description", ""),
            solution=exp.get("solution", ""),
            sources=f"- {exp['created_at'][:10]}: {exp['title']}",
        )
    
    async def _compile_multi(self, exps: list[dict], group_key: str) -> str:
        """多条编译（可选 LLM 合成）"""
        # 如果有 LLM 配置，让 LLM 把多条碎片合成为连贯的文档
        # 否则简单拼接
        pass
```

**验收标准**：
- [ ] 单条 Experience 编译为合法 Markdown + frontmatter
- [ ] 多条 Experience 编译为一篇合成 Markdown
- [ ] frontmatter 包含 promoted_from 字段
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 4-2：Janitor promotion 完善 — 编译 + 写文件

**描述**：完善阶段 1 的 run_promotion()，加入编译 Markdown 和写入 Obsidian 目录的逻辑。

**TDD 流程**：
1. 写测试：`tests/test_janitor_promotion_v2.py`
   - 测试 promoted Experience 被编译为 Markdown
   - 测试 Markdown 文件被写入正确的 Obsidian 目录
   - 测试 git add + git commit 被执行
   - 测试 Experience 被标记为 promoted
2. 修改 `services/janitor.py`
3. 新增 `config/janitor.py` 的 promotion 输出目录配置
4. 验证测试通过

**关键修改**：

`config/janitor.py` — 新增：
```python
promotion_output_dirs: dict[str, str] = {}
"""project → Obsidian 输出目录映射
例: {"ad_learning": "/Users/yeshouyou/Work/ad_learning/docs/promoted"}
"""
```

`services/janitor.py` — run_promotion 完善：
```python
async def run_promotion(self, project=None):
    # ... 查找符合条件的 Experience（阶段 1 已实现）
    
    # 新增：编译 + 写文件
    compiler = PromotionCompiler(self._llm_config)
    output_dirs = self._get_config("promotion_output_dirs", {})
    
    for group_key, exps in groups.items():
        # 编译
        markdown = await compiler.compile(exps, group_key)
        
        # 写文件
        proj = exps[0].project
        output_dir = output_dirs.get(proj)
        if output_dir:
            filename = f"promoted-{datetime.now().strftime('%Y-%m-%d')}-{group_key or 'misc'}.md"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write(markdown)
            
            # git add + commit
            subprocess.run(["git", "add", filepath], cwd=output_dir)
            subprocess.run(["git", "commit", "-m", f"promoted: {group_key or filename}"], cwd=output_dir)
        
        # 标记 promoted
        for exp in exps:
            exp.exp_status = "promoted"
```

**验收标准**：
- [ ] promoted Experience 被编译为 Markdown 文件
- [ ] Markdown 文件出现在配置的 Obsidian 目录中
- [ ] git add + commit 被执行
- [ ] Experience 被标记为 promoted
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 4-3：was_used 自动判定 + 周报

**描述**：实现 was_used 的精确匹配和模糊匹配，以及周报生成。

**TDD 流程**：
1. 写测试：`tests/test_evaluation_was_used.py`
   - Agent 回复包含 [mem:exp-123] → was_used=True
   - Agent 回复不包含标记但语义相似 → was_used=True（模糊匹配）
   - Agent 回复完全不相关 → was_used=False
2. 完善 `services/evaluation.py`
3. 新建 `scripts/hooks/weekly_report.py`
4. 验证测试通过

**weekly_report.py 伪代码**：
```python
async def generate_weekly_report():
    """生成记忆系统周报"""
    stats = await search_log_repo.get_stats(days=7)
    
    report = f"""
    记忆系统周报 ({monday} ~ {sunday})
    ══════════════════════════
    
    检索：{stats['total']} 次
    命中：{stats['hit']} 次（有 score>阈值的结果）
    使用：{stats['used']} 次（Agent 实际引用）
    使用率：{stats['use_rate']:.1%}
    
    本周新增：{new_count} 条痕迹
    本周提升：{promoted_count} 条 → 知识
    本周衰减：{decayed_count} 条 → 低分
    
    Top 5 被使用的记忆：
    {top_5_used}
    
    评估：{'系统有效 ✓' if stats['use_rate'] > 0.4 else '需要调整 ⚠' if stats['use_rate'] > 0.2 else '系统无效 ✗'}
    """
    
    return report
```

**验收标准**：
- [ ] 精确匹配：[mem:xxx] 在 Agent 回复中 → was_used=True
- [ ] 模糊匹配：语义相似度 > 0.8 → was_used=True
- [ ] 周报包含检索次数、使用率、Top 5 被使用的记忆
- [ ] 使用率 > 40% 时显示"系统有效"
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 4-4：Hermes 侧管线集成

**描述**：让 Hermes 通过 skill 机制集成管线逻辑，不走外部 Hook。

**TDD 流程**：
1. 设计 Hermes skill：`memory-pipeline`
2. 实现自动检索：每轮对话开始时调 MCP
3. 实现草稿缓冲：每轮结束时更新草稿
4. 实现收敛检测和写入
5. 手动验证

**Hermes skill 核心逻辑**：
```
skill: memory-pipeline
trigger: 每轮对话自动执行

步骤：
1. 解析当前上下文（项目路径 + 用户消息）
2. 调 TM MCP memory_recall 或 memory_context
3. 把检索结果注入上下文
4. 每轮结束后：
   a. 解析本轮对话
   b. 更新草稿缓冲区
   c. 检测收敛信号
   d. 收敛 → 调 memory_draft_publish
5. 使用记忆时标注 "根据你之前的经验 [mem:xxx]"
```

**验收标准**：
- [ ] Hermes 每轮对话自动检索 TM
- [ ] 检索结果自然融入回复
- [ ] 对话中的发现被写入草稿
- [ ] 收敛后草稿被提炼
- [ ] 使用记忆时标注 [mem:xxx]

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 4-5：afterAgentResponse Hook 集成 was_used 判定

**描述**：在 Cursor/Claude Code 的 afterAgentResponse Hook 中加入 was_used 自动判定。

**TDD 流程**：
1. 写测试：`tests/test_hook_was_used.py`
   - Hook 收到 Agent 回复后，调 evaluation service 判定 was_used
   - 精确匹配：[mem:xxx] 在回复中
   - 模糊匹配：语义相似度
2. 修改 `scripts/hooks/cursor_after_response.py`
3. 验证测试通过

**验收标准**：
- [ ] Agent 回复包含 [mem:xxx] 时，SearchLog.was_used 被更新为 True
- [ ] 不包含标记时，模糊匹配判定 was_used
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 4-6：全系统端到端验证

**描述**：从写入到检索到提升到评估，跑通完整闭环。

**验证流程**：
1. 在 Hermes 中进行一段调试对话
2. 确认草稿被写入 TM
3. 确认下次对话能检索到
4. 反复使用同一条记忆 3 次
5. 等待 Janitor 运行
6. 确认记忆被提升为 Markdown
7. 确认 Obsidian 目录中有新文件
8. 生成周报，确认使用率 > 0

**验收标准**：
- [ ] Hermes/Cursor → 草稿 → TM → 检索命中 → [mem:xxx] 标注 → was_used 判定
- [ ] use_count≥3 → Janitor promotion → Markdown 文件 → Obsidian
- [ ] 周报显示非零使用率
- [ ] Obsidian 文件通过 Git 索引回 TM（循环闭环）

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

## 阶段 4 人工验收条目

- [ ] 在 Hermes 中调试一个问题，确认草稿被写入 TM（source=pipeline）
- [ ] 在 Cursor 中问相关的问题，确认检索到了 Hermes 写入的记忆
- [ ] Agent 使用记忆时标注了"根据你之前的经验 [mem:xxx]"
- [ ] 反复使用同一条记忆 3 次后，手动调 Janitor，确认记忆被提升
- [ ] Obsidian 目录中出现了 promoted-xxx.md 文件
- [ ] 该文件被 git commit，且在 TM 中能被检索到
- [ ] 运行周报脚本，确认有数据（检索次数、使用率）
- [ ] 全流程闭环：写入 → 检索 → 使用 → 提升 → Obsidian → Git 索引 → 再次检索
