# TeamMemory 计划完成度验证方案

> 使用者视角：通过 MCP 调用 + 代码/单测 双重验证核心路径，三角色两轮审查后再输出结论。

## 一、核心路径与验收标准（来自 plan）

| Step | 核心路径（MCP/行为） | 验收标准 |
|------|----------------------|----------|
| 1 评分闭环 | tm_search 返回含 feedback_hint；top-1 use_count+1；无评分时 final_score=similarity | 有结果时 JSON 含 feedback_hint；单测 rating_weight=0 不惩罚 |
| 2 Embedding | save/save_group 的 embed 含 tags/子 title | 单测 embed_text 包含 tags 与子 title |
| 3a/3b FTS | FTS 使用 simple；query 经 tokenizer（jieba）再查 | 单测/源码：simple、tokenize 接入 |
| 4a 同义词 | 检索前 query 按 tag_synonyms 扩展，缓存键为原 query | 单测 _expand_query_synonyms；缓存 key 未改 |
| 4b 短查询 | 短 query 使用 min_similarity_short=0.45 | 单测/配置：effective_min_similarity 传入 vector 检索 |
| 5 提取 | tm_learn → parse_content 含 few-shot、score<2 重试、as_group 说明 | 单测/源码：prompt 含示例、重试逻辑、as_group 注意 |
| 5b 管道 | 三阶段方案文档可读、可转 tm_task | 三阶段方案文档存在且含阶段说明 |
| 6a Rules | 4 个场景化 rule 文件存在且触发条件明确 | tm-core/tm-web/tm-quality/tm-plan.mdc 存在 |
| 6b 工具描述 | Server Instructions 与工具 description 增强 | server.py instructions 含 preflight、when-to-use |
| 7 文档 | README 及本次改动相关文档更新 | 文档存在且提及本次改动 |

## 二、Agent1 用例集 A（MCP + 代码）

1. **tm_preflight**：调用一次，检查返回含 complexity、search_depth、action_hint。
2. **tm_search**（有/无结果）：有结果时检查 message、results、feedback_hint；无结果时检查 message、results=[]。
3. **tm_solve**（有/无结果）：有结果时检查 feedback_hint；无结果时检查 suggestion。
4. **tm_save**：保存一条，检查返回 id、title、publish_status。
5. **tm_learn**（可选）：短文本提取，检查返回含 message、id 或 error。
6. **代码/单测**：运行 test_scoring_feedback、test_embedding_text、test_tokenizer、test_synonym_expansion；确认 repository 中 rating_weight 降级、search_pipeline 中 retrieval_query 与 effective_min_similarity。

## 三、Agent2 用例集 B（同一路径，不同用例）

1. **tm_preflight**：不同 task_description（如「修改 Web 端搜索框」），检查 complexity 可能为 medium/complex。
2. **tm_search**：不同 query（短查询如「UI」、带 tag 的查询），检查结构一致。
3. **tm_solve**：不同 problem 描述，检查 suggestion/tm_save 引导。
4. **tm_task**：action=list，检查返回结构；action=create 可选。
5. **tm_feedback**：experience_id 无效时检查错误信息；有效时检查成功消息。
6. **代码**：确认 llm_parser 中 few-shot 与 quality_retry_once、config SearchConfig 含 short_query_max_chars/min_similarity_short。

## 四、Agent3 审查要点（两轮）

- 第一轮：Agent1/2 的 MCP 调用是否覆盖上表路径；无数据时是否区分「未实现」与「无数据导致无法观测」；单测是否通过。
- 第二轮：结论是否与 plan 的 todo 状态一致；是否有遗漏（如 FTS 迁移、re_embed 脚本）；建议与待确认项是否列出。

## 五、执行与输出

- 先执行 Agent1（用例集 A），记录每步 MCP 返回与单测结果。
- 再执行 Agent2（用例集 B），记录差异与补充点。
- Agent3 执行两轮审查，修正结论后输出「完成度报告」与「待确认/建议」清单。
