---
id: TASK-7
title: RefinementWorker LLM精炼
status: Agent Done
assignee: [hermes]
created_date: '2026-04-27 15:36'
labels: [pipeline, llm]
dependencies: [TASK-1, TASK-4]
---

## 目标
收敛草稿由 qwen-plus 后台精炼，抽取 title/problem/solution/tags/experience_type，更新已发布 Experience。

## 完成内容
- RefinementWorker 类 (scripts/daemon/refinement_worker.py)
- qwen-plus 集成 (src/team_memory/services/llm_parser.py: call_llm_openai + parse_content_openai)
- config.yaml refinement 段 (scripts/hooks/config.yaml)
- RefinementSettings dataclass (scripts/daemon/config.py)
- app.py lifespan 注册 + graceful shutdown
- draft_buffer 新增 mark_needs_refinement / get_needs_refinement
- pipeline.py 收敛后调 mark_for_refinement (不再直接发布)
- tm_sink 新增 update_experience 方法
- memory_operations 新增 op_experience_update + dedup 处理

## 验证步骤
1. 启动 daemon: `PYTHONPATH=src:scripts python -m daemon`
2. 触发收敛: `curl -X POST localhost:3901/after_response -d '...'`
3. 检查日志: `draft_save result` 是否有 id
4. 检查 qwen-plus 是否被调用 (LiteLLM proxy 日志)
5. 检查 Experience 记录是否被更新了 title/problem/solution

## 已知问题
- **id=None bug**: daemon 上下文 draft_save 返回 {'id': None}
  - CLI 直接调 op_draft_save 正常 (返回有效 id)
  - daemon 内调 sink.draft_save 异常
  - 疑似 async session 上下文或 event loop 问题
  - 阻塞了整个精炼 pipeline 的 e2e 验证

## 相关文件
- scripts/daemon/refinement_worker.py
- scripts/daemon/app.py
- scripts/daemon/config.py
- scripts/hooks/config.yaml
- scripts/hooks/draft_buffer.py
- scripts/daemon/draft_buffer.py
- scripts/daemon/pipeline.py
- scripts/daemon/tm_sink.py
- src/team_memory/services/llm_parser.py
- src/team_memory/services/memory_operations.py
