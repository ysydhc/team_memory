---
id: TASK-8
title: Fix id=None bug in daemon context
status: In Progress
assignee: [hermes]
created_date: '2026-04-27 15:36'
labels: [bug, critical]
dependencies: [TASK-7]
---

## 问题描述
daemon 运行时 `sink.draft_save()` 返回 `{'id': None, 'status': 'draft'}`，但 CLI 直接调 `op_draft_save()` 返回正常 id。

## 已排除
- 不是 dedup 问题 (DB 0 rows for project=team_doc)
- 不是 embedding 失败 (Ollama 200 OK)
- 不是 to_dict() 问题 (确认包含 `"id": str(self.id)`)

## 可能原因
1. async session 上下文冲突 — daemon 的 FastAPI event loop 和 draft_save 的 async with get_session 可能嵌套
2. Experience ORM 对象在 session 提交前被 to_dict() 序列化
3. daemon 中某个中间层吞掉了 id

## 验证方法
1. 在 `op_draft_save` 的 `service.save()` 调用前后加 logger.info 打印 result
2. 在 `ExperienceService.save()` 的 `repo.create()` 后加 logger.info 打印 experience.id
3. 确认是否是 daemon 独有的问题 (vs 所有 async 上下文)

## 相关文件
- src/team_memory/services/memory_operations.py:914-963
- src/team_memory/services/experience.py:130-259
- src/team_memory/storage/models.py:135-162 (to_dict)
- scripts/daemon/tm_sink.py:146-164 (LocalTMSink.draft_save)
- scripts/daemon/refinement_worker.py:115-143
