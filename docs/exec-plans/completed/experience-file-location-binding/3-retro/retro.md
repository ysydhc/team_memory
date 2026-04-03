# 经验文件位置绑定 — 项目完成度与质量总结

> Plan: [1-plan/plan.md](../1-plan/plan.md)  
> 执行记录: [2-plan/execute.md](../2-plan/execute.md)

## 需求覆盖

| 需求 | 状态 |
|------|------|
| 经验与文件位置（路径+行范围）绑定，内容指纹稳定匹配 | ✅ 已实现 |
| 可配置 TTL（默认 30 天）、访问时刷新 | ✅ 已实现 |
| 检索中 location_weight 融合位置得分（批量查询+内存，禁止 N×M） | ✅ 已实现 |
| 过期绑定定时清理、可观测性日志 | ✅ 已实现 |
| Web/MCP 写入 file_locations、检索传入 current_file_locations | ✅ 已实现 |
| 配置 API 与 Web 设置（location_weight、TTL、refresh、cleanup） | ✅ 已实现 |
| 文档与 MCP 工具列表更新、E2E 量化验收 | ✅ 已实现 |

## 交付物清单

- **配置**：SearchConfig.location_weight、FileLocationBindingConfig（ttl_days、refresh_on_access、cleanup_enabled、cleanup_interval_hours）
- **存储**：ExperienceFileLocation 模型与迁移 6ab06751f40e、replace/get/list_bindings_by_paths/find/delete_expired/refresh_file_location_bindings
- **工具**：utils/location_fingerprint.py（指纹、重叠、常量导出）
- **服务**：ExperienceService.save/update 接受 file_locations；SearchPipeline 中 _apply_location_boost（批量+内存）
- **MCP**：tm_save/tm_save_typed 的 file_locations；tm_search/tm_solve 的 current_file_locations
- **Web**：API schema、创建/更新/详情 file_locations；设置项 location_weight 与文件位置绑定配置；录入与展示 UI
- **运维**：bootstrap 中 file_location 清理循环与日志
- **文档**：README、MCP 运维文档参数说明（当时另有一份设计稿；现已并入根 README / design-docs）
- **测试**：test_config、test_repository_file_locations、test_location_fingerprint、test_service、test_server、test_search_pipeline、test_web、test_bootstrap、test_integration（E2E）

## 测试结果

- **本计划相关**：配置/仓储/指纹/服务/MCP/管道/Web/清理/E2E 等用例均通过；在无 PostgreSQL 时部分集成与 E2E 被 skip（与项目惯例一致）。
- **既有失败**：make test 中仍有与本计划无关的失败（如 test_logging_json、test_task_group_completed 的 replace_architecture_bindings）；make lint 存在既有问题（__init__.py E501、server.py E402/I001）。按执行基线决策未在本计划内修复。

## 规格合规结论

**✅ Spec compliant**（由 spec-reviewer 逐项读代码核对）

- 批量查询无 N×M；location_score 三档与常量单点；TTL/refresh/cleanup 与 Plan 一致。
- 可选优化：设计文档中 current_file_locations 表可将「snippet」改为或补充「file_content（用于指纹重锚）」以与 API 一致。

## 遗留问题与建议

1. **既有 lint/test**：建议单独排期修复 __init__.py、server.py 的 lint 与 test_logging_json、test_task_group_completed 的失败，避免影响后续 MR。
2. **设计文档**：采纳 spec-reviewer 建议，在设计文档中统一 current_file_locations 字段表述为 file_content（可选）。
3. **E2E 环境**：有 PostgreSQL 时运行 `pytest tests/test_integration.py::TestServiceIntegration::test_file_locations_e2e_location_score_and_final_score` 可做完整 E2E 验证。
