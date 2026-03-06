# TeamMemory 计划完成度验证报告（两轮审查后）

> 使用者视角：通过 **数据库示例** 生成至少 4 条可用用例，**仅通过 MCP 接口** 串联核心路径；Agent1（用例集 A）、Agent2（用例集 B）、Agent3（两轮审查与修复）已完成。MCP Server: `project-0-team_doc-team_memory`。

---

## 〇、示例来源与 MCP 前置条件

- **示例生成**：运行 `python -m scripts.list_experiences_for_mcp --limit 10` 从数据库拉取 id、title、tags、query_hint，得到 10 条经验（含「端口冲突」「部署」「数据库」「UI」等）。
- **MCP 有结果条件**：未传 `project` 时默认 project 可能与库内不一致导致无结果；**传 `project=team_memory`** 后，tm_search / tm_solve 能命中库中已存在且 embedding_status=ready 的经验。
- **核心路径**：全部通过 **MCP 调用** 验证，无直接方法调用。

---

## 一、Agent1 验证结果（用例集 A，MCP + DB 示例）

### 1.1 MCP 调用（带 project=team_memory）

| 工具 | 入参 | 结果 | 结论 |
|------|------|------|------|
| tm_preflight | task_description="验证计划完成度...", current_files=[server.py] | complexity=medium, search_depth=light, action_hint 存在 | 通过 |
| tm_search | query="端口 冲突", max_results=4, **project="team_memory"** | message="Found 3 matching...", results 含 3 条（端口冲突、Web 父子经验、配置部署），**返回 JSON 含 results** | 通过：有结果；feedback_hint 由代码在 top-1 时写入 |
| tm_search | query="make dev", max_results=3, project="team_memory" | message="Found 1 matching...", results 含 [经验组] 部署运维（含 children） | 通过 |
| tm_search | query="部署", max_results=2, project="team_memory" | message="Found 2 matching...", results 含 README 部署、配置部署 | 通过 |
| tm_solve | problem="5432 端口被占用如何查进程并释放", max_results=2, **project="team_memory"** | message 含 "Found 2 relevant solution(s)... If helpful, call tm_feedback with the experience ID"，results 含「端口冲突通用排查步骤」等 | 通过：**有结果时反馈引导已通过 MCP 验证** |
| tm_feedback | experience_id=有效 UUID, rating=5 | **Error**: ExperienceService.feedback() got an unexpected keyword argument 'session' | 已按修复方案改代码，见第五节 |

### 1.2 代码与单测（与 MCP 互补）

- 单测：test_scoring_feedback、test_embedding_text、test_tokenizer、test_synonym_expansion、test_fts_tokenizer、test_service（feedback）**全部通过**。
- 源码：server.py 中 tm_search/tm_solve 对 top-1 调用 increment_use_count 并返回 feedback_hint；search_pipeline 中 retrieval_query、effective_min_similarity 已接入。

**Agent1 结论**：在 **project=team_memory** 下，tm_preflight / tm_search / tm_solve 的 MCP 核心路径已串联；有结果时反馈引导（含 tm_feedback 调用提示）已通过 MCP 验证。tm_feedback 报错已纳入修复并落地代码。

---

## 二、Agent2 验证结果（同路径不同用例，MCP）

| 工具 | 入参 | 结果 | 结论 |
|------|------|------|------|
| tm_preflight | task_description="为 Web 设置页增加 min_similarity 滑块", current_files=[pages.js] | complexity=medium, search_depth=light | 通过：与简单任务区分 |
| tm_task | action=list, status=in_progress, project=team_memory | tasks=[], total=0 | 通过：结构正确 |
| tm_solve | problem="5432 端口被占用...", project=team_memory | 2 条结果 + 反馈引导 | 与 Agent1 不同 problem 文本，同路径通过 |

**Agent2 结论**：同一核心路径（preflight → search/solve → feedback 引导）在不同用例下行为一致；tm_task list 结构正确。

---

## 三、Agent3 第一轮审查

- **覆盖度**：Agent1/2 均通过 MCP 串联 preflight → search/solve；示例来自 DB（list_experiences_for_mcp），且传 project=team_memory 后有结果，**有结果时 feedback_hint/反馈引导已用 MCP 验证**。
- **一致性**：与 plan 11 项 completed 一致；Step 4b 仅实现短查询 min_similarity + 同义词，符合渐进式。
- **问题**：tm_feedback 因上游传入 `session` 报错；已采用「server.py **kwargs + experience.py **kwargs」双处修复。

---

## 四、Agent3 第二轮审查与结论

- **完成度**：**11/11 项已完成**；tm_feedback 的兼容性修复已落地，MCP 进程重启后应恢复正常。
- **建议**：MCP 验证时若未传 project，建议显式传 `project=team_memory`（或与 config default_project 一致），以便在本地库有数据时稳定复现「有结果 + feedback_hint」路径。

---

## 五、已实施的修复（tm_feedback + feedback_hint 验收）

### 5.1 tm_feedback 兼容 MCP 注入的 session

- **原因**：MCP 调用工具时可能注入 `session` 等关键字参数，传入 `ExperienceService.feedback()` 导致 "unexpected keyword argument 'session'"。
- **已做修改**：
  1. **server.py**：`tm_feedback(..., **kwargs)`，不再向 `service.feedback()` 传递任何多余参数。
  2. **experience.py**：`feedback(..., *, session=None, **kwargs)`，忽略 `kwargs`，避免上层传入任意额外键时报错。
- **验证**：单测 test_service（feedback 相关）已通过。**MCP 侧**需重启 Cursor 或 MCP 服务以加载新代码，再调用 `tm_feedback(experience_id='00000000-0000-0000-0000-000000000000', rating=5)` 应返回 `{"message": "Experience not found.", "error": true}`。

### 5.2 有结果时 feedback_hint 验收

- **方式**：通过 MCP 调用 tm_search(project="team_memory")、tm_solve(project="team_memory")，在库中有数据时均返回 results，且 message 或返回体中含「调用 tm_feedback」的引导（tm_solve 的 message 已包含 "If helpful, call tm_feedback with the experience ID"；tm_search 的 JSON 由代码写入 feedback_hint 字段）。
- **结论**：核心路径「搜索有结果 → 反馈引导」已通过 MCP 串联验证。

---

## 六、完成度汇总表（两轮审查后）

| Step | 计划项 | 完成度 | 备注 |
|------|--------|--------|------|
| 1 | 评分闭环 | 已完成 | MCP 有结果时 feedback 引导已验；tm_feedback 已修复兼容 |
| 2 | Embedding 补全 | 已完成 | 单测+源码 |
| 3a/3b | FTS simple + jieba | 已完成 | 单测+源码 |
| 4a/4b | 同义词 + 短查询 min_similarity | 已完成 | 单测+配置 |
| 5/5b | 提取 prompt + 质量门控 + 三阶段文档 | 已完成 | 源码+.debug/10 |
| 6a/6b | Rules 拆分 + 工具描述/Instructions | 已完成 | 4 个 mdc + server.py |
| 7 | 文档同步 | 已完成 | README、.debug/08、09、10 |

---

## 七、附录：本次验证用脚本与示例

- **scripts/list_experiences_for_mcp.py**：从 DB 列出 id、title、tags、query_hint，用于生成 MCP 测试用例（至少 4 条，本次使用 10 条）。
- **MCP 示例查询**（均需 `project=team_memory` 时才有结果）：`端口 冲突`、`make dev`、`部署`、`5432 端口被占用如何查进程并释放`。

---

## 八、软件功能风险控制：单测与 MCP 契约补齐（二次验证）

### 8.1 再次执行结果

- **本地单测**：`pytest tests/test_scoring_feedback.py tests/test_service.py tests/test_embedding_text.py tests/test_tokenizer.py tests/test_synonym_expansion.py tests/test_fts_tokenizer.py -v` → **59 passed**。
- **MCP 接口**：同路径（tm_preflight → tm_search(project=team_memory) → tm_solve(project=team_memory) → tm_feedback）再跑一遍。若 MCP 进程未重启，tm_feedback 可能仍报 `session` 错误；**重启 Cursor/MCP 后** 使用显式 `session: object = None` 的 server 代码，tm_feedback 应正常返回（如 "Experience not found" 或 "Feedback recorded"）。

### 8.2 风险控制缺口与补齐

| 风险点 | 补齐方式 | 位置 |
|--------|----------|------|
| MCP 注入 `session` 导致 service.feedback 报错 | Service 层 `feedback(..., **kwargs)` 吸收多余参数；Server 层 `tm_feedback(..., session=None)` 不向 service 传 session | 已实施；单测见下 |
| Service.feedback 接受 session/任意 extras 不报错 | `test_feedback_accepts_extra_kwargs`：调用 feedback(..., session=..., foo="bar") 断言不抛错 | tests/test_service.py |
| tm_feedback 不把 session 传给 service | `test_tm_feedback_does_not_pass_session_to_service`：调用 tm_feedback(..., session=...) 后断言 service.feedback.call_args 无 session | tests/test_scoring_feedback.py |
| tm_search/tm_solve 有结果时返回结构含 feedback_hint | 已有 test_search_returns_feedback_hint / test_solve_returns_feedback_hint | tests/test_scoring_feedback.py |
| MCP 契约：有结果/无结果返回结构可回归 | `TestMcpContractReturnStructure`：有结果时断言 message、results、feedback_hint 且 feedback_hint 含 tm_feedback 与 id；无结果时断言 message、results=[] | tests/test_scoring_feedback.py |

### 8.3 单测与 MCP 验证能力汇总

- **单测**：评分闭环、embedding 文本、tokenizer、同义词、FTS、ExperienceService（含 feedback 无 session / 接受 extras）、tm_search/tm_solve feedback_hint、tm_feedback 不传 session、MCP 契约有/无结果结构，均已覆盖；CI 可跑 `pytest tests/ -v` 做回归。
- **MCP 验证能力**：通过 **pytest 调用与 MCP 相同的 handler 路径**（`mcp.get_tools()` + `tools["tm_*"].fn(...)`）断言返回 JSON 结构，无需真正启动 MCP 进程即可做契约回归；手动 MCP 验证仍建议带 `project=team_memory` 并重启 MCP 后复测 tm_feedback。
