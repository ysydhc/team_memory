# Implementation Plan: Pipeline Acceptance Test System

**Spec**: spec.md
**Status**: Ready for review

## Architecture

### 测试策略: 分层验证

```
层级 1: 单元测试 (已有，通过)
  每个模块独立测试，mock 上下游

层级 2: 集成测试 (本次新增)
  真实 daemon + 真实 SQLite + 真实 PostgreSQL
  不 mock 模块间调用

层级 3: 端到端测试 (本次新增)
  模拟真实 hook payload
  验证完整数据流
```

### 涉及模块

| 模块 | 文件 | 验证点 |
|------|------|--------|
| HookPayload | daemon/app.py | 接受各平台格式 |
| pipeline | daemon/pipeline.py | 消息存储 + 收敛检测 |
| ConversationBuffer | daemon/conversation_buffer.py | SQLite 读写 |
| RefinementWorker | daemon/refinement_worker.py | LLM 调用 + 发布 |
| SearchLogWriter | daemon/search_log_writer.py | 日志写入 |
| EvaluationService | services/evaluation.py | 标记注入 + was_used |
| FaithfulnessJudge | daemon/faithfulness_judge.py | LLM 评分 |
| ResponseBuffer | daemon/response_buffer.py | 批量触发 |

## Implementation Steps

### Step 1: 创建测试基础设施
- 文件: tests/test_pipeline_acceptance.py
- 内容: fixture 启动真实 daemon (测试用 SQLite)
- 验证: daemon 启动 + health check

### Step 2: Hook Payload 验证
- 文件: tests/test_pipeline_acceptance.py
- 内容: 测试各平台 payload 格式都能被接受
- 验证: Cursor/Claude/Hermes 格式都不报错

### Step 3: 消息存储验证
- 文件: tests/test_pipeline_acceptance.py
- 内容: 发送 before_prompt + after_response，检查 ConversationBuffer
- 验证: user + assistant 消息都入库

### Step 4: 收敛检测验证
- 文件: tests/test_pipeline_acceptance.py
- 内容: 发送 3 条 assistant 消息，检查 session 状态
- 验证: session 变为 needs_refinement

### Step 5: RefinementWorker 验证
- 文件: tests/test_pipeline_acceptance.py
- 内容: 手动触发 _tick()，检查 PostgreSQL
- 验证: 经验入库 + 字段正确
- 风险: 依赖 LLM 可用，需 mock 或跳过

### Step 6: 检索验证
- 文件: tests/test_pipeline_acceptance.py
- 内容: 用 tm-cli recall 搜索刚存入的经验
- 验证: 结果包含新经验 + project 过滤正确

### Step 7: 评估系统验证
- 文件: tests/test_pipeline_acceptance.py
- 内容: 检查 search_log + fuzzy_match
- 验证: 日志写入 + was_used 检测
- 风险: faithfulness judge 需要 LLM，可选验证

## Test Strategy

### 必须通过 (P0)
- Hook payload 接受各平台格式
- 消息存储到 ConversationBuffer
- 收敛检测触发 (3+ 消息)
- 搜索返回结果

### 应该通过 (P1)
- RefinementWorker 处理 + PostgreSQL 写入
- search_log 写入
- fuzzy_match 检测

### 可选验证 (P2)
- faithfulness judge LLM 调用
- personal memory extraction

## Risks

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM 不可用 | RefinementWorker 失败 | mock LLM 或标记跳过 |
| Ollama 不可用 | embedding 失败 | mock embedding 或标记跳过 |
| Cursor payload 格式变化 | before_prompt 拿不到用户消息 | PayloadCapture 记录 + Lenient 模型 |

## Output
- tests/test_pipeline_acceptance.py
- make test-acceptance target
