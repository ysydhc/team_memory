# Feature Specification: Failure Injection Tests

**Created**: 2026-05-14
**Status**: Draft
**Input**: 系统性故障注入测试，验证 PG/LiteLLM/Ollama/daemon/磁盘满等场景下数据不丢失

## 设计决策

| 决策 | 选择 |
|------|------|
| 实现方式 | pytest + shell helper (混合方案 C) |
| 测试粒度 | 一个场景一个文件 |
| daemon 生命周期 | conftest.py fixture 管理 daemon 启停 |
| docker 管理 | subprocess.run(["docker","compose","stop/start"]) |
| CI 集成 | 加 docker compose job, 完整跑 |

## User Scenarios

### P1: PG 不可用 → 数据入队列 → 恢复后补写
- As 系统
- I want to PG 挂了不丢数据
- Acceptance Criteria:
  1. docker compose stop postgres
  2. 发 3 轮 hook → 数据写入 ConversationBuffer + convergence 触发
  3. RefinementWorker 尝试 publish → 失败 → 入 pending_ops
  4. tm-cli queue stats → 显示 pending > 0
  5. docker compose start postgres → 等待 retry
  6. tm-cli queue stats → 显示 0 pending
  7. PG 有经验入库

### P2: LiteLLM 不可用 → backoff 重试 → 最终 fallback
- As 系统
- I want to LLM 429 时不丢数据
- Acceptance Criteria:
  1. 停 LiteLLM (或 mock 429)
  2. 触发 refinement → _call_llm 失败 → 入 pending_ops
  3. check_all → backoff 退避 (10min 窗口)
  4. 启用 LiteLLM → 重试成功 → 经验发布

### P3: kill daemon mid-refinement → 重启恢复
- As 系统
- I want to daemon 被 kill 时不丢数据
- Acceptance Criteria:
  1. session 进入 refining 状态
  2. kill daemon (SIGTERM)
  3. 重启 daemon → 日志显示 "Reset N refining sessions"
  4. session 回到 needs_refinement → 重新精炼成功

### P4: kill daemon mid-publishing → skip_llm 生效
- As 系统
- I want to daemon 在 publishing 期间被 kill → 不重复调 LLM
- Acceptance Criteria:
  1. session 进入 publishing 状态 (refined_payload 已存储)
  2. kill daemon
  3. 重启 → 日志 "Reset M publishing sessions"
  4. skip_llm=true → 直接 draft_publish → 不调 _call_llm

### P5: Ollama 不可用 → embedding 失败 → 降级处理
- As 系统
- I want to Ollama 挂了也能保存经验
- Acceptance Criteria:
  1. 停 Ollama → embedding 不可用
  2. 经验保存 → 成功 (可能无向量)

### P6: 磁盘满 → record() 不崩溃
- As 系统
- I want to 磁盘满了 metrics record 不抛异常
- Acceptance Criteria:
  1. mock 磁盘满 → metrics.record() 不崩溃 → 日志有 error

## Technical Requirements

### 文件结构
```
tests/failure_tests/
  __init__.py
  conftest.py              # daemon fixture (module scope)
  test_pg_down.py          # P1 + shell helper: docker compose
  test_llm_down.py         # P2
  test_ollama_down.py      # P5
  test_kill_refining.py    # P3
  test_kill_publishing.py  # P4
  test_disk_full.py        # P6 (单元测试, 不需要 daemon)

.github/workflows/
  ci-failure-injection.yml # 新 CI job
```

### conftest.py fixture
```python
@pytest.fixture(scope="module")
def daemon():
    # 1. make daemon-start
    # 2. wait for :3901/status → 200
    # 3. yield
    # 4. make daemon-stop
    # 5. cleanup test data in SQLite + PG
```

### 每个测试的模板
```python
class TestPgDown:
    @pytest.mark.failure_injection
    def test_pg_down_data_queued_and_recovered(self, daemon):
        # 1. docker compose stop postgres
        # 2. send 3 rounds of hook → convergence
        # 3. assert: tm-cli queue stats → pending > 0
        # 4. docker compose start postgres; sleep 45
        # 5. assert: tm-cli queue stats → pending = 0
        # 6. assert: PG has new experience
```

### CI workflow
```yaml
ci-failure-injection:
  runs-on: ubuntu-latest
  services:
    postgres: ...
  steps:
    - docker compose up -d postgres
    - make daemon-start
    - pytest tests/failure_tests/ -v -m failure_injection
```

### Shell helpers (scripts/failure-tests/)
- `ensure_daemon_ready.sh`: wait for :3901/status
- `cleanup_test_data.sh`: DELETE test sessions from SQLite + PG

## Edge Cases
- daemon 启动失败 → fixture timeout → pytest skip
- docker compose 不在 PATH → 跳过需 docker 的测试
- 测试间数据隔离 → 每个测试用唯一 session_id

## Out of Scope
- Ollama 实际启停 (太重，用 mock)
- 网络分区测试
- 多实例并发故障
