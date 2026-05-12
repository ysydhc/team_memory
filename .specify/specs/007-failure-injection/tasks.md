# Tasks: Failure Injection Tests

**Plan**: plan.md
**Total tasks**: 9

## Task 1: 创建 conftest.py + shell helpers
- File(s): tests/failure_tests/conftest.py (新), tests/failure_tests/__init__.py (新), scripts/failure-tests/ (新目录)
- Action:
  1. 新建 tests/failure_tests/ 目录 + __init__.py
  2. conftest.py: daemon fixture (module scope, make daemon-start/stop, wait :3901/status, cleanup)
  3. scripts/failure-tests/ensure_daemon_ready.sh
  4. scripts/failure-tests/cleanup_test_data.sh
- Test: `pytest tests/failure_tests/ --collect-only` → 收集到 fixture
- Depends on: none
- Status: [x] done

## Task 2: P1 — PG 启停恢复 (TDD: 先写失败测试)
- File(s): tests/failure_tests/test_pg_down.py (新)
- Action: 写入完整测试代码 (PG停→发hook→查队列→PG启→查队列空→查PG有经验)
- Test: `pytest tests/failure_tests/test_pg_down.py -v` → 预期结果取决于 docker 和 daemon 是否可用
- Depends on: Task 1
- Status: [x] done

## Task 3: P3 — kill daemon mid-refining
- File(s): tests/failure_tests/test_kill_refining.py (新)
- Action: 设定 session refining → kill daemon → restart → 验证恢复日志 → session 被重新处理
- Test: `pytest tests/failure_tests/test_kill_refining.py -v`
- Depends on: Task 1
- Status: [x] done

## Task 4: P4 — kill daemon mid-publishing
- File(s): tests/failure_tests/test_kill_publishing.py (新)
- Action: 设定 session publishing (有 refined_payload) → kill → restart → skip_llm=true → 直接 publish
- Test: `pytest tests/failure_tests/test_kill_publishing.py -v`
- Depends on: Task 1, Task 3 (复用 kill/restart 逻辑)
- Status: [x] done

## Task 5: P5 — Ollama 不可用
- File(s): tests/failure_tests/test_ollama_down.py (新)
- Action: mock Ollama embedding 返回失败 → 调用 draft_save → 验证不崩溃 (返回 error dict)
- Test: `pytest tests/failure_tests/test_ollama_down.py -v`
- Depends on: Task 1
- Status: [x] done

## Task 6: P2 — LiteLLM 不可用 (mock)
- File(s): tests/failure_tests/test_llm_down.py (新)
- Action: 用 monkeypatch mock _call_llm 抛异常 → 触发 refinement → 入 pending_ops → backoff 检查
- Test: `pytest tests/failure_tests/test_llm_down.py -v`
- Depends on: Task 1
- Status: [x] done

## Task 7: P6 — 标记已有测试
- File(s): tests/test_metrics.py (修改)
- Action: test_record_and_query_multiple 加 @pytest.mark.failure_injection
- Test: `pytest tests/test_metrics.py -v -m failure_injection` → 收集到 1 test
- Depends on: none
- Status: [x] done

## Task 8: CI workflow
- File(s): .github/workflows/ci-failure-injection.yml (新)
- Action: 建 CI job: services:postgres → docker compose up → make daemon-start → pytest failure_tests/
- Test: push → CI runs (需要在 GitHub 验证)
- Depends on: Task 2-7
- Status: [x] done

## Task 9: 全量验证 + commit
- Action: ruff + pytest -q -m "not integration" + make e2e-verify + commit
- Depends on: Task 2-7
- Status: [x] done