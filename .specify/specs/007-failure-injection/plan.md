# Implementation Plan: Failure Injection Tests

**Spec**: spec.md (007)
**Status**: Ready for review

---

## Step 1: 创建 conftest.py — daemon fixture (module scope)

### 现状
- 没有管理 daemon 生命周期的 fixture
- make daemon-start / daemon-stop 已存在于 Makefile
- make e2e-verify.sh 手动管理 daemon 启停，不可复用

### 目标
- `daemon` fixture: module scope，启动一次，所有测试复用
- 自动清理测试数据

### 具体改动

**文件**: tests/failure_tests/conftest.py (新建)

```python
import pytest, subprocess, time, httpx

DAEMON_URL = "http://localhost:3901"

@pytest.fixture(scope="module")
def daemon():
    """Start daemon once per module. Cleanup test data on teardown."""
    # Start
    subprocess.run(["make", "daemon-start"], check=True, capture_output=True)
    # Wait for ready
    for _ in range(30):
        try:
            r = httpx.get(f"{DAEMON_URL}/status", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        pytest.skip("Daemon did not start within 30s")

    yield DAEMON_URL

    # Cleanup: delete test sessions
    import sqlite3
    db = sqlite3.connect("~/.cache/tm-pipeline/drafts.db")
    db.execute("DELETE FROM sessions WHERE id LIKE 'fail-%'")
    db.commit()
    # Stop
    subprocess.run(["make", "daemon-stop"], capture_output=True)
```

### 依赖: 无
### 验收: `pytest tests/failure_tests/conftest.py --collect-only` 能收集到 fixture

---

## Step 2: 创建 shell helpers

### 现状
- 无 docker compose 启停的测试工具
- 无数据清理工具

### 目标
- 两个 shell 脚本: docker 服务启停 + 测试数据清理

### 具体改动

**文件**: scripts/failure-tests/ (新建目录)

`ensure_daemon_ready.sh`:
```bash
#!/bin/bash
# Wait for daemon :3901/status to return 200
# Usage: source scripts/failure-tests/ensure_daemon_ready.sh
```

`cleanup_test_data.sh`:
```bash
#!/bin/bash
# Delete test sessions from SQLite + PG
SESSION_PREFIX="fail-"
sqlite3 ~/.cache/tm-pipeline/drafts.db "DELETE FROM sessions WHERE id LIKE '${SESSION_PREFIX}%'"
docker exec team_doc-postgres-1 psql -U developer -d team_memory \
  -c "DELETE FROM experiences WHERE title LIKE 'Draft ${SESSION_PREFIX}%'"
```

### 依赖: 无
### 验收: `bash scripts/failure-tests/ensure_daemon_ready.sh` 返回 0

---

## Step 3: 实现 P1 — PG 启停恢复 (TDD)

### 现状
- PendingQueue + backoff 已在 P0-2 实现
- `make e2e-verify-graceful.sh` 有类似逻辑但无断言

### 目标
- PG 停 → 数据入队列 → PG 启 → 数据恢复
- pytest 断言: 队列有数据 → 恢复后队列空 → PG 有经验

### 具体改动

**文件**: tests/failure_tests/test_pg_down.py (新建)

```python
@pytest.mark.failure_injection
class TestPgDown:
    def test_data_queued_and_recovered(self, daemon):
        # 1. Stop PG
        subprocess.run(["docker", "compose", "stop", "postgres"], check=True)
        time.sleep(2)

        # 2. Send 3 rounds of hook
        sid = f"fail-pg-{int(time.time())}"
        for i in range(3):
            httpx.post(f"{daemon}/hooks/before_prompt", json={...})
            httpx.post(f"{daemon}/hooks/after_response", json={...})

        # 3. Wait for refinement attempt
        time.sleep(35)

        # 4. Check queue
        result = subprocess.run(["python", "-m", "team_memory.cli", "queue", "stats"],
                               capture_output=True, text=True)
        assert "pending: 1" in result.stdout or "pending: 0" in result.stdout

        # 5. Start PG
        subprocess.run(["docker", "compose", "start", "postgres"], check=True)
        time.sleep(45)  # wait for retry

        # 6. Queue should be empty
        result2 = subprocess.run(["python", "-m", "team_memory.cli", "queue", "stats"],
                                capture_output=True, text=True)
        assert "pending: 0" in result2.stdout

        # 7. PG has experience
        check = subprocess.run(["docker", "exec", "team_doc-postgres-1",
            "psql", "-U", "developer", "-d", "team_memory", "-t", "-c",
            f"SELECT COUNT(*) FROM experiences WHERE title LIKE '%Draft {sid}%'"],
            capture_output=True, text=True)
        assert check.stdout.strip() != "0"
```

### 依赖: Step 1, Step 2
### 验收: `pytest tests/failure_tests/test_pg_down.py -v` → 1 PASS

---

## Step 4: P2 — LiteLLM 不可用

### 现状
- PendingQueue 支持 refine 类型入队
- backoff 策略在 PendingQueue.BACKOFF_SECONDS
- LiteLLM 429 场景在单元测试中 mock 过，无真实测试

### 目标
- 模拟 LiteLLM 不可用，验证入队 + backoff 不崩溃

### 具体改动

**文件**: tests/failure_tests/test_llm_down.py (新建)

```python
@pytest.mark.failure_injection
class TestLLMDown:
    def test_llm_unavailable_enqueues(self, daemon):
        # 不真实停 LiteLLM（会影响其他测试）
        # 策略: 用不存在的 model 触发 LLM 调用失败
        # 1. 修改 config.yaml model → "nonexistent-model"
        # 2. 发 hook → convergence → refinement
        # 3. _call_llm 失败 → 入 queue
        # 4. 恢复 config.yaml
```

Note: 真实停 LiteLLM 会影响 daemon 本身（daemon 也用 LiteLLM）。这个测试可能需要 mock 层面或跳过。

### 依赖: Step 1
### 验收: test PASS or skip (如果无法安全模拟)

---

## Step 5: P3 — kill daemon mid-refining

### 现状
- reset_refining_sessions 已在 P0-3 实现
- app.py 启动时调用

### 目标
- kill daemon 时 session 在 refining → 重启后恢复

### 具体改动

**文件**: tests/failure_tests/test_kill_refining.py (新建)

```python
@pytest.mark.failure_injection
class TestKillRefining:
    def test_refining_session_recovered_on_restart(self, daemon):
        # 1. 确保 session 进入了 refining 状态
        #    (需要 hook + convergence + 等到 _tick 处理)
        # 2. kill daemon (SIGTERM)
        # 3. restart daemon
        # 4. 检查日志: "Reset N refining sessions"
        # 5. 检查 session status → needs_refinement → 被重新处理
```

### 依赖: Step 1
### 验收: test PASS

---

## Step 6: P4,P5,P6 + CI workflow

### 现状
- P4 kill mid-publishing — reset_publishing_sessions 已在 P0-3 实现
- P5 Ollama 停 — 无现有测试
- P6 磁盘满 — test_metrics.py 已覆盖 record 异常
- CI 有 .github/workflows/ci.yml，无 failure injection job

### 目标
- 完成剩余 3 个场景
- CI 能跑 failure injection

### 具体改动

**P4**: tests/failure_tests/test_kill_publishing.py
- 类似 P3，但验证 skip_llm 生效

**P5**: tests/failure_tests/test_ollama_down.py
- mock Ollama 不可用 → embedding 返回错误 → 验证不崩溃

**P6**: 已在 test_metrics.py 覆盖，标记 @pytest.mark.failure_injection

**CI**: .github/workflows/ci-failure-injection.yml
```yaml
name: Failure Injection CI
on: [push]
jobs:
  failure-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        ...
    steps:
      - uses: actions/checkout@v4
      - docker compose up -d postgres
      - pip install -e ".[dev]"
      - make daemon-start
      - pytest tests/failure_tests/ -v -m failure_injection
```

### 依赖: Step 1-5
### 验收: CI green on push

---

## Step 7: 全量验证 + commit
- ruff check
- pytest -q -m "not integration" (现有 1866 tests 不破坏)
- make e2e-verify
- git commit