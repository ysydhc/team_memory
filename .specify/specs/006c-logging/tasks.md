# Tasks: Structured Logging

**Total**: 4

## Task 1: Create StructuredLogger + test
- File: structured_logger.py (new), test_structured_logger.py (new)
- Action: 实现 info/warning/error/debug 方法, JSON 输出
- Test: 2 tests
- Status: [x] done

## Task 2: Apply to pipeline + refinement_worker
- Files: pipeline.py, refinement_worker.py
- Action: 替换关键日志调用
- Test: 现有 tests 不破坏, 手动检查日志输出
- Status: [x] done

## Task 3: filter-logs.py + make logs
- Files: scripts/filter-logs.py, Makefile
- Action: 过滤脚本 + make target
- Test: make logs SESSION=test 无报错
- Status: [x] done

## Task 4: Verify + commit
- Action: ruff + pytest + e2e-verify + commit
- Status: [x] done
