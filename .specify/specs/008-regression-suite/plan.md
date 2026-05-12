# Implementation Plan: Regression Test Suite

## Step 1: 创建 Makefile test-regression target
- 现状: test-acceptance / e2e-verify / verify-pipeline 各自独立, 无聚合命令
- 目标: make test-regression 串行执行上面三个
- 改动: Makefile 加 target, 每个失败立即退出
- 验收: make test-regression 本地通过

## Step 2: CI 集成
- 现状: ci.yml 有 lint + unit-test, 无回归测试
- 目标: 加 regression job
- 改动: .github/workflows/ci.yml 加 job (services: postgres)
- 验收: push → CI green

## Step 3: 验证 + commit
