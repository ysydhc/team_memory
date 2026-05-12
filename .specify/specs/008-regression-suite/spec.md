# Feature Specification: Regression Test Suite

**Created**: 2026-05-14
**Status**: Draft
**Input**: make test-regression 一键回归 (test-acceptance + e2e-verify + verify-pipeline), CI 集成, PR 必过

## User Scenarios

### P1: 一键回归测试
- As 开发者
- I want to make test-regression
- So that 一次命令跑完所有关键链路验证
- Acceptance: test-acceptance(21) + e2e-verify(10) + verify-pipeline(15) 全部 pass

### P2: CI 自动回归
- As 团队
- I want to PR CI 自动跑回归
- So that 改动不会破坏关键链路
- Acceptance: ci.yml 新 job "regression", 需要 PG service

## Technical Requirements
- Makefile: test-regression target
- CI: ci.yml 加 regression job (runs-on ubuntu, services: postgres)
