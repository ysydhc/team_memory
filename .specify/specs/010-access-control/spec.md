# Feature Specification: Access Control

**Created**: 2026-05-15
**Status**: Draft
**Input**: Daemon POST 端点加 Bearer token 鉴权, 无 token → 403

## Design Decisions
- 保护: POST 端点 (/hooks/*, /draft/*, /alert/*)
- GET 端点 (/status, /recall, /stats, etc.) 不保护
- Token: TM_DAEMON_TOKEN 在 .env
- 降级: 没配 → 全部 POST 拒绝 (严格模式)
- Hook 脚本: 读 .env → 发 Authorization: Bearer

## User Scenarios

### P1: 有 token → 正常调用
- As hook 脚本
- I want to 带 token 调 POST 端点
- Acceptance: 200 OK

### P2: 无 token → 拒绝
- As 未授权调用
- I want to 不带 token 调 POST
- Acceptance: 403 {"error": "unauthorized"}

### P3: 错误 token → 拒绝
- As 未授权调用
- I want to 带错 token 调 POST
- Acceptance: 403

### P4: token 未配置 → POST 全拒
- As 管理员没配 token
- I want to 系统安全降级
- Acceptance: 所有 POST 返回 403

## Technical Requirements
- app.py: FastAPI middleware 或 Depends 检查 Authorization header
- .env: TM_DAEMON_TOKEN=xxx
- Hook 脚本: 读 .env → headers["Authorization"]="Bearer xxx"
