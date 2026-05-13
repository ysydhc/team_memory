# Implementation Plan: Access Control

## Step 1: FastAPI dependency 加 token 检查
- 现状: 无鉴权，任何 POST 都接受
- 目标: POST 端点需要 Bearer token
- 具体改动:
  app.py: 加 `verify_token` dependency → app.post(... dependencies=[Depends(verify_token)])
  verify_token 读 Authorization header → 对比 .env TM_DAEMON_TOKEN
- 验收: pytest test 验证 200/403

## Step 2: Hook 脚本加 Authorization header
- 现状: hook 脚本不带 token
- 目标: 读 .env TM_DAEMON_TOKEN → headers["Authorization"]="Bearer xxx"
- 具体改动:
  cursor_before_prompt.py, cursor_after_response.py, cursor_session_start.py,
  hermes_user_prompt_submit.py, hermes_stop.py, claude_stop.py
- 验收: 手动验证

## Step 3: Git commit
