#!/bin/sh
# Used as GIT_EDITOR during rebase to inject task-tagged messages.
# Usage: GIT_EDITOR="$(pwd)/scripts/reword_editor.sh" git rebase -i ...
FILE="$1"
if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
  exec "${GIT_EDITOR:-editor}" "$@"
  exit 0
fi
CONTENT=$(cat "$FILE")
if echo "$CONTENT" | grep -q "MCP: 更新 instructions"; then
  cat > "$FILE" << 'EOF'
P1-9: MCP instructions 与 feedback_hint、工具描述与参数

- instructions 强调 tm_preflight 与各工具使用场景
- tm_solve/tm_search 返回 feedback_hint 引导调用 tm_feedback
- tm_search 对 top-1 结果递增 use_count
- tm_feedback/tm_task 描述与 session 参数说明

Made-with: Cursor
EOF
elif echo "$CONTENT" | grep -q "docs: README 更新"; then
  cat > "$FILE" << 'EOF'
P1-10: README 文档（CI/CD、FTS、tm_learn、Make 表）

- 增加 CI/CD 小节与 .debug/25-ci-cd 链接
- make migrate-fts 与 FTS 存量迁移 FAQ
- tm_learn 与经验组、查询优化说明

Made-with: Cursor
EOF
elif echo "$CONTENT" | grep -q "chore: 依赖与配置"; then
  cat > "$FILE" << 'EOF'
P2-2: /metrics 端点与依赖、bootstrap、前端、测试

- pyproject.toml、bootstrap、reranker factory、llm_client 调整
- web metrics 与 static/index.html 更新
- test_service、test_web 用例更新

Made-with: Cursor
EOF
fi
exit 0
