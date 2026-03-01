# ============================================================
# team_memory — 常用命令入口
# 使用 make help 查看所有可用命令
# ============================================================

.DEFAULT_GOAL := help
.PHONY: help setup dev web mcp test lint lint-fix verify verify-web backup health clean migrate migrate-fts install-knowledge release-9111 hooks-install

help:           ## 显示所有可用命令
	@echo ""
	@echo "  team_memory 命令一览"
	@echo "  ================================="
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-18s %s\n", $$1, $$2}'
	@echo ""

release-9111:   ## 释放 9111 端口（删容器+杀进程+等待）；供 dev/web 内部使用，也可单独执行
	@docker compose rm -fs team-memory-web 2>/dev/null || true
	@docker ps -q --filter "publish=9111" 2>/dev/null | while read cid; do docker rm -f "$$cid" 2>/dev/null; done || true
	@lsof -i :9111 -t 2>/dev/null | xargs kill -9 2>/dev/null || true
	@sleep 1
	@lsof -i :9111 -t 2>/dev/null | xargs kill -9 2>/dev/null || true
	@for i in 1 2 3 4 5; do \
	  lsof -i :9111 >/dev/null 2>&1 || exit 0; \
	  sleep 1; \
	done; \
	echo "  ⚠ 端口 9111 仍被占用，请手动检查: lsof -i :9111"; exit 1

setup:          ## 首次安装：启动 Docker + 安装依赖 + 初始化数据库
	docker compose up -d
	@echo "  Waiting for PostgreSQL to be ready..."
	@until docker compose exec -T postgres pg_isready -U developer -d postgres >/dev/null 2>&1; do sleep 2; done
	@docker compose exec -T postgres psql -U developer -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='team_memory'" | grep -q 1 || \
	  (echo "  Creating database team_memory..." && docker compose exec -T postgres psql -U developer -d postgres -c "CREATE DATABASE team_memory;")
	@if lsof -i :11434 >/dev/null 2>&1; then \
	  echo "  ℹ Port 11434 in use: using host Ollama for embedding (skip Ollama container)."; \
	else \
	  echo "  Starting Ollama container (profile ollama)..."; \
	  docker compose --profile ollama up -d; \
	fi
	pip install -e ".[dev]"
	alembic upgrade head
	@echo ""
	@echo "  ✔ Setup complete!"
	@echo "  Run 'make web' to start the Web UI."
	@echo "  Run 'make mcp' to start the MCP server."
	@echo "  If Ollama is unavailable, set embedding.provider to openai in config for embedding."
	@echo ""

dev:            ## 启动全部服务（Docker 基础设施 + 本机 Web）；自动释放 9111 后启动
	@$(MAKE) -s release-9111 || true
	docker compose up -d postgres redis
	@if ! lsof -i :11434 >/dev/null 2>&1; then docker compose --profile ollama up -d 2>/dev/null || true; fi
	python -m team_memory.web.app

web:            ## 仅启动 Web 管理界面（默认端口 9111）；自动释放 9111 后启动
	@$(MAKE) -s release-9111 || true
	python -m team_memory.web.app

mcp:            ## 仅启动 MCP Server（供 Cursor / Claude Desktop 使用）
	python -m team_memory.server

test:           ## 运行全部测试
	pytest tests/ -v

lint:           ## Ruff 代码检查
	ruff check src/

lint-fix:       ## Ruff 代码检查并自动修复
	ruff check src/ --fix

verify:         ## 标准验收：lint + 全量测试
	@$(MAKE) lint
	@$(MAKE) test

verify-web:     ## Web 验收：lint + web测试 + health/stats smoke
	@$(MAKE) -s release-9111 || true
	@$(MAKE) lint
	pytest tests/test_web.py -v
	@API_KEY=$${TEAM_MEMORY_API_KEY:-dev-key}; \
	echo "  Starting Web for smoke on :9111 ..."; \
	python -m team_memory.web.app >/tmp/tm-web-smoke.log 2>&1 & \
	WEB_PID=$$!; \
	trap "kill $$WEB_PID 2>/dev/null || true" EXIT; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
	  curl -fsS http://localhost:9111/health >/dev/null 2>&1 && break; \
	  sleep 1; \
	done; \
	TEAM_MEMORY_API_KEY=$$API_KEY python scripts/smoke_web_dashboard.py --api-key $$API_KEY --port 9111

migrate:        ## 运行数据库迁移
	alembic upgrade head

migrate-fts:    ## 补齐经验表 FTS 字段（存量迁移）；可用 --dry-run 预览
	python scripts/migrate_fts.py

backup:         ## 备份数据库
	./scripts/backup.sh

health:         ## 一键健康检查（检测所有组件状态）
	./scripts/healthcheck.sh

clean:          ## 清理 Python 缓存文件
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

install-knowledge: ## 一键安装固化知识包（rules + skill）
	./scripts/install_codified_knowledge.sh

hooks-install:  ## 安装 Git hooks（post-commit 自动更新任务）
	cp scripts/post-commit-hook.sh .git/hooks/post-commit
	chmod +x .git/hooks/post-commit
	@echo "  ✔ Git post-commit hook installed."
