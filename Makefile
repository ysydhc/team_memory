# ============================================================
# team_memory — 常用命令入口
# 使用 make help 查看所有可用命令
# ============================================================

.DEFAULT_GOAL := help
.PHONY: help setup dev web mcp test lint lint-fix backup health clean migrate install-knowledge

help:           ## 显示所有可用命令
	@echo ""
	@echo "  team_memory 命令一览"
	@echo "  ================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-15s %s\n", $$1, $$2}'
	@echo ""

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

dev:            ## 启动全部服务（Docker 基础设施 + 本机 Web；若 Web 容器在跑会先停掉以释放 9111）
	@docker compose stop team-memory-web 2>/dev/null || true
	docker compose up -d postgres redis
	@if ! lsof -i :11434 >/dev/null 2>&1; then docker compose --profile ollama up -d 2>/dev/null || true; fi
	python -m team_memory.web.app

web:            ## 仅启动 Web 管理界面（默认端口 9111；确保无其他进程占用 9111）
	python -m team_memory.web.app

mcp:            ## 仅启动 MCP Server（供 Cursor / Claude Desktop 使用）
	python -m team_memory.server

test:           ## 运行全部测试
	pytest tests/ -v

lint:           ## Ruff 代码检查
	ruff check src/

lint-fix:       ## Ruff 代码检查并自动修复
	ruff check src/ --fix

migrate:        ## 运行数据库迁移
	alembic upgrade head

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
