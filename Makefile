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
	pip install -e ".[dev]"
	alembic upgrade head
	@echo ""
	@echo "  ✔ Setup complete!"
	@echo "  Run 'make web' to start the Web UI."
	@echo "  Run 'make mcp' to start the MCP server."
	@echo ""

dev:            ## 启动全部服务（Docker 基础设施 + Web 管理界面）
	docker compose up -d
	python -m team_memory.web.app

web:            ## 仅启动 Web 管理界面（默认端口 9111）
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
