# ============================================================
# team_memory — 常用命令入口
# 使用 make help 查看所有可用命令
# ============================================================

# Python 解释器：优先 .env 中的 PYTHON，其次自动探测 (.venv / conda env / system)
# 换电脑时只需改 .env 中的 PYTHON 即可
DOTENV_PYTHON  ?= $(shell grep -E '^PYTHON=' .env 2>/dev/null | sed 's/^PYTHON=//' | tr -d '"' | tr -d "'")
AUTO_PYTHON    ?= $(shell \
	if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif command -v conda >/dev/null 2>&1 && conda run -n team_memory python -c "pass" 2>/dev/null; then echo "conda run -n team_memory python"; \
	else echo python; fi)
PYTHON_BIN     ?= $(or $(DOTENV_PYTHON),$(AUTO_PYTHON))

# 屏蔽 jieba/pkg_resources 等第三方库的 DeprecationWarning；可覆盖：make web PYTHONWARNINGS=
export PYTHONWARNINGS ?= ignore::DeprecationWarning

# 测试默认走项目依赖（SQLAlchemy 2.x）。优先级：uv（含 dev 额外依赖里的 pytest）→ .venv python -m pytest → …
# 勿用裸「uv run pytest」：无 .venv/bin/pytest 时会调用 PATH 上 conda 的 pytest → 旧 SQLAlchemy。
# 覆盖示例：make test PYTEST='python -m pytest'
PYTEST ?= $(shell \
	if command -v uv >/dev/null 2>&1; then echo "uv run --extra dev python -m pytest"; \
	elif [ -x .venv/bin/python ]; then echo ".venv/bin/python -m pytest"; \
	elif [ -x .venv/bin/pytest ]; then echo ".venv/bin/pytest"; \
	else echo pytest; fi)

# 迁移须与项目 SQLAlchemy 2.x 同源；勿运行 PATH 上 conda 的 alembic（会 import 旧版 sqlalchemy）。
ALEMBIC ?= $(shell \
	if command -v uv >/dev/null 2>&1; then echo "uv run python -m alembic"; \
	elif [ -x .venv/bin/python ]; then echo ".venv/bin/python -m alembic"; \
	else echo alembic; fi)

.DEFAULT_GOAL := help
.PHONY: help setup dev web mcp mcp-verify test lint lint-fix lint-js harness-check doc-harness-config-check verify verify-web verify-entities backup health clean migrate migrate-fts hooks-install sync-agent-artifacts daemon-start daemon-stop daemon-run daemon-install daemon-uninstall stats wiki-compile wiki-status entity-backfill entity-dedup embedding-backfill detect-contradictions search-eval

sync-agent-artifacts: ## 由 agents/shared + agents/manifest.yaml 生成 .claude/.cursor 下 agents、prompts、skills
	python scripts/sync_agent_artifacts.py

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
	python -m alembic upgrade head
	@echo ""
	@echo "  ✔ Setup complete!"
	@echo "  Run 'make web' to start the Web UI."
	@echo "  Run 'make mcp' to start the MCP server (requires repo-root .env; see docs/guide/mcp-server.md)."
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

mcp:            ## 启动 MCP（需仓库根 .env；经 scripts/run_mcp_with_dotenv.sh）
	bash scripts/run_mcp_with_dotenv.sh

mcp-verify:     ## 校验 MCP 工具注册（6 个 memory_* 名称与数量）
	$(PYTEST) tests/test_server.py::TestLiteToolRegistration::test_exactly_six_tools tests/test_server.py::TestLiteToolRegistration::test_tool_names -q

test:           ## 运行全部测试（默认 uv / .venv 中的 pytest）
	@case "$(PYTEST)" in \
	  pytest) \
	    if ! python -c "from sqlalchemy.orm import DeclarativeBase" 2>/dev/null; then \
	      echo ""; \
	      echo "  错误: 当前将使用 PATH 上的 pytest，且默认 python 无法导入 SQLAlchemy 2.x（需要 DeclarativeBase）。"; \
	      echo "  你多半在用 conda base 等全局环境，与项目 pyproject 不一致。"; \
	      echo "  请在本仓库根目录任选其一后重试 make verify："; \
	      echo "    uv sync"; \
	      echo "    或  python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e '.[dev]'"; \
	      echo ""; \
	      exit 1; \
	    fi ;; \
	esac
	$(PYTEST) tests/ -v

lint:           ## Ruff 代码检查
	ruff check src/

lint-fix:       ## Ruff 代码检查并自动修复
	ruff check src/ --fix

lint-js:        ## Web 前端 JS 检查：重复声明、语法类问题
	python scripts/lint_js_duplicates.py

doc-harness-config-check:  ## 校验 doc-harness.project.yaml 必填键与引用的文件存在
	python scripts/check_doc_harness_config.py

harness-check:  ## Harness 门禁：import 方向检查 + ruff + lint-js + doc-harness 配置
	python scripts/harness_import_check.py
	$(MAKE) lint
	$(MAKE) lint-js
	$(MAKE) doc-harness-config-check

verify:         ## 标准验收：lint + 全量测试
	@$(MAKE) lint
	@$(MAKE) test

verify-web:     ## Web 验收：lint + lint-js + web测试 + health/stats smoke
	@$(MAKE) -s release-9111 || true
	@$(MAKE) lint
	@$(MAKE) lint-js
	$(PYTEST) tests/test_web.py -v
	@API_KEY=*** \
	echo "  Starting Web for smoke on :9111 ..."; \
	python -m team_memory.web.app >/tmp/tm-web-smoke.log 2>&1 & \
	WEB_PID=$$!; \
	trap "kill $$WEB_PID 2>/dev/null || true" EXIT; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
	  curl -fsS http://localhost:9111/health >/dev/null 2>&1 && break; \
	  sleep 1; \
	done; \
	TEAM_MEMORY_API_KEY=*** python scripts/smoke/smoke_web_dashboard.py --api-key $$API_KEY --port 9111

verify-entities: ## 验证 L2.5 实体图表（建表 + 规则提取 + 样例数据 + 搜索/图遍历）
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/verify/verify_entity_graph.py

stats:           ## 搜索质量报告（--days 30 --granularity day）
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/tm_stats.py $(OPTS)

wiki-compile:    ## 编译 Wiki（--full 全量重建）
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/wiki_cli.py compile $(OPTS)

wiki-status:     ## Wiki 编译状态
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/wiki_cli.py status $(OPTS)

entity-backfill: ## 全量补抽实体（--model 指定模型，--dry-run 预览，--limit N 限量）
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/entity_backfill.py $(OPTS)

entity-dedup:	## 实体去重合并
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/entity_dedup.py $(OPTS)

embedding-backfill:	## 补齐缺失的 experience embedding
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/embedding_backfill.py $(OPTS)

detect-contradictions:	## 检测矛盾经验对（含 LLM 验证）
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) -c "import asyncio; from team_memory.services.contradiction_detector import detect_contradictions; from team_memory.config import load_settings; s=load_settings(); llm=getattr(s,'entity_extraction',None); pairs=asyncio.run(detect_contradictions(str(s.database.url), llm_config=llm)); print(f'Found {len(pairs)} contradiction pairs'); [print(f'  {p.exp_a_title[:50]} vs {p.exp_b_title[:50]} ({p.reason}) [LLM:{p.llm_confirmed}]') for p in pairs]"

search-eval:		## 搜索质量评估（precision/recall）
	@set -a && [ -f .env ] && source .env || true && set +a; \
	PYTHONPATH=src:scripts $(PYTHON_BIN) scripts/daemon/search_eval.py $(OPTS)

migrate:        ## 运行数据库迁移（默认 uv / .venv 内 python -m alembic）
	@case "$(ALEMBIC)" in \
	  alembic) \
	    if ! python -c "from sqlalchemy.orm import DeclarativeBase" 2>/dev/null; then \
	      echo ""; \
	      echo "  错误: 将使用 PATH 上的 alembic，且当前 python 无 SQLAlchemy 2.x。"; \
	      echo "  请在本仓库根目录执行:  uv sync   或   pip install -e .   后重试，或使用:"; \
	      echo "    uv run python -m alembic upgrade head"; \
	      echo ""; \
	      exit 1; \
	    fi ;; \
	esac
	$(ALEMBIC) upgrade head

migrate-fts:    ## 补齐经验表 FTS 字段（存量迁移）；可用 --dry-run 预览
	python scripts/migrate_fts.py

migrate-fts-jieba:  ## 回填 jieba 分词列（方案 C）；需先执行 alembic upgrade
	python scripts/migrate_fts_jieba.py

backup:         ## 备份数据库
	./scripts/backup.sh

health:         ## 一键健康检查（检测所有组件状态）
	./scripts/healthcheck.sh

clean:          ## 清理 Python 缓存文件
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

hooks-install:  ## 安装 Git hooks（post-commit 自动更新任务）
	cp scripts/post-commit-hook.sh .git/hooks/post-commit
	chmod +x .git/hooks/post-commit
	@echo "  ✔ Git post-commit hook installed."

daemon-start:    ## Start TM Daemon (launchd)
	launchctl load ~/Library/LaunchAgents/com.teammemory.daemon.plist

daemon-stop:     ## Stop TM Daemon (launchd)
	launchctl unload ~/Library/LaunchAgents/com.teammemory.daemon.plist

daemon-run:      ## Run TM Daemon in foreground (for testing)
	PYTHONPATH=src:scripts $(PYTHON_BIN) -m daemon

daemon-install:  ## Install launchd plist (auto-start on login + crash restart)
	@PLIST_SRC=scripts/daemon/com.teammemory.daemon.plist; \
	PLIST_DST=$(HOME)/Library/LaunchAgents/com.teammemory.daemon.plist; \
	REPO_ABS=$$(pwd); \
	sed -e "s|__PYTHON_PATH__|$(PYTHON_BIN)|g" -e "s|__REPO_ROOT__|$$REPO_ABS|g" "$$PLIST_SRC" > "$$PLIST_DST"; \
	launchctl load "$$PLIST_DST" 2>/dev/null || true; \
	echo "  ✔ TM Daemon installed as launchd service (python=$(PYTHON_BIN))"

daemon-uninstall: ## Uninstall launchd plist (stop auto-start)
	launchctl unload ~/Library/LaunchAgents/com.teammemory.daemon.plist 2>/dev/null || true
	rm -f ~/Library/LaunchAgents/com.teammemory.daemon.plist
	@echo "  ✔ TM Daemon uninstalled from launchd"
