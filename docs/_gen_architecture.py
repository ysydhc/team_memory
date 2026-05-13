#!/usr/bin/env python3
"""
生成 `const MODULES = { ... };` 片段，用于手工替换 docs/architecture.html 中同名块。
生成：python3 docs/_gen_architecture.py > /tmp/mods.js
然后替换 architecture.html 内从 `const MODULES = {` 到对应 `};`（仅 MODULES 对象）。
"""

from __future__ import annotations

import json
from pathlib import Path


def M(**kwargs):
    return kwargs


def module_entry_js(m: dict) -> str:
    o = {k: v for k, v in m.items() if k not in ("_in", "_out")}
    o["dependencies"] = {"in": m.get("_in") or [], "out": m.get("_out") or []}
    mid = m["id"]
    body = json.dumps(o, ensure_ascii=False, separators=(",", ":"))
    return f"  {json.dumps(mid, ensure_ascii=False)}: M({body}),"


def build_rows() -> list[dict]:
    rows: list[dict] = []

    def add(**m):
        rows.append(m)

    # entry
    add(
        id="mcp-server",
        name="server.py",
        path="src/team_memory/server.py",
        domain="entry",
        subgroup="入口",
        layer="entry",
        level="L3",
        band="entry",
        description="FastMCP：memory_* 工具；远程 HTTP。",
        tags=["MCP"],
        _out=["memory-operations", "search-orchestrator"],
    )
    add(
        id="cli",
        name="cli.py",
        path="src/team_memory/cli.py",
        domain="entry",
        subgroup="入口",
        layer="entry",
        level="L3",
        band="entry",
        description="tm-cli。",
        tags=["CLI"],
        _out=["remote-client"],
    )
    add(
        id="bootstrap",
        name="bootstrap.py",
        path="src/team_memory/bootstrap.py",
        domain="entry",
        subgroup="入口",
        layer="services",
        level="L2",
        band="services",
        description="AppContext。",
        graphExpand=["cfg-settings", "llm-client"],
        _out=["web-app", "database", "event-bus"],
    )
    add(
        id="client-config",
        name="client_config.py",
        path="src/team_memory/client_config.py",
        domain="entry",
        subgroup="入口",
        layer="services",
        level="L3",
        band="services",
        description="客户端 TOML。",
        cli_make=["tm-cli config show"],
    )
    add(
        id="project-resolver",
        name="project_resolver.py",
        path="src/team_memory/project_resolver.py",
        domain="entry",
        subgroup="入口",
        layer="services",
        level="L3",
        band="services",
        description="项目名推断。",
    )
    add(
        id="remote-client",
        name="remote_client.py",
        path="src/team_memory/remote_client.py",
        domain="entry",
        subgroup="入口",
        layer="services",
        level="L3",
        band="services",
        description="HTTP 远程客户端。",
        tags=["HTTP"],
        _in=["cli"],
        _out=["web-app"],
    )

    # web
    add(
        id="web-app",
        name="web/app.py",
        path="src/team_memory/web/app.py",
        domain="web",
        subgroup="核心",
        layer="entry",
        level="L2",
        band="entry",
        description="FastAPI。",
        tags=["Web"],
        graphExpand=["routes-search", "middleware-stack"],
        _in=["bootstrap"],
        _out=[
            "routes-search",
            "routes-experiences",
            "routes-auth",
            "routes-archives",
            "routes-janitor",
            "routes-entities",
            "routes-personal-memory",
            "routes-config",
            "routes-dedup",
            "routes-mcp-compat",
        ],
    )
    add(
        id="middleware-stack",
        name="web/middleware.py",
        path="src/team_memory/web/middleware.py",
        domain="web",
        subgroup="核心",
        layer="services",
        level="L3",
        band="services",
        description="全局中间件。",
        _in=["web-app"],
    )
    add(
        id="web-dependencies",
        name="dependencies.py",
        path="src/team_memory/web/dependencies.py",
        domain="web",
        subgroup="核心",
        layer="services",
        level="L3",
        band="services",
        description="依赖注入。",
        _in=["web-app"],
    )
    add(
        id="web-mcp-auth",
        name="mcp_auth_middleware.py",
        path="src/team_memory/web/mcp_auth_middleware.py",
        domain="web",
        subgroup="核心",
        layer="services",
        level="L3",
        band="services",
        tags=["MCP"],
        _in=["auth-provider"],
        _out=["web-app"],
    )
    add(
        id="auth-session",
        name="auth_session.py",
        path="src/team_memory/web/auth_session.py",
        domain="web",
        subgroup="核心",
        layer="services",
        level="L3",
        band="services",
        _in=["routes-auth"],
        _out=["web-app"],
    )
    add(
        id="web-schemas",
        name="schemas.py",
        path="src/team_memory/web/schemas.py",
        domain="web",
        subgroup="核心",
        layer="services",
        level="L3",
        band="services",
        description="Web Pydantic。",
        _out=["routes-experiences"],
    )

    route_specs = [
        ("routes-search", "routes/search.py", ["web-app"], ["search-orchestrator"]),
        ("routes-experiences", "routes/experiences.py", ["web-app"], ["experience-service"]),
        ("routes-auth", "routes/auth.py", ["web-app"], ["auth-provider", "auth-session"]),
        ("routes-archives", "routes/archives.py", ["web-app"], ["archive-service"]),
        ("routes-janitor", "routes/janitor.py", ["web-app"], ["janitor"]),
        ("routes-entities", "routes/entities.py", ["web-app"], ["entity-extractor"]),
        ("routes-personal-memory", "routes/personal_memory.py", ["web-app"], ["personal-memory-service"]),
        ("routes-config", "routes/config.py", ["web-app"], ["bootstrap"]),
        ("routes-dedup", "routes/dedup.py", ["web-app"], ["repository"]),
        ("routes-mcp-compat", "routes/mcp_compat.py", ["web-app"], ["memory-operations"]),
    ]
    for rid, rpath, rin, rout in route_specs:
        add(
            id=rid,
            name=rpath.split("/")[-1],
            path=f"src/team_memory/web/{rpath}",
            domain="web",
            subgroup="Routes",
            layer="services",
            level="L3",
            band="services",
            tags=["API"],
            _in=rin,
            _out=rout,
        )

    def hook(hid, fname, plat, life, outd, desc=None):
        d = dict(
            id=hid,
            name=fname,
            path=f"src/team_memory/hooks/{fname}",
            domain="hooks",
            subgroup=plat,
            subhooksPlatform=plat,
            subhooksLifecycle=life,
            layer="entry",
            level="L3",
            band="entry",
            tags=["Hook"],
            _out=outd,
        )
        if desc:
            d["description"] = desc
        add(**d)

    hook("hook-claude-session-start", "claude_session_start.py", "Claude", "会话开始", ["daemon-pipeline"])
    hook("hook-claude-user-prompt-submit", "claude_user_prompt_submit.py", "Claude", "提示词/输入", ["daemon-pipeline"])
    hook("hook-claude-post-tool-use", "claude_post_tool_use.py", "Claude", "工具/响应后", ["daemon-pipeline"])
    hook("hook-claude-stop", "claude_stop.py", "Claude", "会话结束", ["daemon-pipeline"])
    hook("hook-cursor-session-start", "cursor_session_start.py", "Cursor", "会话开始", ["daemon-pipeline"])
    hook("hook-cursor-before-prompt", "cursor_before_prompt.py", "Cursor", "提示词/输入", ["daemon-pipeline"])
    hook("hook-cursor-after-response", "cursor_after_response.py", "Cursor", "工具/响应后", ["daemon-pipeline"])
    hook("hook-cursor-after-mcp", "cursor_after_mcp.py", "Cursor", "工具/响应后", ["daemon-pipeline"])
    hook("hook-hermes-user-prompt-submit", "hermes_user_prompt_submit.py", "Hermes", "提示词/输入", ["daemon-pipeline"])
    hook("hook-hermes-post-tool-call", "hermes_post_tool_call.py", "Hermes", "工具/响应后", ["daemon-pipeline"])
    hook("hook-hermes-stop", "hermes_stop.py", "Hermes", "会话结束", ["daemon-pipeline"])
    hook("hook-hermes-pipeline", "hermes_pipeline.py", "Hermes", "完整流水线", ["daemon-pipeline"])
    hook("hook-common", "common.py", "公共", "公共", [])
    hook("hook-shared", "shared.py", "公共", "公共", [])
    hook("hook-git-watcher", "git_watcher.py", "公共", "公共", ["daemon-pipeline"])
    hook("hook-initial-index", "initial_index.py", "公共", "公共", ["daemon-markdown-indexer"])
    hook("hook-retrieval-trigger", "retrieval_trigger.py", "公共", "公共", ["daemon-pipeline"])
    hook("hook-session-timeout", "session_timeout.py", "公共", "公共", ["daemon-pipeline"])
    hook("hook-weekly-report", "weekly_report.py", "公共", "公共", ["daemon-pipeline"])

    def da(
        hid,
        fname,
        sg,
        desc="",
        *,
        out=None,
        inn=None,
        _in=None,
        L="daemon",
        lv="L3",
        band="daemon",
        tags=None,
        ge=None,
    ):
        if _in is not None:
            inn = _in
        d = dict(
            id=hid,
            name=fname,
            path=f"src/team_memory/daemon/{fname}",
            domain="daemon",
            subgroup=sg,
            layer=L,
            level=lv,
            band=band,
            description=desc,
        )
        if tags:
            d["tags"] = tags
        if ge:
            d["graphExpand"] = ge
        if inn:
            d["_in"] = inn
        d["_out"] = out or []
        add(**d)

    da("daemon-__main__", "__main__.py", "核心运行", "tm-daemon CLI。", out=["daemon-app"])
    da(
        "daemon-app",
        "app.py",
        "核心运行",
        "守护进程 HTTP。",
        out=[
            "daemon-pipeline",
            "daemon-watcher",
            "daemon-health-check",
            "daemon-pending-queue",
            "daemon-metrics",
            "daemon-markdown-indexer",
            "daemon-reloader",
            "daemon-payload-capture",
            "daemon-response-buffer",
        ],
        lv="L2",
        ge=["daemon-draft-buffer", "daemon-refinement-worker", "daemon-metrics"],
    )
    da("daemon-config", "config.py", "核心运行", "", _in=["client-config"], out=["daemon-app"])
    da(
        "daemon-pipeline",
        "pipeline.py",
        "核心运行",
        "Hook 事件管线。",
        inn=["daemon-app"],
        out=[
            "daemon-conversation-buffer",
            "daemon-tm-sink",
            "daemon-draft-buffer",
            "daemon-draft-refiner",
            "daemon-convergence",
            "daemon-refinement-worker",
            "daemon-search-log-writer",
            "daemon-personal-memory-extractor",
        ],
        lv="L2",
        ge=["daemon-draft-refiner", "daemon-convergence"],
    )
    da("daemon-reloader", "reloader.py", "核心运行", "", _in=["daemon-app"])
    da("daemon-health-check", "health_check.py", "核心运行", "", _in=["daemon-app"])

    da("daemon-conversation-buffer", "conversation_buffer.py", "缓冲/队列", "", _in=["daemon-pipeline"])
    da("daemon-draft-buffer", "draft_buffer.py", "缓冲/队列", "", _in=["daemon-pipeline"])
    da("daemon-response-buffer", "response_buffer.py", "缓冲/队列", "", _in=["daemon-app"])
    da("daemon-pending-queue", "pending_queue.py", "缓冲/队列", "", tags=["Event"], _in=["daemon-app"])
    da("daemon-payload-capture", "payload_capture.py", "缓冲/队列", "", _in=["daemon-app"])

    da(
        "daemon-draft-refiner",
        "draft_refiner.py",
        "精炼/加工",
        "",
        _in=["daemon-pipeline"],
        out=["llm-client"],
    )
    da("daemon-refinement-worker", "refinement_worker.py", "精炼/加工", "", _in=["daemon-pipeline"])
    da("daemon-intent-classifier", "intent_classifier.py", "精炼/加工", "", _in=["daemon-pipeline"])
    da("daemon-convergence", "convergence_detector.py", "精炼/加工", "", _in=["daemon-pipeline"])
    da(
        "daemon-personal-memory-extractor",
        "personal_memory_extractor.py",
        "精炼/加工",
        "",
        _in=["daemon-pipeline"],
        out=["personal-memory-service"],
    )

    da(
        "daemon-embedding-backfill",
        "embedding_backfill.py",
        "补全/批处理",
        "",
        out=["repository", "embedding-ollama", "embedding-openai", "embedding-generic", "embedding-local"],
        inn=["bootstrap"],
        tags=["批处理"],
    )
    da(
        "daemon-entity-backfill",
        "entity_backfill.py",
        "补全/批处理",
        "",
        out=["entity-extractor", "repository"],
        tags=["批处理"],
    )
    da("daemon-entity-dedup", "entity_dedup.py", "补全/批处理", "", out=["repository"], tags=["批处理"])
    da("daemon-solution-backfill", "solution_backfill.py", "补全/批处理", "", out=["repository"])
    da("daemon-faithfulness-batch", "faithfulness_batch.py", "补全/批处理", "", out=["repository"])
    da(
        "daemon-faithfulness-judge",
        "faithfulness_judge.py",
        "补全/批处理",
        "",
        out=["llm-client"],
    )

    da(
        "daemon-search-eval",
        "search_eval.py",
        "搜索评估",
        "",
        out=["search-pipeline"],
    )
    da(
        "daemon-search-log-writer",
        "search_log_writer.py",
        "搜索评估",
        "",
        out=["search-log-repo"],
    )

    da("daemon-llm-wiki", "llm_wiki.py", "Wiki/知识", "", out=["llm-client"])
    da("daemon-wiki-cli", "wiki_cli.py", "Wiki/知识", "")
    da("daemon-wiki-compiler", "wiki_compiler.py", "Wiki/知识", "", out=["repository"])
    da(
        "daemon-markdown-indexer",
        "markdown_indexer.py",
        "Wiki/知识",
        "",
        _in=["daemon-app"],
        out=["repository"],
    )

    da("daemon-metrics", "metrics_collector.py", "监控/统计", "", _in=["daemon-app"])
    da("daemon-tm-stats", "tm_stats.py", "监控/统计", "")
    da(
        "daemon-tm-sink",
        "tm_sink.py",
        "监控/统计",
        "POST Web API",
        _in=["daemon-pipeline"],
        out=["web-app"],
    )
    da("daemon-structured-logger", "structured_logger.py", "监控/统计", "")
    da("daemon-watcher", "watcher.py", "监控/统计", "", _in=["daemon-app"], out=["daemon-pipeline"])

    da("daemon-auto-dream", "auto_dream.py", "实验", "")
    da("daemon-test-llm", "test_llm.py", "实验", "", out=["llm-client"])

    # services
    SG = {
        "search": "搜索",
        "exp": "经验管理",
        "llm": "LLM",
        "ent": "实体/模式",
        "arc": "归档",
        "maint": "维护",
        "base": "基础",
    }

    def svc(
        sid,
        fname,
        sub,
        *,
        out=None,
        inn=None,
        _in=None,
        lv="L3",
        L="L2",
        desc="",
        tags=None,
        ge=None,
        db=None,
        band="services",
    ):
        if _in is not None:
            inn = _in
        d = dict(
            id=sid,
            name=fname,
            path=f"src/team_memory/services/{fname}",
            domain="services",
            subgroup=sub,
            layer="services",
            level=lv,
            band=band,
            description=desc,
        )
        if tags:
            d["tags"] = tags
        if ge:
            d["graphExpand"] = ge
        if db:
            d["db_tables"] = db
        if inn:
            d["_in"] = inn
        d["_out"] = out or []
        add(**d)

    svc(
        "search-orchestrator",
        "search_orchestrator.py",
        SG["search"],
        lv="L2",
        L="L2",
        desc="搜索编排。",
        db=["search_logs"],
        ge=["search-pipeline", "entity-search", "tokenizer", "reranker-factory"],
        _in=["memory-operations", "routes-search"],
        out=["search-pipeline", "evaluation"],
        tags=["RRF"],
    )
    svc(
        "search-pipeline",
        "search_pipeline.py",
        SG["search"],
        desc="RRF + rerank。",
        _in=["search-orchestrator"],
        out=["repository", "cache-service", "reranker-factory", "embedding-ollama"],
        tags=["RRF", "向量"],
    )
    svc(
        "entity-search",
        "entity_search.py",
        SG["search"],
        _in=["search-orchestrator"],
        out=["repository"],
    )
    svc("tokenizer", "tokenizer.py", SG["search"], _in=["search-pipeline"])
    svc(
        "cache-service",
        "cache.py",
        SG["search"],
        desc="检索缓存。",
        _in=["search-pipeline"],
        out=["redis"],
    )
    svc(
        "evaluation",
        "evaluation.py",
        SG["search"],
        _in=["search-orchestrator"],
        out=["experience-service"],
        db=["experiences"],
    )

    svc(
        "memory-operations",
        "memory_operations.py",
        SG["exp"],
        lv="L2",
        L="L2",
        desc="MCP 共享编排。",
        _in=["mcp-server", "routes-mcp-compat"],
        out=[
            "experience-service",
            "archive-service",
            "search-orchestrator",
            "personal-memory-service",
            "search-log-repo",
        ],
        tags=["MCP"],
    )
    svc(
        "experience-service",
        "experience.py",
        SG["exp"],
        lv="L2",
        L="L2",
        db=["experiences", "experience_feedbacks", "experience_entities"],
        ge=["janitor", "intent-router"],
        _in=["memory-operations", "routes-experiences"],
        out=["repository", "embedding-ollama"],
    )
    svc(
        "personal-memory-service",
        "personal_memory.py",
        SG["exp"],
        db=["personal_memories"],
        _in=["memory-operations"],
        out=["personal-memory-repo"],
    )
    svc("promotion-compiler", "promotion_compiler.py", SG["exp"], _in=["experience-service"], out=["repository"])
    svc("intent-router", "intent_router.py", SG["exp"], _in=["search-orchestrator"])

    svc("llm-client", "llm_client.py", SG["llm"], _in=["bootstrap"], out=["openai-api"])
    svc("llm-provider", "llm_provider.py", SG["llm"], _in=["bootstrap"])
    svc("llm-parser", "llm_parser.py", SG["llm"])

    svc("entity-extractor", "entity_extractor.py", SG["ent"], out=["repository"])
    svc("pattern-extractor", "pattern_extractor.py", SG["ent"], _in=["llm-client"], out=["repository"])
    svc("topic-discovery", "topic_discovery.py", SG["ent"])
    svc("contradiction-detector", "contradiction_detector.py", SG["ent"], _in=["llm-client"], out=["repository"])

    svc(
        "archive-service",
        "archive.py",
        SG["arc"],
        db=["archives"],
        _in=["memory-operations", "routes-archives"],
        out=["archive-repository"],
    )

    svc("janitor", "janitor.py", SG["maint"], _in=["experience-service"], out=["repository"])
    svc("janitor-scheduler", "janitor_scheduler.py", SG["maint"], _in=["janitor"], out=["event-bus"], tags=["Event"])
    svc("event-bus", "event_bus.py", SG["maint"], tags=["Event"], _in=["bootstrap"])

    svc("task-runner", "task_runner.py", SG["base"], _in=["bootstrap"])
    svc("io-logger", "io_logger.py", SG["base"])

    # embedding
    def emb(eid, fname, sub="providers"):
        add(
            id=eid,
            name=fname,
            path=f"src/team_memory/embedding/{fname}",
            domain="embedding",
            subgroup=sub,
            layer="services",
            level="L3",
            band="services",
            tags=["Embed"],
        )

    emb("embedding-base", "base.py", "core")
    emb("embedding-ollama", "ollama_provider.py")
    emb("embedding-openai", "openai_provider.py")
    emb("embedding-local", "local_provider.py")
    emb("embedding-generic", "generic_provider.py")

    # reranker
    def rr(rid, fname, sub="providers"):
        add(
            id=rid,
            name=fname,
            path=f"src/team_memory/reranker/{fname}",
            domain="reranker",
            subgroup=sub,
            layer="services",
            level="L3",
            band="services",
            tags=["Rerank"],
        )

    rr("reranker-base", "base.py", "core")
    add(
        id="reranker-factory",
        name="factory.py",
        path="src/team_memory/reranker/factory.py",
        domain="reranker",
        subgroup="core",
        layer="services",
        level="L3",
        band="services",
        description="Reranker 工厂。",
        graphExpand=["reranker-jina", "reranker-cross", "reranker-ollama", "reranker-noop"],
        _in=["search-pipeline"],
        out=["reranker-jina", "reranker-cross", "reranker-ollama", "reranker-noop"],
        tags=["Rerank"],
    )
    rr("reranker-cross", "cross_encoder_provider.py")
    rr("reranker-jina", "jina_provider.py")
    rr("reranker-noop", "noop_provider.py")
    rr("reranker-ollama", "ollama_llm_provider.py", "providers")

    # config modules (12) + aggregate
    cfg_files = [
        ("cfg-settings", "settings.py"),
        ("cfg-auth", "auth.py"),
        ("cfg-database", "database.py"),
        ("cfg-embedding", "embedding.py"),
        ("cfg-llm", "llm.py"),
        ("cfg-mcp", "mcp.py"),
        ("cfg-reranker", "reranker.py"),
        ("cfg-search", "search.py"),
        ("cfg-web", "web.py"),
        ("cfg-janitor", "janitor.py"),
        ("cfg-lifecycle", "lifecycle.py"),
        ("cfg-logging", "logging_config.py"),
    ]
    for cid, fn in cfg_files:
        add(
            id=cid,
            name=fn,
            path=f"src/team_memory/config/{fn}",
            domain="config",
            subgroup="模块",
            layer="services",
            level="L3",
            band="services",
            description=f"config/{fn}",
            _out=["bootstrap"] if cid == "cfg-settings" else [],
        )
    add(
        id="config-root",
        name="⚙ 配置系统",
        path="src/team_memory/config/",
        domain="config",
        subgroup="_aggregate",
        layer="services",
        level="L1",
        band="services",
        description="点击展开 12 个子模块（DAG）。",
        isConfigAggregate=True,
    )

    # auth
    add(
        id="auth-provider",
        name="provider.py",
        path="src/team_memory/auth/provider.py",
        domain="auth",
        subgroup="provider",
        layer="services",
        level="L3",
        band="services",
        db_tables=["api_keys"],
        _in=["bootstrap"],
        _out=["routes-auth"],
    )
    add(
        id="auth-permissions",
        name="permissions.py",
        path="src/team_memory/auth/permissions.py",
        domain="auth",
        subgroup="permissions",
        layer="services",
        level="L3",
        band="services",
        _in=["auth-provider"],
    )
    add(
        id="auth-oauth",
        name="oauth.py",
        path="src/team_memory/auth/oauth.py",
        domain="auth",
        subgroup="oauth",
        layer="services",
        level="L3",
        band="services",
        _in=["routes-auth"],
        _out=["auth-provider"],
    )
    add(
        id="init-admin",
        name="init_admin.py",
        path="src/team_memory/auth/init_admin.py",
        domain="auth",
        subgroup="init_admin",
        layer="services",
        level="L3",
        band="services",
        _in=["bootstrap"],
        _out=["postgres"],
    )

    # storage
    def st(sid, fname, sub, **kw):
        inn = kw.pop("_in", kw.pop("inn", []))
        out = kw.pop("_out", kw.pop("out", []))
        add(
            id=sid,
            name=fname,
            path=f"src/team_memory/storage/{fname}",
            domain="storage",
            subgroup=sub,
            layer="data",
            level="L3",
            band="data",
            **kw,
            _in=inn,
            _out=out,
        )

    st(
        "database",
        "database.py",
        "database",
        _in=["bootstrap"],
        _out=["postgres"],
    )
    st(
        "models-orm",
        "models.py",
        "models",
        description="SQLAlchemy ORM。",
        db_tables=["experiences", "archives", "search_logs"],
        _out=["postgres"],
        ddlPreview="experiences / archives / entities …",
    )
    st(
        "repository",
        "repository.py",
        "repository",
        _in=["experience-service", "search-pipeline", "daemon-wiki-compiler", "daemon-markdown-indexer"],
        _out=["postgres"],
    )
    st(
        "archive-repository",
        "archive_repository.py",
        "archive_repo",
        _in=["archive-service"],
        _out=["postgres"],
    )
    st(
        "personal-memory-repo",
        "personal_memory_repository.py",
        "personal_memory_repo",
        db_tables=["personal_memories"],
        _in=["personal-memory-service"],
        _out=["postgres"],
    )
    st(
        "search-log-repo",
        "search_log_repository.py",
        "search_log_repo",
        db_tables=["search_logs"],
        _in=["memory-operations", "daemon-search-log-writer"],
        _out=["postgres"],
    )
    st("query-builders", "query_builders.py", "query_builders")

    # monitoring
    for mid, fn in [
        ("mon-alerting", "alerting.py"),
        ("mon-heartbeat", "heartbeat.py"),
        ("mon-watchdog", "watchdog.py"),
        ("mon-config", "config.py"),
    ]:
        add(
            id=mid,
            name=fn,
            path=f"src/team_memory/monitoring/{fn}",
            domain="monitoring",
            subgroup="runtime",
            layer="services",
            level="L3",
            band="services",
        )

    # external
    add(
        id="postgres",
        name="PostgreSQL + pgvector",
        path="Docker",
        domain="external",
        subgroup="infra",
        layer="external",
        level="L2",
        band="external",
        tags=["DB"],
    )
    add(
        id="redis",
        name="Redis",
        path="config.search.cache",
        domain="external",
        subgroup="infra",
        layer="external",
        level="L3",
        band="external",
        tags=["Cache"],
    )
    add(
        id="ollama-embed",
        name="Ollama",
        path="localhost:11434",
        domain="external",
        subgroup="infra",
        layer="external",
        level="L3",
        band="external",
        tags=["Embed"],
    )
    add(
        id="openai-api",
        name="OpenAI 兼容 HTTP",
        path="LiteLLM",
        domain="external",
        subgroup="infra",
        layer="external",
        level="L3",
        band="external",
        tags=["LLM"],
    )

    return rows


def main() -> None:
    rows = build_rows()
    print("const MODULES = {")
    for r in rows:
        print(module_entry_js(r))
    print("};")


if __name__ == "__main__":
    main()
