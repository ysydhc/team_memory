# TM Daemon 架构重构 — 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有 hooks 直调模式重构为本地 daemon 架构，支持本地/远端 TM 双模，解决 Obsidian 索引脱离 Agent 触发的问题。

**Architecture:** 本地 TM Daemon 常驻运行（launchd 管理），暴露 HTTP API 供 hooks 转发事件，内置 watchfiles 监听 Obsidian vault 变更。所有重逻辑（收敛检测、提炼、索引构建）在 daemon 内完成。TMSink 抽象统一本地/远端 TM 存储访问。Hook 脚本极薄 — 只负责接收 stdin JSON → POST 到 daemon。

**Tech Stack:** FastAPI + uvicorn (已有), watchfiles (已有), httpx (已有), SQLite (草稿缓冲)

---

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│  TM Daemon (localhost:3901, launchd 管理)                    │
│                                                             │
│  ┌─ HTTP API ─────────────────────────────────────────────┐ │
│  │  POST /hooks/after_response                            │ │
│  │  POST /hooks/session_start                             │ │
│  │  POST /hooks/before_prompt                             │ │
│  │  POST /hooks/session_end                               │ │
│  │  GET  /recall?q=...&project=...                        │ │
│  │  POST /draft/save                                      │ │
│  │  POST /draft/publish                                   │ │
│  │  GET  /status                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ File Watcher (watchfiles) ───────────────────────────┐  │
│  │  监听 Obsidian vault 目录                               │  │
│  │  .md 文件变化 → 防抖 → MarkdownIndexer → TMSink.push   │  │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─ 内部模块 (从 scripts/hooks/ 迁入) ───────────────────┐  │
│  │  DraftBuffer (SQLite)                                  │  │
│  │  ConvergenceDetector                                   │  │
│  │  DraftRefiner                                          │  │
│  │  MarkdownIndexer                                       │  │
│  │  SessionTimeoutManager                                 │  │
│  │  TMSink (LocalTMSink / RemoteTMSink)                  │  │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         ↕ TMSink                        ↕ TMSink
   本地 TM (import)                 远端 TM (HTTP)

┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────┐
│ Cursor Hook          │  │ Claude Code Hook     │  │ Hermes      │
│ (thin, curl/python)  │  │ (thin, curl/python)  │  │ (hermes     │
│ → POST localhost:3901│  │ → POST localhost:3901│  │  pipeline)  │
└──────────────────────┘  └──────────────────────┘  └─────────────┘
```

## 数据流

### 草稿流程
1. Cursor afterAgentResponse → hook 读 stdin → POST /hooks/after_response
2. Daemon 累积文本到 DraftBuffer
3. ConvergenceDetector 检测收敛
4. 收敛 → DraftRefiner 提炼 → TMSink.draft_save → TMSink.draft_publish
5. 30min 未收敛 → SessionTimeoutManager 触发提炼

### Obsidian 索引流程
1. watchfiles 检测 vault .md 文件变化
2. 防抖后 MarkdownIndexer 解析
3. TMSink.save (source=obsidian)

### 检索流程
1. Cursor sessionStart/beforeSubmitPrompt → hook → POST /hooks/session_start 或 /hooks/before_prompt
2. Daemon 调 TMSink.recall
3. 返回结果，hook 脚本输出 additional_context

## 与现有代码的关系

| 现有文件 | 处理方式 |
|---------|---------|
| `scripts/hooks/draft_buffer.py` | 迁入 daemon，保留原文件做 import 兼容 |
| `scripts/hooks/convergence_detector.py` | 迁入 daemon |
| `scripts/hooks/draft_refiner.py` | 迁入 daemon，TMClient 换成 TMSink |
| `scripts/hooks/markdown_indexer.py` | 迁入 daemon |
| `scripts/hooks/session_timeout.py` | 迁入 daemon（用 asyncio 定时器替代独立进程） |
| `scripts/hooks/shared.py` | TMClient → 废弃，由 TMSink 替代 |
| `scripts/hooks/common.py` | 保留，hook 薄脚本仍需 parse_hook_input |
| `scripts/hooks/cursor_after_response.py` | 改薄：stdin → POST daemon |
| `scripts/hooks/cursor_session_start.py` | 改薄：stdin → POST daemon |
| `scripts/hooks/cursor_before_prompt.py` | 改薄：stdin → POST daemon |
| `scripts/hooks/claude_session_start.py` | 改薄：stdin → POST daemon |
| `scripts/hooks/git_watcher.py` | 废弃，由 daemon File Watcher 替代 |
| `scripts/hooks/initial_index.py` | 保留为 CLI 工具，改用 TMSink |
| `scripts/hooks/hermes_pipeline.py` | 改薄：调 daemon HTTP API |
| `scripts/hooks/weekly_report.py` | 保留为 CLI 工具，改用 TMSink |
| `scripts/hooks/on_*.py` | 废弃，统一由 thin hooks + daemon 处理 |
| `scripts/hooks/retrieval_trigger.py` | 废弃，daemon 内部处理 |

## 配置变更

`scripts/hooks/config.yaml` 新增：
```yaml
daemon:
  host: "127.0.0.1"
  port: 3901

tm:
  mode: "local"          # local | remote
  base_url: "http://localhost:3900"  # remote 模式下的远端地址

obsidian:
  vaults:
    - path: "/Users/yeshouyou/ObsidianVault"
      project: "knowledge"
      exclude: [".obsidian", ".trash"]
```
