# MCP Server（Cursor 集成）

> 运维文档 | MCP 启动、配置、验证  
> **入口**：`team_memory.server`（仅 `memory_*`）— 见 [mcp-lite-default.md](../decision/mcp-lite-default.md)  
> 相关：[quick-start 快速启动](quick-start.md) | [web-server Web 服务](web-server.md)

MCP 给 Cursor / Claude Desktop / Claude Code 里的 Agent 用，与 **Web** 是独立进程，可同时运行。

**已安装 PyPI 包、不用克隆仓库跑 MCP**：见 [mcp-pypi-local.md](mcp-pypi-local.md)（独立 venv、`TEAM_MEMORY_DB_URL`、无源码 `cwd`）。

## 推荐：方案 A（`.env` + 包装脚本，密钥不入 mcp.json）

**原则**：`TEAM_MEMORY_API_KEY`、`TEAM_MEMORY_PROJECT` 放在**仓库根** **`.env`**（已由 `.gitignore` 忽略）；MCP 的 JSON 只声明 **`bash` + [scripts/run_mcp_with_dotenv.sh](../../scripts/run_mcp_with_dotenv.sh) + `cwd`**，不把密钥写进 JSON。换工程时主要改各自 `.env`，**不必**在 JSON 里改 project/Key。

**`project` 解析**（代码顺序，见 `src/team_memory/utils/project.py`）：工具入参 **`project`** > 环境变量 **`TEAM_MEMORY_PROJECT`** > 配置 **`default_project`** > `"default"`。

1. 复制模板 **[env.team-memory.example](../../example/env.team-memory.example)** 为仓库根 **`.env`**，填写至少：
   - **`TEAM_MEMORY_API_KEY`**（与 Web API Key 登录一致）
   - **`TEAM_MEMORY_PROJECT`**（TeamMemory 租户名，如 `team_memory`）
2. 确保脚本可执行：`chmod +x scripts/run_mcp_with_dotenv.sh`
3. 将 **[cursor-mcp-team-memory.example.json](../../example/cursor-mcp-team-memory.example.json)** 中的 **`/ABSOLUTE/PATH/TO/team_doc`** 换成你的本机路径，写入 **`.cursor/mcp.json`**（Cursor）和/或 **`.mcp.json`**（Claude Code 等）；两文件内容应**同构**，避免分叉。
4. 重启 IDE 或重载 MCP。

**本仓库终端**：`make mcp` 会调用该包装脚本（需存在 `.env`）。

### 与旧方式的对比

| 做法 | 说明 |
|------|------|
| 推荐 | `command`=`bash`，`args`=[绝对路径/`run_mcp_with_dotenv.sh`]，`cwd`=仓库根；**无** `env` 块中的 Key |
| 不推荐 | 在 mcp.json `env` 里写 `TEAM_MEMORY_API_KEY`（易泄露、多份维护） |

### 故障排查

| 现象 | 处理 |
|------|------|
| 脚本提示 missing `.env` | 从 `example/env.team-memory.example` 复制并填写 |
| MCP / `No authenticated user` | `.env` 中 Key 无效，或 MCP 未通过包装脚本启动导致未加载 `.env` |
| 数据总是进了 `default` 项目 | 检查 `.env` 是否设置 `TEAM_MEMORY_PROJECT`，或工具调用是否显式传 `project` |

**档案馆（不增加 MCP 工具的前提）**：结构化文案用 MCP **`memory_archive_upsert`**；**附件**用 HTTP **`POST /api/v1/archives/{archive_id}/attachments/upload`** 或 **`python -m team_memory.cli upload`**（终端需能读到 **`TEAM_MEMORY_API_KEY`**，可与 `.env` 同源：`set -a && source .env`）。设计背景已迁入 Team Memory 档案馆（标题 **`【docs-plans】mcp-archive-api-redesign`**，可 `memory_recall(include_archives=true)` 或 Web 档案馆检索）。

### 多仓库归类 + 附件（简要）

| 环节 | 做法 |
|------|------|
| **多项目** | 各仓库**独立** `.env` 中的 **`TEAM_MEMORY_PROJECT`**；或单次工具调用传 **`project`** |
| **终端 curl/tm-cli** | 同一 shell `source .env` 后再执行，或与 MCP **不是**同一份进程环境 |
| **附件** | 先 `memory_archive_upsert` 拿 `archive_id`，再 CLI/HTTP 上传；可加 `?project=` / `--project` |

---

## MCP 启动

**本仓库**：`make mcp` → `scripts/run_mcp_with_dotenv.sh` → `python -m team_memory.server`（需 `.env`）。

不通过 Makefile、但已配置 `.env` 时也可：

```bash
cd /path/to/team_doc
bash scripts/run_mcp_with_dotenv.sh
```

### `memory_*` 工具（6 个）

| 工具名 | 用途（概要） |
|--------|----------------|
| `memory_context` | 当前任务上下文 + 画像 / 相关经验 |
| `memory_recall` | 按 problem / query / file 路径检索经验（可选档案预览） |
| `memory_save` | 直接保存或 `content` 长文解析保存（`scope=archive` 已移除） |
| `memory_get_archive` | 按 `archive_id` 拉取档案 L2 全文 |
| `memory_archive_upsert` | 创建/更新档案馆记录（与 `POST /api/v1/archives` 一致；不接文件字节） |
| `memory_feedback` | 对检索结果评分 |

`memory_recall` 在 `solve` / `search` / `suggest` 返回的 JSON 顶层可能包含 **`reranked`**（布尔）。`memory_context` 可能包含 **`search_reranked`**。

### MCP vs HTTP（档案馆相关）

| 操作 | 推荐入口 |
|------|----------|
| 档案馆 upsert | MCP `memory_archive_upsert` 或 HTTP `POST /api/v1/archives` |
| 附件 / 大文件 | HTTP multipart 或 **`python -m team_memory.cli upload`** |

当前 MCP **未** 注册 Resources / Prompts。

## 配置 Cursor / Claude Code（mcp.json 最小示例）

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "bash",
      "args": ["/ABSOLUTE/PATH/TO/team_doc/scripts/run_mcp_with_dotenv.sh"],
      "cwd": "/ABSOLUTE/PATH/TO/team_doc"
    }
  }
}
```

完整模板见 [cursor-mcp-team-memory.example.json](../../example/cursor-mcp-team-memory.example.json)。**Claude Code** 可使用仓库根 `.mcp.json`；**Cursor** 通常使用 `.cursor/mcp.json`，**内容保持一致**即可。

## CLI 兼容层

所有 6 个 MCP 工具均可通过 `tm-cli` 命令行等价调用，无需 MCP 协议支持。

### 前提

- Web 服务运行中（`make dev` 或 `make web`）
- 环境变量 `TEAM_MEMORY_API_KEY` 已设置

### 命令速查

| 工具 | CLI 命令 | 示例 |
|------|---------|------|
| memory_save | `tm-cli save` | `tm-cli save --title "Bug fix" --problem "..." --solution "..."` |
| memory_recall | `tm-cli recall` | `tm-cli recall --query "MCP 配置"` |
| memory_context | `tm-cli context` | `tm-cli context --file-paths "src/server.py"` |
| memory_get_archive | `tm-cli get-archive` | `tm-cli get-archive --id "uuid"` |
| memory_archive_upsert | `tm-cli archive` | `tm-cli archive --title "..." --solution-doc "..."` |
| memory_feedback | `tm-cli feedback` | `tm-cli feedback --experience-id "uuid" --rating 5` |

### REST API

CLI 底层调用 `/api/v1/mcp/*` 端点，认证方式与 Web API 一致（`Authorization: Bearer $TEAM_MEMORY_API_KEY`）。

## 验证 MCP

1. `make mcp-verify`（工具注册数量与名称）
2. IDE 重载 MCP 后对话中试用 **`memory_context`** / **`memory_recall`**

若仍看到 **`tm_search`** 等旧工具名，请改为 **`-m team_memory.server`**（经包装脚本）并升级包。

## 资源（Resources）

当前构建的 `server.py` **未** 暴露 MCP resources。

---

## 落地执行清单

| 步骤 | 动作 |
|------|------|
| 1 | `make setup` 或至少 DB 就绪 + `alembic upgrade head` |
| 2 | [env.team-memory.example](../../example/env.team-memory.example) → 复制为仓库根 `.env` 并填写 |
| 3 | `chmod +x scripts/run_mcp_with_dotenv.sh` |
| 4 | `make mcp-verify` |
| 5 | 按示例配置 `.cursor/mcp.json` / `.mcp.json`，**勿在 JSON 写 Key** |
| 6 | 重载 MCP 后验证工具；附件见上文 CLI/HTTP |
