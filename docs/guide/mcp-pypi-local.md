# MCP：本机通过 PyPI 包（`pip install team_memory`）接入

> 运维文档 | 与「克隆本仓库、方案 A 包装脚本」并列的一种接法
> 相关：[mcp-server.md](mcp-server.md)（方案 A、附件、验证清单）· [mcp-lite-default.md](../decision/mcp-lite-default.md) · [README.md](../../README.md)

## 何时用这种方式

| 场景 | 建议 |
|------|------|
| 已安装 PyPI 上的 **`team_memory`**，**不依赖**本仓库 `src/`、`PYTHONPATH`、根目录 `config.*.yaml` 默认解析 | 用本文档：独立 venv + **`python -m team_memory.server`** |
| 日常改 TM 源码、要与 `make mcp` / Alembic / 根配置一致 | 优先 [mcp-server.md](mcp-server.md) **方案 A**（`run_mcp_with_dotenv.sh` + 仓库 `cwd`） |

两者入口模块相同：**`team_memory.server`**，工具均为 **`memory_*`**（见 [mcp-lite-default.md](../decision/mcp-lite-default.md)）。

## 1. 安装

使用**专用虚拟环境**可避免与仓库 `.venv`、系统 Python 混用：

```bash
python3.11 -m venv ~/.venvs/team-memory-mcp
source ~/.venvs/team-memory-mcp/bin/activate
python -m pip install -U pip
pip install "team_memory>=0.2.0"
```

确认版本与模块：

```bash
python -c "import team_memory; print(team_memory.__version__)"
which python
```

记下 **`which python` 的绝对路径**，写入下节 `mcp.json` 的 **`command`**。

## 2. 环境变量（MCP 进程内）

MCP 须能连上**与你的 Web 相同**的 PostgreSQL（pgvector），并携带合法 API Key。

| 变量 | 说明 |
|------|------|
| **`TEAM_MEMORY_DB_URL`** | 通常**必填**（pip 独立运行时无仓库默认 `config`）。`postgresql+asyncpg://用户:密码@主机:端口/库名` |
| **`TEAM_MEMORY_API_KEY`** | 与 Web / 数据库用户 API Key 一致 |
| **`TEAM_MEMORY_PROJECT`** | 推荐，如 `team_memory`，与 `resolve_project` 默认租户一致 |
| **`TEAM_MEMORY_USER`** | 可选；不设 Key 或解析失败时的回退 |
| **`TEAM_MEMORY_CONFIG_PATH`** | 可选；指向**单文件** YAML 时，可替代或补充部分环境配置（与 `load_settings` 行为一致） |

若使用 `TEAM_MEMORY_CONFIG_PATH` 指向本仓库内的 `config.development.yaml`，需保证路径对本机可读。连接串还可通过 **`TEAM_MEMORY_DB_URL`**（项目内常用）或嵌套形式 **`TEAM_MEMORY_DATABASE__URL`** 覆盖 YAML 中的 `database.url`（见 `Settings` / `bootstrap`）。

**安全**：密钥可放在 Cursor **用户级** MCP 配置或本机脚本中 `source` 的 `.env`（勿提交）；与 [mcp-server.md](mcp-server.md) 方案 A 相比，pip 模式常把变量写在 `mcp.json` 的 **`env`**，更应注意勿把含密钥的 JSON 提交到 Git。

## 3. Cursor `mcp.json` 示例

**不要求** `cwd` 指向 `team_doc` 源码树（除非仅用 `TEAM_MEMORY_CONFIG_PATH` 指回仓库配置）。

```json
{
  "mcpServers": {
    "team_memory": {
      "command": "/Users/你的用户名/.venvs/team-memory-mcp/bin/python",
      "args": ["-m", "team_memory.server"],
      "env": {
        "TEAM_MEMORY_DB_URL": "postgresql+asyncpg://developer:devpass@localhost:5433/team_memory",
        "TEAM_MEMORY_API_KEY": "你的APIKey",
        "TEAM_MEMORY_PROJECT": "team_memory"
      }
    }
  }
}
```

将 `command` 换成你机器上 **步骤 1** 的 `which python` 输出；`TEAM_MEMORY_DB_URL` 与端口以实际 `docker compose` / 部署为准。

## 4. 验证

1. 保存配置后 **重载 MCP** 或重启 Cursor。
2. 对话中调用 **`memory_context`** 或 **`memory_recall`**，应返回 JSON，而非 DB/鉴权错误。
3. 命令行等价冒烟（变量与 MCP 一致）：

```bash
export TEAM_MEMORY_DB_URL="postgresql+asyncpg://..."
export TEAM_MEMORY_API_KEY="..."
export TEAM_MEMORY_PROJECT="team_memory"
/path/to/team-memory-mcp/bin/python -m team_memory.server
```

进程以 stdio 挂起且无启动异常即可 Ctrl+C 退出。

## 5. 升级 PyPI 包

```bash
source ~/.venvs/team-memory-mcp/bin/activate
pip install -U "team_memory>=0.2.0"
```

升级后若变更了依赖或入口，可再次执行 `python -c "import team_memory; print(team_memory.__version__)"` 确认。

## 6. `tm-cli` 与附件

PyPI 包提供控制台入口 **`tm-cli`**（与 `python -m team_memory.cli` 等价）。档案馆附件上传仍走 HTTP multipart；本机需 **`TEAM_MEMORY_API_KEY`**、**`TM_BASE_URL`**（默认 `http://localhost:9111`），详见 [mcp-server.md](mcp-server.md) 档案馆与 CLI 说明。

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-07 | 初稿：pip + 独立 venv、环境变量、无 `cwd` 的 `mcp.json` 示例 |
