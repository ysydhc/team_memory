# 日志系统设计文档

> **状态**：✅ 已完成  
> **执行记录**：[execute.md](../2-plan/execute.md) · [复盘](../3-retro/retro.md)

> **创建日期**：2025-03-10  
> **Plan ID**：logging-system-impl  
> **用途**：为 io_logger、文件日志、QueueListener 等实现提供设计依据。

---

## 一、配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `LOG_IO_ENABLED` | bool | false | 是否启用 I/O 日志（io_logger） |
| `LOG_IO_DETAIL` | str | "mcp" | 日志粒度：mcp / service / pipeline / full |
| `LOG_IO_TRUNCATE` | int | 300 | 单条日志超过该字符数时截断（0=不截断） |
| `LOG_FILE_ENABLED` | bool | false | 是否启用文件日志 |
| `LOG_FILE_PATH` | str | "logs/team_memory.log" | 日志文件路径 |
| `LOG_FILE_MAX_BYTES` | int | 10 * 1024 * 1024 | 单文件最大字节数（10M）；DEBUG 下不限制 |
| `LOG_FILE_BACKUP_COUNT` | int | 5 | 轮转后保留的备份文件数 |

**环境变量**：通过 `TEAM_MEMORY_LOG_IO_ENABLED`、`TEAM_MEMORY_LOG_FILE_PATH` 等覆盖；或 config.yaml 中 `logging` 节配置。

---

## 二、内部节点映射（detail 级别）

当 `LOG_IO_DETAIL` 为 `detail` 或需要按节点过滤时，使用以下节点 ID 表：

| 节点 ID | 名称 | 对应层级 | 说明 |
|---------|------|----------|------|
| `mcp` | MCP 层 | track_usage | server.py 中 @track_usage 装饰器内；MCP 工具调用入口/出口 |
| `service` | 服务层 | ExperienceService | experience.py 中 save/search/feedback 等业务方法 |
| `pipeline` | 管道层 | SearchPipeline | search_pipeline.py 中检索、重排、LLM 扩展等步骤 |
| `full` | 全链路 | 以上全部 | 包含 mcp + service + pipeline 的完整调用链 |

**使用方式**：io_logger 在记录时写入 `node_id` 字段；`LOG_IO_DETAIL` 控制输出哪些节点的日志（如 `mcp` 仅 MCP 层，`full` 输出全部）。

---

## 三、架构

### 3.1 分层与职责

```
┌─────────────────────────────────────────────────────────────────┐
│  L0: config.py                                                   │
│  - LOG_IO_*, LOG_FILE_* 配置项（Settings）                       │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  L3: bootstrap.py                                                │
│  - _configure_logging：根据 config 初始化 Handler/Formatter       │
│  - 添加 FileHandler 时创建 QueueHandler + QueueListener           │
│  - stop_background_tasks 中调用 listener.stop()                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────────┐
│ track_usage   │         │ SearchPipeline  │         │ ExperienceService  │
│ (server.py)   │         │ (search_        │         │ (experience.py)     │
│               │         │  pipeline.py)   │         │                     │
│ MCP 工具入口  │         │ 检索、重排、扩展 │         │ save/search/feedback│
│ 调用 io_logger│         │ 调用 io_logger  │         │ 调用 io_logger      │
└───────────────┘         └─────────────────┘         └─────────────────────┘
```

### 3.2 调用关系

- **io_logger**（L0 概念，实际为 `logging.getLogger("team_memory.io")`）：被 track_usage、SearchPipeline、ExperienceService 独立调用；不依赖 mcp_debug_log。
- **bootstrap**：统一配置入口，在 `_configure_logging` 中根据 config 添加 StreamHandler、可选的 FileHandler。
- **track_usage**：MCP 工具装饰器，在工具调用前后调用 io_logger。
- **SearchPipeline**：检索管道，在关键步骤（向量检索、FTS、重排、LLM 扩展）调用 io_logger。
- **ExperienceService**：业务服务，在 save/search/feedback 等入口调用 io_logger。

---

## 四、DEBUG 下行为

- **文件大小**：当 `LOG_LEVEL=DEBUG` 或 `TEAM_MEMORY_DEBUG=1` 时，`LOG_FILE_MAX_BYTES` 不生效，即不限制单文件大小（便于本地调试时保留完整日志）。
- **判定**：`is_debug = os.environ.get("TEAM_MEMORY_DEBUG", "0") == "1" or logging.getLogger("team_memory").level == logging.DEBUG`。

---

## 五、文件 I/O 耗时与 QueueListener

### 5.1 设计目标

避免文件写入阻塞主线程/事件循环，将 I/O 移至后台线程。

### 5.2 实现方式

1. **QueueHandler + QueueListener**：
   - 主线程：Logger → QueueHandler → 有界队列（`queue.Queue(maxsize=10000)`）
   - 后台线程：QueueListener 从队列取 LogRecord，交给 FileHandler 写入磁盘

2. **队列有界**：`maxsize=10000`，防止日志暴增导致 OOM。

3. **队列满时**：`put(block=False)` 失败则丢弃该条日志，并打 `logger.warning("io_log queue full, dropping log record")`。

4. **应用退出**：在 `stop_background_tasks` 中：
   ```python
   # 伪代码
   asyncio.to_thread(listener.stop)
   await asyncio.wait_for(asyncio.to_thread(lambda: listener.wait_for_queue_empty(timeout=5)), timeout=5)
   ```
   确保队列中剩余记录被处理完毕，超时 5 秒后强制结束。

### 5.3 QueueListener 生命周期

| 时机 | 动作 |
|------|------|
| **start()** | 在 `_configure_logging` 添加 FileHandler 时，创建 QueueHandler + QueueListener，调用 `listener.start()` |
| **stop()** | 在 `stop_background_tasks` 中调用 `listener.stop()`，并等待队列清空（超时 5s） |

---

## 六、io_logger 与 mcp_debug_log 边界

| 特性 | io_logger | mcp_debug_log |
|------|-----------|---------------|
| 用途 | 结构化 I/O 可观测性（生产/CI 可选） | MCP 输入输出调试（仅本地排查） |
| 开关 | LOG_IO_ENABLED | TEAM_MEMORY_DEBUG / TEAM_MEMORY_MCP_DEBUG |
| 调用位置 | track_usage、SearchPipeline、ExperienceService | track_usage 内 |
| 关系 | **独立调用**，可同时开启 | **独立调用**，可同时开启 |

**结论**：两者不合并不替换；可同时开启，各自按配置生效。

---

## 七、生产环境建议

- **10M 默认偏小**：生产环境日志量较大时，可配置 `LOG_FILE_MAX_BYTES` 更大（如 50M、100M）。
- **按天保留**：可选实现按天轮转（如 `TimedRotatingFileHandler`），保留 N 天；当前设计为按大小轮转（`RotatingFileHandler`），`LOG_FILE_BACKUP_COUNT` 控制备份数。
- **配置示例**：
  ```yaml
  # config.production.yaml
  logging:
    file_enabled: true
    file_path: /var/log/team_memory/app.log
    file_max_bytes: 52428800   # 50M
    file_backup_count: 30      # 保留 30 个备份
  ```

---

## 八、验收清单

- [ ] 配置项：LOG_IO_ENABLED, LOG_IO_DETAIL, LOG_IO_TRUNCATE, LOG_FILE_ENABLED, LOG_FILE_PATH, LOG_FILE_MAX_BYTES, LOG_FILE_BACKUP_COUNT 已实现
- [ ] 内部节点映射表（mcp/service/pipeline/full）已用于 detail 级别过滤
- [ ] io_logger、bootstrap、track_usage、SearchPipeline、ExperienceService 架构符合设计
- [ ] DEBUG 下不限制文件大小
- [ ] QueueHandler + QueueListener 实现，队列 maxsize=10000，满时丢弃并 warning
- [ ] QueueListener start 在 _configure_logging，stop 在 stop_background_tasks
- [ ] io_logger 与 mcp_debug_log 独立，可同时开启
