# 日志 JSON 格式规范

> **用途**：统一 team_memory 日志输出格式，便于日志聚合、检索、告警。纯 Harness，不依赖 tm。
> **Phase 4 可观测性**：本规范为 Task 2 实现提供设计依据。

---

## 一、JSON 行日志格式

### 1.1 格式定义

每条日志为**单行 JSON 对象**（JSON Lines / NDJSON），便于流式解析、grep、ELK/Loki 等工具消费。

```
{"timestamp":"2025-03-07T10:30:00.123Z","level":"INFO","logger":"team_memory.web","message":"Request completed"}
```

- **编码**：UTF-8
- **换行**：每条日志以 `\n` 结尾，无多行 JSON
- **解析**：每行可独立 `json.loads(line)` 解析

### 1.2 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | string | ISO 8601 格式，如 `2025-03-07T10:30:00.123Z` 或 `2025-03-07T10:30:00.123+08:00` |
| `level` | string | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `logger` | string | Logger 名称，如 `team_memory.web`、`team_memory.bootstrap` |
| `message` | string | 主日志内容，人类可读摘要 |

### 1.3 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | string | 请求追踪 ID，便于关联同一请求的多条日志 |
| `module` | string | 模块路径，如 `team_memory.web.routes.auth`（可与 logger 冗余，用于细粒度过滤） |
| `extra` | object | 键值对，存放业务上下文（如 `path`、`method`、`duration_ms` 等） |

### 1.4 示例

**普通应用日志：**
```json
{"timestamp":"2025-03-07T10:30:00.123Z","level":"INFO","logger":"team_memory.web","message":"Request completed","extra":{"path":"/api/v1/experiences","method":"GET","status":200}}
```

**请求日志（request_logger）：**
```json
{"timestamp":"2025-03-07T10:30:00.124Z","level":"INFO","logger":"team_memory.web.request","message":"request","extra":{"event":"request","path":"/api/v1/experiences","method":"GET","status":200,"duration_ms":12,"ip":"127.0.0.1","user":"admin"}}
```

**带 request_id 的日志：**
```json
{"timestamp":"2025-03-07T10:30:00.125Z","level":"ERROR","logger":"team_memory.web","message":"Unhandled exception","request_id":"req-abc123","extra":{"path":"/api/v1/save","ops_error_id":"web-1a2b"}}
```

---

## 二、与现有 logging.getLogger("team_memory.*") 的衔接

### 2.1 现状

- 各模块使用 `logging.getLogger("team_memory.<module>")`，如：
  - `team_memory.web`、`team_memory.web.request`
  - `team_memory.bootstrap`、`team_memory.auth`、`team_memory.search_pipeline` 等
- 根 logger 或各 Handler 使用默认 `logging.Formatter`，输出 human-readable 文本
- `request_logger`（`team_memory.web.request`）在 `app.py` 中单独配置：自定义 `StreamHandler` + `Formatter("%(message)s")`，直接输出 `json.dumps(payload)`

### 2.2 衔接方式

1. **统一配置入口**：在 `bootstrap.py`（L3）中根据 `config.LOG_FORMAT` 初始化全局 Handler/Formatter，作用于 `logging.getLogger("team_memory")` 及其子 logger（`team_memory.*` 均继承）。
2. **保持 logger 名称不变**：不修改现有 `getLogger("team_memory.xxx")` 调用，仅替换输出格式。
3. **不改变调用方式**：继续使用 `logger.info(...)`、`logger.warning(...)` 等，无需引入 structlog 或改变调用习惯。
4. **实现方式**：优先使用 `python-json-logger` 或标准 `logging` 的 `Formatter` 子类，输出符合本规范的 JSON 行；structlog 可作为备选，但需评估改动面。

---

## 三、request_logger 与 JSON 统一

### 3.1 现状

- `request_logger` 在 `app.py` 中单独配置：`propagate=False`，自定义 Handler + `Formatter("%(message)s")`
- 当前输出：`json.dumps({"event":"request","path":...,"method":...,"status":...,"duration_ms":...,"ip":...,"user":...})`，缺少 `timestamp`、`level`、`logger` 等必填字段

### 3.2 设计决策

| 选项 | 说明 |
|------|------|
| **统一走 JSON** | 当 `LOG_FORMAT=json` 时，request_logger 使用与其它 logger 相同的 JSON Formatter，输出完整必填字段 + `extra` 承载原有 payload |
| **保留单独配置** | request_logger 可保留独立 Handler，但 Formatter 需与全局一致：统一使用 JSON Formatter 或统一使用 human Formatter |

**推荐**：**统一配置**。bootstrap 初始化时，对 `team_memory` 根 logger 配置 Handler；request_logger 移除自定义 Handler，设置 `propagate=True`，使其继承根 logger 的 Handler 与 Formatter。若需隔离 request 日志到单独流，可在 bootstrap 中为 `team_memory.web.request` 单独添加 Handler，但 Formatter 与全局一致（JSON 或 human 二选一）。

### 3.3 实现要点

- 移除 `app.py` 中 request_logger 的硬编码 Handler/Formatter
- 在 bootstrap 中统一配置 `team_memory` 及子 logger 的 Handler；request_logger 的 `propagate` 设为 `True`，或由 bootstrap 显式为其添加与全局一致的 Formatter
- request 日志的 `message` 可固定为 `"request"`，原有 payload 放入 `extra`

---

## 四、开发/生产切换

### 4.1 开关方式

| 方式 | 优先级 | 说明 |
|------|--------|------|
| 环境变量 `LOG_FORMAT` | 高 | `LOG_FORMAT=json` 启用 JSON；`LOG_FORMAT=human` 或未设置时使用 human-readable |
| config.yaml | 中 | 若存在 `logging.format: json`，与 env 合并；env 优先 |
| config.py (Settings) | 实现 | L0 config 仅提供 `LOG_FORMAT` 字段（从 env 或 yaml 读取），不依赖 structlog/python-json-logger |

### 4.2 默认值

| 环境 | 默认格式 | 说明 |
|------|----------|------|
| 开发（本地） | human-readable | 便于人工阅读，`LOG_FORMAT` 未设置时 |
| 生产 / CI | JSON | 通过 `LOG_FORMAT=json` 或 CI 环境变量设置；生产部署时建议显式设置 |

### 4.3 判定逻辑（bootstrap 中）

```python
# 伪代码
format_type = os.environ.get("LOG_FORMAT") or getattr(settings, "log_format", None) or "human"
use_json = format_type.lower() == "json"
```

---

## 五、敏感字段脱敏

### 5.1 脱敏约定

以下字段在写入日志前须脱敏，不得输出明文：

| 类型 | 示例 | 脱敏方式 |
|------|------|----------|
| API Key | `OPENAI_API_KEY`、`TEAM_MEMORY_API_KEY` | 仅保留前 4 位 + `***`，如 `sk-1***` |
| 密码 | `password`、`default_admin_password` | 全文替换为 `***` |
| Token | Bearer token、JWT | 仅保留前 8 位 + `***` |
| 数据库 URL | `postgresql://user:pass@host/db` | 隐藏密码部分，如 `postgresql://user:***@host/db` |

### 5.2 实现方式

- **应用层**：在 `logger.info(..., extra={...})` 传入前，对 `extra` 中敏感键做脱敏
- **Formatter 层**：若使用 python-json-logger 等，可配置 `default_redact_keys` 或自定义序列化逻辑，对已知敏感键自动脱敏
- **约定**：新增日志字段时，若涉及敏感信息，须在本文档或实现中声明并脱敏

### 5.3 敏感键名单（可扩展）

```
api_key, apikey, api-key, password, secret, token, authorization, auth_header
```

---

## 六、实现参考（Python）

### 6.1 使用 python-json-logger

```python
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        if "extra" not in log_record and hasattr(record, "extra"):
            log_record["extra"] = record.extra
```

### 6.2 使用标准 logging

```python
import json
import logging

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "extra"):
            log_record["extra"] = record.extra
        return json.dumps(log_record, ensure_ascii=False)
```

### 6.3 request_logger 输出统一

当 `LOG_FORMAT=json` 时，request 中间件应通过标准 `logger.info(message, extra={...})` 输出，由 Formatter 统一序列化为本规范格式，而非手动 `json.dumps`。

---

## 七、验收清单

- [x] 文档存在：`docs/design-docs/logging-format.md`
- [x] JSON 行格式定义清晰，可被 `json.loads(line)` 解析
- [x] 必填字段：timestamp、level、logger、message
- [x] 可选字段：request_id、module、extra
- [x] 与 `logging.getLogger("team_memory.*")` 衔接方式已写明
- [x] request_logger 是否走 JSON、与 web 自定义 Formatter 的统一方式已写明
- [x] 开发/生产切换：`LOG_FORMAT=json` 或 config 开关，默认值已写明
- [x] 敏感字段脱敏约定已写明
- [x] 格式可被 Python structlog 或 logging 配置实现（含示例）
