# Python 代码约定

## 工具链

| 工具 | 用途 | 命令 |
|------|------|------|
| ruff | lint + format | `make lint` |
| pytest | 测试 | `make test` |
| alembic | 数据库迁移 | `make migrate` |

配置文件：`pyproject.toml`，禁止在其他地方重复配置 lint 规则。

## 命名约定

```python
# 文件名：snake_case
experience_service.py
search_pipeline.py

# 类名：PascalCase
class ExperienceService: ...
class ExperienceRepository: ...

# Pydantic Schema（请求/响应）：动词+名词+Request/Response
class SearchRequest(BaseModel): ...
class ExperienceCreate(BaseModel): ...

# MCP 工具：tm_ 前缀，snake_case
async def tm_search(...): ...
async def tm_save(...): ...

# 常量：UPPER_SNAKE_CASE
MAX_SEARCH_RESULTS = 20
ROLE_PERMISSIONS: dict[str, set[str]] = {...}

# 私有方法：_前缀
def _build_query_vector(self, text: str): ...
def _active_filter(current_user: str | None): ...
```

## 类型注解

所有函数必须有类型注解，包括返回类型：

```python
# 正确
async def search(self, query: str, limit: int = 5) -> list[ExperienceResult]:
    ...

# 错误——缺少类型注解
async def search(self, query, limit=5):
    ...
```

## 异步规范

```python
# 正确——所有数据库操作使用 async
async def get_by_id(self, experience_id: int) -> Experience | None:
    async with get_session(db_url) as session:
        result = await session.execute(select(Experience).where(...))
        return result.scalar_one_or_none()

# 错误——在 async 函数中使用同步 ORM
def get_by_id(self, experience_id: int):
    with Session() as session:  # 会阻塞事件循环
        return session.get(Experience, experience_id)
```

## 日志规范

```python
import logging
logger = logging.getLogger("team_memory")  # 或 "team_memory.service" 等子模块

# 正确——结构化，包含上下文
logger.info("experience_saved", extra={"experience_id": exp.id, "type": exp.type})
logger.error("search_failed", extra={"query": query, "error": str(e)}, exc_info=True)

# 错误——裸 print，无法被日志系统捕获
print(f"saved: {exp.id}")
```

## 配置读取

```python
# 正确——通过 Settings 对象读取
from team_memory.config import get_settings
settings = get_settings()
db_url = settings.database.url

# 入口层（server.py、web/app.py）可读 os.environ 获取运行时覆盖
# 如 TEAM_MEMORY_API_KEY、TEAM_MEMORY_USER、TEAM_MEMORY_PROJECT
api_key = os.environ.get("TEAM_MEMORY_API_KEY", "")

# 错误——业务逻辑直接读环境变量
import os
db_url = os.environ["DATABASE_URL"]  # 应使用 get_settings().database.url
```

## 数据验证边界

所有外部输入必须经过 Pydantic 验证：

```python
# MCP 工具参数 → 内联 BaseModel 或 schemas 中定义
# HTTP 请求体 → XxxRequest(BaseModel) 在 web/app.py 或 routes 中
# 配置文件值 → Settings(BaseSettings)
# 数据库查询结果映射 → Pydantic model_validate(orm_obj)
```

## 分层约束（强制）

- **Storage 层**：只调用 `storage/models.py`、`storage/database.py`，不 import services/web/server
- **Services 层**：只调用 storage、auth、embedding、reranker，不 import web/server
- **Server/Web 层**：可调用 services，禁止直接 import storage 做 CRUD（除 bootstrap 等初始化场景）

提交前必须通过 `make harness-check`，import 方向由 `scripts/harness_import_check.py` 校验。

## 禁止事项

- 禁止硬编码 API Key、密码、连接串（使用环境变量或配置文件）
- 禁止 `SELECT *`（显式列出需要的字段）
- 禁止在 migration 文件中写业务逻辑
- 禁止在 model 文件中写服务层逻辑
- 禁止捕获所有异常后静默忽略（`except Exception: pass`）
- 禁止裸 `print()` 调试（使用 `logger.debug`）
