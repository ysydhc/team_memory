# 架构分层定义（Architecture Layers）

> **用途**：定义 team_doc 分层与依赖方向，供 import 检查脚本与 CI 强制约束。纯 Harness，不依赖 tm。

---

## 一、分层表（L0～L3）

| 层 | 模块 | 路径 | 可依赖 | 禁止依赖 |
|----|------|------|--------|----------|
| **L0 基础** | schemas | `src/team_memory/schemas.py` | 无 | - |
| **L0 基础** | config | `src/team_memory/config.py` | 无 | - |
| **L1 存储** | storage | `src/team_memory/storage/` | L0 | services, web, server, bootstrap |
| **L2 服务** | services | `src/team_memory/services/` | L0, L1 | web, server, bootstrap |
| **L2 服务** | auth | `src/team_memory/auth/` | L0, L1 | web, server, bootstrap |
| **L2 服务** | embedding | `src/team_memory/embedding/` | L0, L1 | web, server, bootstrap |
| **L2 服务** | reranker | `src/team_memory/reranker/` | L0, L1 | web, server, bootstrap |
| **L2 服务** | architecture | `src/team_memory/architecture/` | L0, L1 | web, server, bootstrap |
| **L3 入口** | web | `src/team_memory/web/` | L0, L1, L2 | - |
| **L3 入口** | server | `src/team_memory/server.py` | L0, L1, L2 | - |
| **L3 入口** | bootstrap | `src/team_memory/bootstrap.py` | L0, L1, L2 | - |
| **L3 入口** | workflow_oracle | `src/team_memory/workflow_oracle.py` | L0, L1, L2 | - |

**依赖方向**：只能向前依赖，禁止反向（如 services 不能 import web）。

---

## 二、允许的依赖矩阵

### 2.1 层间依赖

```
L0 (schemas, config)
  ↑
L1 (storage)
  ↑
L2 (services, auth, embedding, reranker, architecture)
  ↑
L3 (web, server, bootstrap, workflow_oracle)
```

| 源层 | 允许依赖 | 禁止依赖 |
|------|----------|----------|
| L0 | 无（仅标准库、第三方） | - |
| L1 | L0 | L2, L3 |
| L2 | L0, L1 | L3（含 web, server, bootstrap） |
| L3 | L0, L1, L2 | - |

### 2.2 同层横向依赖

- **L0**：schemas 与 config 之间**禁止**互相引用（保持无依赖基础层）。
- **L1**：storage 为单模块，无同层横向依赖。
- **L2**：services、auth、embedding、reranker、architecture 之间**允许**互相引用（同属服务层，需协作）。
- **L3**：web、server、bootstrap、workflow_oracle 之间**允许**互相引用；脚本**不校验** L3 内部方向。

---

## 三、特殊规则

1. **bootstrap、server 禁止被 L0–L2 引用**  
   二者为单向依赖汇聚点，L0–L2 模块不得 import `bootstrap` 或 `server`，避免循环依赖。

2. **L3 内部不校验**  
   web、server、bootstrap、workflow_oracle 之间的 import 方向由脚本跳过，不报错。

3. **模块归属**  
   新增模块须先归入某层再实现；不得出现「未分层」模块。

---

## 四、Brownfield 对齐

**Brownfield**：指现有代码/遗留结构。

### 4.1 目录映射

`src/team_memory/<module>` 与分层表一一对应：

| 模块 | 路径 | 层 |
|------|------|-----|
| schemas | `schemas.py` | L0 |
| config | `config.py` | L0 |
| storage | `storage/` | L1 |
| services | `services/` | L2 |
| auth | `auth/` | L2 |
| embedding | `embedding/` | L2 |
| reranker | `reranker/` | L2 |
| architecture | `architecture/` | L2 |
| web | `web/` | L3 |
| server | `server.py` | L3 |
| bootstrap | `bootstrap.py` | L3 |
| workflow_oracle | `workflow_oracle.py` | L3 |

### 4.2 新增模块流程

1. 在本文档分层表中确定归属层；
2. 在允许的依赖矩阵中确认可依赖范围；
3. 实现代码；
4. 通过 import 检查脚本验证。

### 4.3 已知待修复（Phase 3 执行中）

- `architecture` → `web.architecture_models`：反向依赖，需将 architecture_models 迁至 schemas 或 `team_memory.schemas.architecture`
- `auth` → `web.app`：反向依赖，按分层文档方案调整

---

## 五、豁免规则

### 5.1 单行豁免：`# noqa: layer-check`

在 import 行末尾添加注释，该行豁免层检查：

```python
from team_memory.web.app import get_app  # noqa: layer-check
```

**用途**：临时迁移、已知待修复、需人工复审的例外。

### 5.2 类型注解块：`if TYPE_CHECKING:`

`if TYPE_CHECKING:` 块内的 import 仅用于类型注解，运行时不会执行，**豁免**：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team_memory.web.app import FastAPI
```

### 5.3 白名单：exclude_paths

在脚本配置或本文档中声明 `exclude_paths`，指定路径下的文件不参与检查。

**格式示例**：

```yaml
exclude_paths:
  - path: scripts/migrations/legacy_import_fix.py
    reason: 一次性迁移脚本，完成后移除
    review_by: 2025-04-01
```

**要求**：

- 白名单须在本文档中记录路径与理由；
- 约定复审日期（review_by），修复后移除；
- 定期复审，避免长期豁免。

**当前白名单**（空，待补充）：

| 路径 | 理由 | 复审日期 |
|------|------|----------|
| （无） | - | - |

---

## 六、异常处理约定

import 检查脚本遇到以下情况时，**跳过该文件并记录到 stderr**，**不中断**整体检查：

| 情况 | 行为 |
|------|------|
| 语法错误 | 跳过，记录 `syntax error: <file>:<line>` |
| 无法解析文件 | 跳过，记录 `parse failed: <file>` |
| 非 Python 文件 | 跳过，不记录 |
| 空文件 | 跳过，不记录 |

**原则**：单文件异常不影响其他文件检查；所有异常须可追溯。

---

## 七、例外说明

以下为架构上的已知例外，不视为违规：

### 7.1 server.py 同时挂 MCP 与 Web

`server.py` 同时挂载 MCP 与 Web 入口，属于 L3 入口层，可依赖 L0–L2 及 web、bootstrap 等 L3 模块。

### 7.2 bootstrap 跨层初始化

`bootstrap.py` 负责跨层初始化（如数据库、服务、路由），会 import 多层级模块，属于 L3 入口职责。

### 7.3 其他 L3 入口

`workflow_oracle.py` 作为工作流入口，与 server、web、bootstrap 同属 L3，可互相引用。

---

## 八、脚本引用说明

import 检查脚本（如 `scripts/harness_import_check.py`）应：

1. 读取本文档或等价配置，获取分层表与依赖矩阵；
2. 解析 `src/team_memory/**/*.py` 的顶层 import；
3. 按第二、三节规则校验；
4. 应用第五节的豁免规则；
5. 按第六节处理异常。

**检查范围**：默认不包含 `tests/`、迁移脚本；`tests/` 下对 `src/` 的 import 不纳入检查。
