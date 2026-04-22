# 阶段 1：TM 内核扩展

## 项目背景

TM（TeamMemory）目前是为"Agent 主动调用"设计的——memory_save/recall 都是 Agent 主动调的。
我们需要让 TM 支持"管线模式"：

1. **管线写入**：Hook 脚本自动写入草稿，不是 Agent 主动调的
2. **draft 来源约束**：draft 状态的 Experience 只能来自管线（source=pipeline）
3. **promoted 状态**：L2 痕迹被提升为 L3 知识后，标记为 promoted
4. **意图路由**：检索前先分类意图，不同意图走不同检索策略
5. **评估标记**：检索结果带 [mem:xxx] 标记，自动评估 Agent 是否使用
6. **检索日志**：记录每次检索的 query、意图、结果、是否被使用

## 预计改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `storage/models.py` | 修改 | Experience 新增 promoted 状态 |
| `storage/models.py` | 修改 | 新增 SearchLog 模型 |
| `config/janitor.py` | 修改 | 新增 promotion 配置项 |
| `services/janitor.py` | 修改 | 新增 run_promotion() |
| `services/search_orchestrator.py` | 修改 | 新增意图路由接口 |
| `services/intent_router.py` | 新建 | 意图路由实现 |
| `services/evaluation.py` | 新建 | 评估服务（[mem:xxx] 注入 + was_used 判定） |
| `server.py` | 修改 | 新增 memory_draft_save / memory_draft_publish MCP 工具 |
| `services/memory_operations.py` | 修改 | 新增 op_draft_save / op_draft_publish |
| `web/routes/search.py` | 修改 | 检索结果注入 [mem:xxx] 标记 |
| `web/routes/evaluation.py` | 新建 | 评估相关 API |
| `migrations/versions/012_*.py` | 新建 | promoted 状态 + SearchLog 表 |

## 架构图

### 旧架构：检索管线

```
Agent 查询
    ↓
SearchOrchestrator.search()
    ↓
SearchPipeline.execute()
    ├── 1. Cache check
    ├── 2. Embedding
    ├── 3. Retrieval (vector + FTS)
    ├── 4. RRF Fusion
    ├── 5. Exact match boost
    ├── 6. Adaptive Filtering
    ├── 7. Rerank
    ├── 8. Confidence labeling
    ├── 9. Archive merging
    └── 10. Cache store
    ↓
返回结果（无评估标记）
```

### 新架构：检索管线

```
Agent 查询
    ↓
IntentRouter.classify()          ← 新增：意图路由
    ├── 事实型 / 探索型 / 时序型 / 因果型
    └── 输出 IntentResult（type + 检索参数建议）
    ↓
SearchOrchestrator.search(intent) ← 改造：接收 intent 参数
    ↓
SearchPipeline.execute()
    ├── （10 个阶段不变）
    └── 新增：draft Experience 分数 ×0.7
    ↓
EvaluationService.inject_markers() ← 新增：注入 [mem:xxx] 标记
    ↓
SearchLog 写入                    ← 新增：记录检索日志
    ↓
返回结果（带 [mem:xxx] 标记）
```

### 旧架构：Experience 状态机

```
  draft ──→ published
    ↓
  (30天过期，soft delete)
```

### 新架构：Experience 状态机

```
  draft ──→ published ──→ promoted
    ↓         ↓            ↓
  (管线专用) (正式痕迹)  (已提升为 L3 知识)
  
  source=pipeline → 才能设 draft
  source=api/manual → 只能设 review/published
```

### 新增模型：SearchLog

```
┌────────────────────────────────────────────────┐
│  search_logs                                    │
│                                                │
│  id: UUID (PK)                                 │
│  query: TEXT                                    │
│  intent_type: STRING(20)  -- factual/exploratory/... │
│  project: STRING(100)                           │
│  source: STRING(20) -- mcp/api/hook            │
│  result_ids: JSON  -- [{id, score, source_layer}] │
│  was_used: BOOL (nullable) -- True/False/None  │
│  agent_response_snippet: TEXT (nullable)        │
│  created_at: TIMESTAMP                          │
└────────────────────────────────────────────────┘
```

## 子任务拆分

---

### 任务 1-1：Experience 新增 promoted 状态 + source 约束

**描述**：Experience 的 exp_status 新增 "promoted" 取值，source=pipeline 时才能设 draft。

**TDD 流程**：
1. 写测试：`tests/test_experience_promoted.py`
   - 测试 Experience 可以设 exp_status="promoted"
   - 测试 source=pipeline 时可以设 exp_status="draft"
   - 测试 source=api 时设 exp_status="draft" 会报错
   - 测试 promoted 的 Experience 不再被检索返回（除非显式指定）
2. 修改 `storage/models.py`
3. 修改 `services/experience.py` 的 save 逻辑，加入 source+status 约束
4. 新增 migration：`migrations/versions/012_add_promoted_status.py`
5. 验证测试通过

**关键代码修改**：

`storage/models.py` — Experience.exp_status 注释更新：
```python
# 现有
exp_status: Mapped[str] = mapped_column(
    String(20), default="draft", nullable=False, server_default="draft"
)  # draft, published

# 修改后
exp_status: Mapped[str] = mapped_column(
    String(20), default="draft", nullable=False, server_default="draft"
)  # draft, published, promoted
```

`services/experience.py` — save 方法新增约束：
```python
# 在 save 方法开头新增约束检查
def _validate_source_status(source: str, exp_status: str):
    """draft 状态只允许 source=pipeline"""
    if exp_status == "draft" and source != "pipeline":
        raise ValueError(f"exp_status='draft' requires source='pipeline', got source='{source}'")
```

`services/search_orchestrator.py` — promoted 不参与默认检索：
```python
# search 方法中，默认排除 promoted 状态
# 但 draft 参与检索（分数打折 ×0.7）
```

**验收标准**：
- [ ] `Experience(exp_status="promoted")` 能正常创建
- [ ] `Experience(source="api", exp_status="draft")` 抛出 ValueError
- [ ] `Experience(source="pipeline", exp_status="draft")` 正常创建
- [ ] 默认检索不返回 promoted 的 Experience
- [ ] draft 的 Experience 参与检索但分数 ×0.7
- [ ] migration 能正常运行：`make migrate`
- [ ] 所有测试通过：`make test`

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 1-2：新增 SearchLog 模型 + migration

**描述**：新增 search_logs 表，记录每次检索的 query、意图、结果、是否被使用。

**TDD 流程**：
1. 写测试：`tests/test_search_log.py`
   - 测试能创建 SearchLog
   - 测试能按时间范围查询 SearchLog
   - 测试能更新 was_used 字段
   - 测试能统计使用率
2. 修改 `storage/models.py` — 新增 SearchLog 类
3. 新增 `storage/search_log_repository.py`
4. 新增 migration
5. 验证测试通过

**关键代码**：

`storage/models.py` — 新增：
```python
class SearchLog(Base):
    """Search query log for evaluation."""
    __tablename__ = "search_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    intent_type: Mapped[str] = mapped_column(String(20), default="unknown")
    project: Mapped[str] = mapped_column(String(100), default="default")
    source: Mapped[str] = mapped_column(String(20), default="mcp")
    result_ids: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # [{"id": "exp-xxx", "score": 0.85, "source_layer": "L2"}, ...]
    was_used: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # None=未判定
    agent_response_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
```

`storage/search_log_repository.py`：
```python
class SearchLogRepository:
    async def create(self, query, intent_type, project, source, result_ids) -> SearchLog
    async def mark_used(self, log_id: uuid.UUID, agent_snippet: str) -> None
    async def get_stats(self, days: int = 7) -> dict:
        """返回 {total, hit, used, use_rate}"""
```

**验收标准**：
- [ ] SearchLog 能正常创建和查询
- [ ] `mark_used()` 能更新 was_used=True
- [ ] `get_stats(7)` 返回最近 7 天的使用率统计
- [ ] migration 能正常运行
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 1-3：新增意图路由接口 + 默认实现

**描述**：在 SearchOrchestrator 前面加一层意图路由，先分类查询意图，再决定检索策略。

**TDD 流程**：
1. 写测试：`tests/test_intent_router.py`
   - 测试 DefaultIntentRouter：所有查询返回 "general" 类型
   - 测试 IntentRouter 接口可以被替换
   - 测试 SearchOrchestrator 接收 intent 参数后正确传递
2. 新建 `services/intent_router.py`
3. 修改 `services/search_orchestrator.py`
4. 验证测试通过

**关键代码**：

`services/intent_router.py`（新建）：
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class IntentResult:
    """意图分类结果"""
    intent_type: str  # factual / exploratory / temporal / causal / general
    params: dict      # 检索参数建议

class IntentRouter(ABC):
    """意图路由抽象基类"""
    @abstractmethod
    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        pass

class DefaultIntentRouter(IntentRouter):
    """默认实现：所有查询返回 general，不改变检索策略"""
    async def classify(self, query: str, context: dict | None = None) -> IntentResult:
        return IntentResult(intent_type="general", params={})
```

`services/search_orchestrator.py` — 修改构造函数和 search 方法：
```python
class SearchOrchestrator:
    def __init__(
        self,
        search_pipeline: object | None,
        embedding_provider: EmbeddingProvider,
        db_url: str,
        intent_router: IntentRouter | None = None,  # ← 新增
    ) -> None:
        self._search_pipeline = search_pipeline
        self._embedding = embedding_provider
        self._db_url = db_url
        self._intent_router = intent_router or DefaultIntentRouter()  # ← 新增

    async def search(self, query, ...):
        # ← 新增：意图路由
        intent = await self._intent_router.classify(query)
        # 根据 intent.params 调整检索参数
        # 当前只记录 intent_type，不改变检索策略
        # ...
```

`bootstrap.py` — AppContext 新增 intent_router：
```python
class AppContext:
    # ... 现有字段
    intent_router: IntentRouter | None = None  # ← 新增
```

**验收标准**：
- [ ] IntentRouter 是抽象基类，可以被继承替换
- [ ] DefaultIntentRouter 对所有查询返回 intent_type="general"
- [ ] SearchOrchestrator 接受 IntentRouter 参数
- [ ] 不传 IntentRouter 时使用 DefaultIntentRouter（向后兼容）
- [ ] 现有测试不受影响：`make test`
- [ ] 新测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 1-4：评估服务 — [mem:xxx] 注入 + was_used 判定

**描述**：检索结果注入 [mem:xxx] 标记，提供 was_used 判定方法。

**TDD 流程**：
1. 写测试：`tests/test_evaluation.py`
   - 测试 `inject_markers()`：给检索结果附加 [mem:exp-xxx] 标记
   - 测试 `check_was_used()`：Agent 回复包含 [mem:exp-xxx] → was_used=True
   - 测试 `check_was_used_fuzzy()`：Agent 回复不包含标记但内容相似 → was_used=True
   - 测试 `get_weekly_stats()`：返回使用率统计
2. 新建 `services/evaluation.py`
3. 修改 `web/routes/search.py` 或 `server.py` 的检索返回逻辑
4. 验证测试通过

**关键代码**：

`services/evaluation.py`（新建）：
```python
class EvaluationService:
    def __init__(self, db_url: str, embedding_provider=None):
        self._db_url = db_url
        self._embedding = embedding_provider

    def inject_markers(self, results: list[dict]) -> list[dict]:
        """给每个检索结果的 content 追加 [mem:exp-xxx] 标记"""
        for r in results:
            exp_id = r.get("id", "")
            r["_marker"] = f"[mem:{exp_id}]"
            # 在 description/solution 后追加标记
            if "solution" in r and r["solution"]:
                r["solution"] += f" {r['_marker']}"
            elif "description" in r:
                r["description"] += f" {r['_marker']}"
        return results

    def check_was_used(self, agent_response: str, result_ids: list[str]) -> dict[str, bool]:
        """检查 Agent 回复中是否包含标记"""
        used = {}
        for rid in result_ids:
            marker = f"[mem:{rid}]"
            used[rid] = marker in agent_response
        return used

    async def check_was_used_fuzzy(self, agent_response: str, results: list[dict]) -> dict[str, bool]:
        """模糊匹配：Agent 回复和检索结果的嵌入相似度"""
        # 如果精确匹配失败，用嵌入相似度 > 0.8 判定
        pass

    async def get_weekly_stats(self) -> dict:
        """返回本周评估统计"""
        # 调 SearchLogRepository.get_stats()
        pass
```

**SearchOrchestrator.search() 修改**：
```python
# 在返回结果前，注入 [mem:xxx] 标记
if self._evaluation_service:
    result.results = self._evaluation_service.inject_markers(result.results)
    # 写入 SearchLog
```

**验收标准**：
- [ ] 检索结果的 solution/description 后有 [mem:exp-xxx] 标记
- [ ] `check_was_used("根据经验 [mem:exp-123]...", ["exp-123"])` 返回 {"exp-123": True}
- [ ] `check_was_used("试试 TUN 模式", ["exp-123"])` 返回 {"exp-123": False}
- [ ] `inject_markers()` 不改变检索结果的原始内容（只追加标记）
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 1-5：Janitor 新增 promotion 任务

**描述**：Janitor 新增第 6 个任务 run_promotion()，自动提升符合条件的 L2 Experience。

**TDD 流程**：
1. 写测试：`tests/test_janitor_promotion.py`
   - 测试 use_count≥3 的 Experience 被标记为 promoted
   - 测试 同 group_key≥5 条的 Experience 被标记为 promoted
   - 测试 promoted 不再参与默认检索
   - 测试 JanitorConfig 新增的 promotion 配置项
   - 测试 run_all() 包含 promotion 步骤
2. 修改 `config/janitor.py`
3. 修改 `services/janitor.py`
4. 验证测试通过

**关键代码**：

`config/janitor.py` — 新增配置项：
```python
@dataclass
class JanitorConfig:
    # ... 现有配置
    
    # ← 新增
    promotion_enabled: bool = True
    """是否启用 L2→L3 自动提升"""

    promotion_use_count_threshold: int = 3
    """use_count 达到此值触发提升"""

    promotion_group_key_threshold: int = 5
    """同 group_key 达到此条数触发提升"""
```

`services/janitor.py` — 新增方法：
```python
async def run_promotion(self, project: str | None = None) -> dict:
    """L2→L3 自动提升"""
    use_count_threshold = self._get_config("promotion_use_count_threshold", 3)
    group_key_threshold = self._get_config("promotion_group_key_threshold", 5)
    
    promoted_by_use_count = 0
    promoted_by_group = 0
    
    async with self._session() as session:
        repo = ExperienceRepository(session)
        
        # 1. 按 use_count 提升
        high_use = await repo.find_by_use_count(
            min_count=use_count_threshold, 
            status="published",
            project=project
        )
        for exp in high_use:
            exp.exp_status = "promoted"
            promoted_by_use_count += 1
        
        # 2. 按 group_key 聚合提升
        groups = await repo.find_groups_above_threshold(
            min_count=group_key_threshold,
            status="published",
            project=project
        )
        for group_exps in groups:
            for exp in group_exps:
                exp.exp_status = "promoted"
                promoted_by_group += 1
        
        await session.commit()
    
    return {
        "promoted_by_use_count": promoted_by_use_count,
        "promoted_by_group": promoted_by_group,
        "total": promoted_by_use_count + promoted_by_group,
    }
```

`services/janitor.py` — 修改 run_all()：
```python
async def run_all(self, project=None):
    results = {}
    # 1-5 不变
    # ...
    
    # 6. Promotion ← 新增
    if self._get_config("promotion_enabled", True):
        results["promotion"] = await self.run_promotion(project)
```

**验收标准**：
- [ ] use_count=3 的 published Experience 在 run_promotion 后变为 promoted
- [ ] 同 group_key 有 5 条 published Experience，全部变为 promoted
- [ ] promoted 的 Experience 不再出现在默认检索结果中
- [ ] JanitorConfig 新增 3 个配置项有默认值
- [ ] run_all() 的结果包含 "promotion" key
- [ ] promotion_enabled=False 时跳过 promotion
- [ ] 所有测试通过：`make test`

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 1-6：MCP 新增管线专用工具

**描述**：新增 memory_draft_save 和 memory_draft_publish 两个 MCP 工具，和现有 memory_save 区分。

**TDD 流程**：
1. 写测试：`tests/test_mcp_draft_tools.py`
   - 测试 memory_draft_save 创建 source=pipeline + exp_status=draft 的 Experience
   - 测试 memory_draft_publish 将 draft → published
   - 测试 memory_draft_save 的 source 不为 pipeline 时报错
   - 测试 memory_save（现有工具）不受影响
2. 修改 `server.py`
3. 修改 `services/memory_operations.py`
4. 验证测试通过

**关键代码**：

`server.py` — 新增两个工具：
```python
@mcp.tool(
    name="memory_draft_save",
    description=(
        "Pipeline-only: save a draft memory. source is always 'pipeline', "
        "exp_status is always 'draft'. Do NOT call this directly as an agent; "
        "this is for the memory pipeline hooks."
    ),
)
@track_usage
async def memory_draft_save(
    title: str,
    content: str,
    tags: list[str] | None = None,
    project: str | None = None,
    group_key: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """管线专用：保存草稿"""
    user = await _get_current_user()
    result = await memory_operations.op_draft_save(
        user, title=title, content=content, tags=tags,
        project=project, group_key=group_key,
        conversation_id=conversation_id,
    )
    return _guard_output(json.dumps(result, ensure_ascii=False))


@mcp.tool(
    name="memory_draft_publish",
    description=(
        "Pipeline-only: promote a draft to published. "
        "Only works on experiences with source='pipeline' and exp_status='draft'."
    ),
)
@track_usage
async def memory_draft_publish(
    draft_id: str,
    refined_content: str | None = None,
) -> str:
    """管线专用：草稿转正式"""
    user = await _get_current_user()
    result = await memory_operations.op_draft_publish(
        user, draft_id=draft_id, refined_content=refined_content,
    )
    return _guard_output(json.dumps(result, ensure_ascii=False))
```

`services/memory_operations.py` — 新增：
```python
async def op_draft_save(user, *, title, content, tags, project, group_key, conversation_id):
    """管线写入草稿"""
    # 强制 source=pipeline, exp_status=draft
    svc = _get_service()
    exp = await svc.save(
        title=title,
        description=content,
        tags=tags,
        project=project or "default",
        source="pipeline",
        exp_status="draft",
        group_key=group_key,
        created_by=user,
    )
    return {"id": str(exp.id), "status": "draft"}

async def op_draft_publish(user, *, draft_id, refined_content):
    """草稿转正式"""
    svc = _get_service()
    exp = await svc.update(
        experience_id=draft_id,
        exp_status="published",
        description=refined_content,  # 如果有精炼后的内容
    )
    return {"id": str(exp.id), "status": "published"}
```

**验收标准**：
- [ ] `memory_draft_save(title="test", content="test")` 创建 draft Experience，source=pipeline
- [ ] `memory_draft_publish(draft_id="xxx")` 将 draft → published
- [ ] 对非 pipeline source 的 Experience 调 draft_publish 报错
- [ ] 现有 memory_save/memory_recall 不受影响
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 1-7：检索结果注入 [mem:xxx] + SearchLog 写入

**描述**：把任务 1-2（SearchLog）和 1-4（评估服务）集成到检索管线中。

**TDD 流程**：
1. 写测试：`tests/test_search_with_evaluation.py`
   - 测试 memory_recall 返回的结果包含 [mem:xxx] 标记
   - 测试每次检索自动写入 SearchLog
   - 测试 SearchLog 的 result_ids 包含正确的 id 和 score
2. 修改 `services/search_orchestrator.py`
3. 修改 `server.py` 的 memory_recall 返回逻辑
4. 验证测试通过

**关键修改**：

`services/search_orchestrator.py` — search 方法末尾：
```python
# 现有：直接返回结果
# 修改后：

# 1. 注入 [mem:xxx] 标记
if self._evaluation_service:
    result.results = self._evaluation_service.inject_markers(result.results)

# 2. 写入 SearchLog
if self._search_log_repo:
    await self._search_log_repo.create(
        query=query,
        intent_type=intent.intent_type,
        project=project or "default",
        source=source,
        result_ids=[{"id": r.get("id"), "score": r.get("score")} for r in result.results],
    )
```

**验收标准**：
- [ ] memory_recall 返回的每条结果包含 [mem:exp-xxx] 标记
- [ ] search_logs 表有对应记录
- [ ] SearchLog 的 intent_type 等于 IntentRouter 的分类结果
- [ ] 现有测试不受影响
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 1-8：集成测试 + 全量回归

**描述**：确保所有新增功能集成后，现有功能不受影响。

**TDD 流程**：
1. 新增 `tests/test_phase1_integration.py`
   - 端到端测试：draft_save → draft_publish → recall → 检查标记
   - 端到端测试：创建 published Experience → use_count 增长 → Janitor promotion → promoted
   - 回归测试：确保 memory_save / memory_recall / memory_feedback 不受影响
2. 运行全量测试：`make verify`
3. 修复任何回归问题

**验收标准**：
- [ ] 端到端流程：draft → publish → recall → 评估标记 全部通过
- [ ] 端到端流程：published → use_count++ → promotion 全部通过
- [ ] `make verify` 全部通过（lint + 全量测试）
- [ ] 无回归问题

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

## 阶段 1 人工验收条目

完成所有子任务后，人工验收以下条目：

- [ ] 启动 TM 服务：`make web`，确认无报错
- [ ] 通过 MCP 调 memory_draft_save，确认创建 draft Experience
- [ ] 通过 MCP 调 memory_draft_publish，确认 draft → published
- [ ] 通过 MCP 调 memory_recall，确认返回结果带 [mem:xxx] 标记
- [ ] 手动把一条 Experience 的 use_count 改为 3，调 Janitor run，确认 promoted
- [ ] 检查 search_logs 表有记录
- [ ] 确认 TM Web UI 正常访问
