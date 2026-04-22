# Memory Janitor 实施计划

## 目标

为 TeamMemory 增加定时轮询记忆清理能力（Memory Janitor），自动识别和清理无用、过时、废弃的记忆内容，保持经验库健康。

## 当前状态

- Experience 模型有 `exp_status`(draft/published)、`is_deleted`(软删除)、`use_count`(引用计数)、`created_at/updated_at`
- README 提到了质量打分系统（阶梯衰减、引用加分、Outdated 等），但**代码中未实现**
- 已有 BackgroundTask 模型 + TaskRunner 后台任务框架
- 已有 EventBus 事件驱动
- 个人记忆 PersonalMemory 无过期机制
- 无定时调度器

## 实施步骤

### Step 1: Experience 模型增加质量评分字段

**文件**: `src/team_memory/storage/models.py`

在 Experience 类上新增字段：
```python
quality_score: Mapped[float] = mapped_column(Float, default=100.0, nullable=False, server_default="100.0")
quality_tier: Mapped[str] = mapped_column(String(20), default="Silver", nullable=False, server_default="Silver")
last_scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
is_pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, server_default="false")
```

更新 `to_dict()` 方法输出这些字段。

**数据库迁移**: 创建 Alembic migration 添加这 4 列。

### Step 2: Repository 层增加 Janitor 所需查询

**文件**: `src/team_memory/storage/repository.py`

新增方法：
- `list_for_score_decay(project, batch_size)` -- 查询 last_scored_at 超过 24h 的 published 经验（排除 is_pinned）
- `batch_update_scores(updates: list[tuple[uuid.UUID, float, str, datetime]])` -- 批量更新 quality_score/quality_tier/last_scored_at
- `list_outdated(project)` -- 查询 quality_tier == "Outdated" 且未被软删除的经验
- `list_soft_deleted_older_than(days)` -- 查询软删除超过 N 天的经验
- `list_expired_drafts(older_than_days, project)` -- 查询超过 N 天仍为 draft 的经验
- `increment_quality_score(experience_id, delta)` -- 引用/反馈时加分

### Step 3: Janitor 服务层

**新建文件**: `src/team_memory/services/janitor.py`

```python
class MemoryJanitor:
    """记忆清理服务 -- 定期评估和清理过期/低价值记忆"""
    
    def __init__(self, db_url, config, embedding_provider=None):
        ...
    
    async def run_score_decay(self, project=None) -> dict:
        """对所有需要衰减的经验计算新分数"""
        # 规则：
        # - is_pinned=True 的跳过
        # - 保护期(10天)内的跳过
        # - 保护期后每天 -1
        # - 低于 50 后每天 -0.5
        # - 更新 quality_tier: Gold(≥120) / Silver(≥60) / Bronze(≥20) / Outdated(≤0)
    
    async def sweep_outdated(self, project=None, auto_soft_delete=False) -> dict:
        """处理 Outdated 经验：标记待处理 或 自动软删除"""
    
    async def purge_soft_deleted(self, older_than_days=30) -> dict:
        """硬删除软删除超过 N 天的经验"""
    
    async def expire_drafts(self, older_than_days=30, project=None) -> dict:
        """将超过 N 天仍为 draft 的经验标记为过期（软删除）"""
    
    async def prune_personal_memory(self, older_than_days=90) -> dict:
        """清理长期未更新的 dynamic 个人记忆"""
    
    async def run_all(self, project=None) -> dict:
        """依次执行所有清理任务，返回汇总报告"""
```

### Step 4: 搜索命中时加分

**文件**: `src/team_memory/services/search_pipeline.py` 或 `memory_operations.py`

在 `memory_recall` 命中结果时，对被命中的经验调用 `increment_quality_score(experience_id, +2)`。
在 `memory_feedback` 4 星以上评价时，加 +1 分。

### Step 5: Janitor 调度器

**新建文件**: `src/team_memory/services/janitor_scheduler.py`

```python
class JanitorScheduler:
    """在 Web 进程中运行的后台定时调度器"""
    
    def __init__(self, janitor: MemoryJanitor, config):
        self._janitor = janitor
        self._config = config
        self._task: asyncio.Task | None = None
    
    async def start(self):
        """启动定时循环"""
    
    async def stop(self):
        """优雅停止"""
    
    async def _run_loop(self):
        """主循环：按配置的间隔执行各项清理任务"""
        # score_decay: 每 24h
        # sweep_outdated: 每 7 天
        # purge_soft_deleted: 每 30 天
        # expire_drafts: 每 7 天
        # prune_personal_memory: 每 7 天
```

### Step 6: 配置支持

**文件**: `src/team_memory/config/` 相关

新增 `janitor` 配置段：
```yaml
janitor:
  enabled: true
  score_decay_interval_hours: 24
  initial_score: 100.0
  protection_days: 10
  decay_per_day: 1.0
  decay_per_day_low: 0.5
  decay_low_threshold: 50.0
  outdated_threshold: 0.0
  auto_soft_delete_outdated: false
  soft_delete_purge_after_days: 30
  draft_expiry_days: 30
  personal_memory_dynamic_max_age_days: 90
  tier_thresholds:
    gold: 120.0
    silver: 60.0
    bronze: 20.0
  recall_score_bonus: 2.0
  feedback_high_score_bonus: 1.0
```

### Step 7: 集成到 Bootstrap

**文件**: `src/team_memory/bootstrap.py`

- 在 `bootstrap()` 中创建 MemoryJanitor 和 JanitorScheduler
- 当 `enable_background=True` 且 `janitor.enabled=True` 时启动调度器
- AppContext 增加 `janitor` 和 `janitor_scheduler` 字段

### Step 8: Web API

**文件**: `src/team_memory/web/routes/` 新增或扩展现有路由

- `POST /api/v1/janitor/run` -- 手动触发清理（admin only）
- `GET /api/v1/janitor/status` -- 查看清理状态和上次执行结果
- `GET /api/v1/experiences/outdated` -- 列出 Outdated 经验

### Step 9: 测试

**文件**: `tests/test_janitor.py`

覆盖：
- 质量评分衰减计算（保护期、阶梯衰减、置顶免衰）
- Outdated 清理
- 软删除硬删除
- Draft 过期
- 个人记忆清理
- 调度器启停
- 搜索命中加分

## 关键决策

1. **Outdated 处理方式**：默认仅标记，不自动删除（`auto_soft_delete_outdated: false`），Web 管理面板提示人工处理
2. **纯规则衰减**：不依赖 LLM，简单可靠，后续可扩展
3. **调度方式**：Web 进程内 asyncio 定时循环（与现有架构一致），同时提供 CLI 和 API 手动触发入口
4. **quality_score 字段**：直接加到 Experience 模型，查询简单，避免 JOIN

## 文件变更清单

| 操作 | 文件 |
|------|------|
| 修改 | `src/team_memory/storage/models.py` |
| 新建 | `alembic/versions/xxxx_add_quality_score_fields.py` |
| 修改 | `src/team_memory/storage/repository.py` |
| 新建 | `src/team_memory/services/janitor.py` |
| 新建 | `src/team_memory/services/janitor_scheduler.py` |
| 修改 | `src/team_memory/config/` (新增 janitor 配置段) |
| 修改 | `src/team_memory/bootstrap.py` |
| 修改 | `src/team_memory/services/search_pipeline.py` (recall 加分) |
| 修改 | `src/team_memory/services/experience.py` (feedback 加分) |
| 新建 | `src/team_memory/web/routes/janitor.py` |
| 新建 | `tests/test_janitor.py` |

## 验证

```bash
make lint          # 零报错
make test          # 全绿
alembic upgrade head  # 迁移成功
```
