# Janitor API 端点

本文档描述了团队记忆清理服务（Janitor）的API端点。

## 概述

Janitor服务提供自动化的内存清理和质量管理功能，包括：
- 质量分数衰减
- 过期经验清理  
- 软删除记录清除
- 草稿过期处理
- 个人记忆修剪

## API 端点

### 1. 手动触发清理任务

**POST** `/api/v1/janitor/run`

手动触发完整的清理任务。需要admin权限。

#### 请求参数
- `project` (可选): 限制清理范围的项目名

#### 响应示例
```json
{
  "message": "Janitor cleanup completed successfully",
  "project": "my-project",
  "results": {
    "score_decay": {
      "processed": 150,
      "updated": 45
    },
    "outdated_sweep": {
      "found": 12,
      "auto_deleted": 3
    },
    "soft_deleted_purge": {
      "purged": 5
    },
    "draft_expiry": {
      "expired": 8
    },
    "personal_memory_prune": {
      "pruned": 25
    }
  }
}
```

#### 错误响应
- `503`: Janitor服务不可用
- `500`: 清理任务执行失败
- `403`: 权限不足（需要admin角色）

### 2. 查看清理状态

**GET** `/api/v1/janitor/status`

查看Janitor服务和调度器的运行状态。需要admin权限。

#### 响应示例
```json
{
  "janitor_available": true,
  "scheduler_available": true,
  "scheduler_running": true,
  "last_run": null,
  "config": {
    "interval_hours": 6,
    "enabled": true
  },
  "janitor_config": {
    "protection_period_days": 10,
    "auto_soft_delete_outdated": false,
    "purge_soft_deleted_days": 30,
    "draft_expiry_days": 7,
    "personal_memory_retention_days": 90
  }
}
```

#### 字段说明
- `janitor_available`: Janitor服务是否可用
- `scheduler_available`: 调度器是否可用
- `scheduler_running`: 调度器是否正在运行
- `config`: 调度器配置信息
- `janitor_config`: Janitor清理规则配置

### 3. 列出过期经验

**GET** `/api/v1/experiences/outdated`

列出质量等级为"Outdated"的经验记录。需要admin权限。

#### 请求参数
- `limit` (可选, 默认20): 返回条数限制 (1-100)
- `offset` (可选, 默认0): 偏移量
- `project` (可选): 项目过滤

#### 响应示例
```json
{
  "items": [
    {
      "id": "uuid-here",
      "title": "过期的经验",
      "problem": "问题描述",
      "solution": "解决方案",
      "quality_score": 45.5,
      "quality_tier": "Outdated",
      "cleanup_info": {
        "quality_score": 45.5,
        "quality_tier": "Outdated",
        "last_scored_at": "2024-01-15T10:30:00Z",
        "is_pinned": false
      }
    }
  ],
  "total": 25,
  "limit": 20,
  "offset": 0,
  "project": "my-project",
  "message": "Found 25 outdated experiences in project 'my-project'"
}
```

## 权限要求

所有Janitor API端点都需要admin权限：
- 用户必须具有`admin`角色
- 通过API Key认证或会话认证

## 使用示例

### 手动清理特定项目
```bash
curl -X POST "http://localhost:9111/api/v1/janitor/run?project=my-project" \
  -H "Authorization: Bearer your-api-key"
```

### 查看服务状态
```bash
curl "http://localhost:9111/api/v1/janitor/status" \
  -H "Authorization: Bearer your-api-key"
```

### 列出过期经验
```bash
curl "http://localhost:9111/api/v1/experiences/outdated?limit=10&project=my-project" \
  -H "Authorization: Bearer your-api-key"
```

## 注意事项

1. **权限控制**: 所有端点都需要admin权限，确保只有管理员可以执行清理操作
2. **服务依赖**: 清理功能依赖于Janitor服务的正确初始化
3. **性能影响**: 手动清理可能会影响数据库性能，建议在低峰时段执行
4. **数据安全**: 清理操作可能会永久删除数据，请谨慎使用

## 相关文档

- [Janitor服务配置](../config/janitor.md)
- [质量管理系统](../design/quality-management.md)
- [API认证](../auth/api-keys.md)