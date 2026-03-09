# CI/CD 流水线说明（P4-3）

> 运维文档 | GitHub Actions
> 相关：[quick-start 快速启动](quick-start.md)

## 一、触发条件

| 事件 | 分支 | 说明 |
|------|------|------|
| **push** | `main`、`develop` | 推送到这两类分支时触发 |
| **pull_request** | `main` | 仅针对目标分支为 `main` 的 PR 触发 |

配置文件：`.github/workflows/ci.yml`。

## 二、Job 一览

| Job | 说明 |
|-----|------|
| **Lint & Type Check** | `ruff check`、`ruff format --check`；Secrets check 敏感信息扫描 |
| **Tests** | PostgreSQL 服务容器、`alembic upgrade head`、`pytest`、Web smoke |
| **Docker Build** | 仅在 **push** 时运行，构建镜像并打 tag |

## 三、本地自检

推送前建议执行：

```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v
```

## 四、如何查看结果

打开仓库 **Actions** 页，选择最近一次 run，查看各 job 的日志与状态。
