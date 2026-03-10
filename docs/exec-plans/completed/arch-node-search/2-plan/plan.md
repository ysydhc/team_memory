# 架构节点搜索功能 — 实施计划

> 来源：`/Users/yeshouyou/.cursor/plans/架构节点搜索功能_8f85648c.plan.md`

## 一、目标

在架构页「图」Tab 增加搜索按钮，支持按文件路径/函数名/类名子串匹配代码节点，结果以列表展示并可在图中高亮聚焦，支持全局/当前集群范围切换。

## 二、任务列表（按依赖排序）

| 序号 | 任务 | 产出 | 依赖 |
|------|------|------|------|
| 1 | Bridge /search | server.js 新增 GET /search，Cypher 查询，q 白名单校验，返回 nodes | — |
| 2 | Provider + 路由 | base.py 增加 search_nodes；gitnexus_provider 实现；architecture.py 注册 /search，q 校验 | 1 |
| 3 | 前端搜索 UI | index.html 搜索框+按钮+范围选择；architecture-viewer.js 搜索、结果列表、图中高亮、状态反馈 | 2 |
| 4 | 测试 | test_web.py 新增 test_search_* 4 个用例 | 2 |

## 三、验收标准

- 搜索框与刷新按钮并排显示
- 输入子串（如 `auth`）可返回匹配节点列表，空结果显示「未找到匹配节点」
- 点击列表项 → 图中对应节点高亮并居中
- 范围切换 → 结果列表与当前 scope 一致；scope=cluster 未选时「当前集群」禁用
- 无 bridge 时返回 503，前端不报错并提示「架构服务未配置」

## 四、测试用例（Task 4）

- test_search_requires_q：缺 q 返回 400
- test_search_returns_empty_when_no_provider：无 Provider 返回 503
- test_search_returns_nodes_when_bridge_configured：mock Bridge 返回 nodes
- test_search_rejects_invalid_q：含非法字符返回 400
