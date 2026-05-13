# Implementation Plan: Interactive Architecture Diagram

**Status**: Ready
**Deliverable**: docs/architecture.html (single file, D3.js CDN)

## Architecture of the Diagram Itself

### Component Breakdown

| Component | Tech | Lines | Role |
|-----------|------|-------|------|
| CSS (light/dark) | CSS variables | ~400 | 明暗双主题, 三栏布局, 节点/连线/面板样式 |
| Left Tree | Vanilla JS | ~250 | 双模式 toggle (功能域/分层), 搜索过滤, 标签标注 |
| Center DAG | D3.js v7 | ~500 | SVG 画布, 5层固定布局, 节点拖拽, 4种流线, 渐进展开 |
| Right Panel | Vanilla JS | ~200 | 堆叠展示详情, [查看完整DDL] 展开, 双向联动 |
| Data Model | JS objects | ~500 | 模块元信息, DB DDL, CLI/Make命令, 依赖边 |
| Footer + Reset | Vanilla JS | ~100 | 状态栏, 重置视图按钮 |

### Data Flow (within the diagram)

```
User clicks tree leaf → dispatchEvent("select", moduleId)
  → DAG handler: pan + highlight node
  → Panel handler: render module details

User double-clicks DAG node → toggleExpand(nodeId)
  → DAG handler: progressive expand/collapse
  → (No tree/panel change needed)

User types in search → filterTree(query)
  → Tree: show matching nodes + ancestors
  → DAG: filter visible nodes
```

---

## Step 1: 搭骨架 — HTML 结构 + CSS 变量 (现状:无, 目标:可打开的空页面)

### 现状
- docs/ 目录不存在 architecture.html
- 没有 D3.js CDN 引用

### 目标
- 三栏布局框架: 顶部导航 + 左栏 + 中栏 + 右栏 + 底部
- 明/暗 CSS 变量: --bg, --text, --card-bg, --border, --accent
- @media (prefers-color-scheme: dark) 自动切换
- D3.js CDN 已加载

### 具体改动
**文件**: docs/architecture.html (新建)
- HTML 骨架: header(标题+搜索框+主题切换) + main(左侧280px+中间flex+右侧320px) + footer
- CSS: CSS variables 两套, 三栏 flex 布局, 滚动条样式
- JS: 主题切换按钮 `toggleTheme()`

### 验收
- 浏览器打开 docs/architecture.html → 显示空白三栏布局
- 系统切换到暗色模式 → 页面跟随变暗
- 手动点切换按钮 → 亮/暗切换

---

## Step 2: 数据模型 — JS 对象定义全部模块信息

### 现状
- 模块元信息散落在源码中, 没有集中定义文件

### 目标
- 一个 JS 对象 `MODULES` 包含全部模块的完整元数据
- 一个 JS 对象 `EDGES` 包含全部数据流边

### 具体改动
**文件**: docs/architecture.html (js 段)

MODULES 结构:
```javascript
const MODULES = {
  "mcp-server": {
    id: "mcp-server", name: "server.py", path: "src/team_memory/server.py",
    domain: "mcp-tools", layer: "entry", level: "L3",
    description: "MCP Server 入口, 注册 6 个 memory_* 工具...",
    inputs: [
      {name: "problem", type: "str", desc: "用户问题/查询语句"},
      ...
    ],
    outputs: [
      {name: "result", type: "str", desc: "JSON 格式搜索结果"},
    ],
    db_tables: ["experiences", "search_logs"],
    cli_make: ["make mcp", "tm-cli recall --query"],
    dependencies: {in: ["mcp-auth"], out: ["search-orchestrator", "experience-service"]},
  },
  // ... 50+ 模块
};
```

EDGES 结构:
```javascript
const EDGES = [
  {from: "mcp-server", to: "search-orchestrator", type: "call"},
  {from: "hook-cursor", to: "daemon-pipeline", type: "async"},
  {from: "search-pipeline", to: "postgres", type: "db"},
  // ...
];
```

8 个功能域, 每个域下 L2/L3 模块:
- MCP 工具服务: 3 modules
- 搜索管线: 6 modules (含 reranker/5)
- 经验管理: 7 modules
- 后台 Daemon: 18 modules
- Hook 集成: 12 modules (3平台×4事件)
- 归档系统: 4 modules
- 认证鉴权: 6 modules
- 基础设施: 12 modules

### 依赖: Step 1
### 验收: 打开控制台 → `MODULES["mcp-server"].inputs[0].desc` 返回 "用户问题/查询语句"

---

## Step 3: 左侧树组件

### 现状
- 有 DOM 但没有树组件

### 目标
- 双模式 toggle (功能域/分层) — 默认功能域
- 每组可折叠/展开
- 叶子项右侧标签 ([MCP] [CLI] [RRF]) 
- 搜索实时过滤
- 点击 → dispatchEvent("select")

### 具体改动
**文件**: docs/architecture.html (js 段)

```javascript
function buildTree(mode = "domain") { /* 遍历 MODULES, 按 mode 分组 */ }
function filterTree(query) { /* 隐藏不匹配节点, 显示祖先链 */ }
function toggleGroup(groupId) { /* 展开/折叠组 */ }
```

交互:
- 点击组名 → toggle + DAG 定位 L1 节点
- 点击叶子 → DAG 定位 L3 + 右侧面板渲染
- Ctrl/Cmd+点击 → 多选

### 依赖: Step 2
### 验收: 左侧显示 8 组 (功能域模式), 可展开/折叠, 搜索过滤

---

## Step 4: 中间 DAG 图 (D3.js)

### 现状
- 有 SVG 容器但无内容

### 目标
- D3.js 绘制 5 层 DAG 节点 + 4 种流线
- 默认展开到 L2 层
- 缩放 (0.3x~2x) + 平移 + 拖拽节点
- 渐进展开: 双击 L1/L2 → 节点变分组框, 子节点在框内出现

### 具体改动
**文件**: docs/architecture.html (js 段)

```javascript
function initDAG() {
  svg = d3.select("#dag-canvas").append("svg")...
  // Layer positions: fixed Y per layer, spread X within layer
  // Draw nodes: append("rect") + append("text") per MODULES entry
  // Draw edges: append("path") with marker-end arrows per EDGES
  // Zoom: d3.zoom().scaleExtent([0.3, 2])
  // Drag: d3.drag() on nodes
  // Legend: 4 line types in bottom-left corner
}
function expandNode(nodeId) {
  // L1 → show L2 children inside dashed group box
  // L2 → show L3 children inside solid group box
}
```

5 层固定 Y 坐标:
- Layer 1 (entry): y=80
- Layer 2 (services): y=220
- Layer 3 (daemon): y=380
- Layer 4 (data): y=520
- Layer 5 (external): y=660

节点按功能域分组, 水平均匀分布

### 依赖: Step 2
### 验收: 中间显示 20-30 个节点 (L1+L2), 带箭头连线, 可拖拽/缩放/双击展开

---

## Step 5: 右侧详情面板

### 现状
- 有空 div 但无内容

### 目标
- 堆叠显示: 模块作用 + 输入参数(含含义) + 输出内容 + 数据库表 + CLI/Make + 上下游依赖
- [查看完整 DDL] 可展开/折叠
- L1/L2 节点只显示模块作用 + 子模块数量

### 具体改动
**文件**: docs/architecture.html (js 段)

```javascript
function renderDetail(moduleId) {
  const m = MODULES[moduleId];
  panel.innerHTML = `
    <h3>${m.name}</h3>
    <p class="path">${m.path}</p>
    ${renderDescription(m)}
    ${renderInputs(m)}
    ${renderOutputs(m)}
    ${renderDBTables(m)}
    ${renderCLI(m)}
    ${renderDependencies(m)}
  `;
}
```

每个 render 函数根据模块类型 (L1/L2/L3) 决定显示内容:
- L1/L2: 只显示 description + children count + dependencies
- L3: 全部显示

### 依赖: Step 2
### 验收: 点击树叶子 → 右侧显示该模块完整信息, 参数含含义说明

---

## Step 6: 联动 + 搜索 + 导出

### 现状
- 三个组件独立, 无联动

### 目标
- 树 ↔ DAG ↔ 面板 双向联动
- 搜索框实时过滤树 + DAG
- 底部状态栏: 模块数 | 展开层 | 重置 | 导出 PNG

### 具体改动
**文件**: docs/architecture.html (js 段)

```javascript
// Event bus
window.addEventListener("select", (e) => {
  highlightDAG(e.detail.id);
  renderDetail(e.detail.id);
});

// Search
searchInput.addEventListener("input", (e) => {
  filterTree(e.target.value);
  filterDAG(e.target.value);
});

// Reset
resetBtn.addEventListener("click", resetToDefault);

// Export (optional — low priority)
```

### 依赖: Step 3, 4, 5
### 验收: 点击树 → DAG 高亮 + 右侧更新; 搜索 "pipeline" → 树和 DAG 都只显示匹配项

---

## Step 7: 数据填充 — 逐个模块录入真实信息 (手动)

### 现状
- MODULES 和 EDGES 有骨架但缺真实数据

### 目标
- 50+ 模块的 description, inputs, outputs, db_tables, cli_make, dependencies 全部填写真实值
- 从源码提取: server.py 签名, models.py DDL, Makefile 命令

### 具体改动
**文件**: docs/architecture.html (js 段)
- 逐个模块填写 (工作量最大的一步)
- 优先 8 个 L1 域 → 再 L2 核心模块 → 最后 L3 叶子

### 依赖: Step 2
### 验收: 每个叶子模块点击后右侧显示完整、准确的信息
