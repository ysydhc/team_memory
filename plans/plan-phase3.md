# 阶段 3：Obsidian + Git 索引

## 项目背景

L3 层是人类知识，存在 Obsidian 仓库的 Markdown 文件中。
这些文件需要被 TM 索引，才能在检索时被 Agent 找到。

索引策略（已确认）：
- Git staged（git add）→ 变更索引（中置信度）
- Git committed → 固定索引（高置信度）
- Git untracked → 忽略（半成品不索引）

第一版只做基础索引能力：
- git add → 建变更索引
- git commit → 升级为固定索引
- git rm → 移除索引

后续迭代再做 branch/tag/blame/diff/revert/.gitattributes 等深度用法。

## 预计改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/hooks/git_watcher.py` | 新建 | Git 事件监听器 |
| `scripts/hooks/markdown_indexer.py` | 新建 | Markdown → Experience 的转换 |
| `scripts/hooks/obsidian_config.yaml` | 新建 | Obsidian 仓库路径映射 |
| `scripts/hooks/common.py` | 修改 | 新增 Git 相关工具函数 |
| `services/obsidian_index.py` | 新建（TM 内） | Obsidian 索引服务 |
| `storage/models.py` | 修改 | Experience 新增 index_source 字段 |

## 架构图

### 新架构：Git 驱动的 L3 索引

```
┌──────────────────────────────────────────────────────────────┐
│  Obsidian 仓库（Git 管理）                                    │
│                                                              │
│  ad_learning/  (git repo)                                    │
│  ├── docs/2026/    → project=ad_learning, type=research      │
│  ├── ad_literacy/  → project=ad_learning, type=knowledge     │
│  └── technology/   → project=ad_learning, type=tech          │
│                                                              │
│  ai_learning/  (git repo)                                    │
│  ├── layer0/       → project=ai_learning, type=learning      │
│  └── llm-skill-map/ → project=ai_learning, type=app          │
│                                                              │
│  team_doc/   (git repo)                                      │
│  └── plans/        → project=team_doc, type=plan             │
└──────────────────────────────────────────────────────────────┘
         │
         │ git add / git commit 事件
         ↓
┌──────────────────────────────────────────────────────────────┐
│  git_watcher.py                                              │
│                                                              │
│  监听方式：git post-commit hook + inotifywait 备选            │
│                                                              │
│  事件：                                                       │
│    git add <file>  → markdown_indexer.index(file, staged)    │
│    git commit      → markdown_indexer.upgrade_to_committed() │
│    git rm <file>   → markdown_indexer.remove(file)           │
│    git mv <a> <b>  → markdown_indexer.move(a, b)             │
└──────────────────────────────────────────────────────────────┘
         │
         ↓
┌──────────────────────────────────────────────────────────────┐
│  markdown_indexer.py                                         │
│                                                              │
│  Markdown → Experience 映射：                                 │
│    title       ← 文件名（去掉 .md）或 frontmatter title       │
│    description ← 文件前 500 字（去掉 frontmatter）            │
│    solution    ← 文件全文                                     │
│    tags        ← frontmatter tags 或目录名推断                │
│    project     ← obsidian_config.yaml 中的映射                │
│    source      ← "obsidian"                                  │
│    exp_status  ← staged: "draft", committed: "published"     │
│    group_key   ← 文件路径的目录部分                            │
└──────────────────────────────────────────────────────────────┘
         │
         ↓
┌──────────────────────────────────────────────────────────────┐
│  TM Experience                                               │
│                                                              │
│  source=obsidian 的 Experience：                              │
│    · 由 Git 事件驱动创建/更新                                  │
│    · staged → exp_status=draft                               │
│    · committed → exp_status=published                        │
│    · 文件修改后 git commit → 更新 Experience 的 embedding     │
│    · 文件删除 → 软删除 Experience                             │
└──────────────────────────────────────────────────────────────┘
```

## 子任务拆分

---

### 任务 3-1：Obsidian 仓库配置 + Markdown 解析

**描述**：定义 Obsidian 仓库路径映射配置，实现 Markdown 文件到 Experience 数据的解析。

**TDD 流程**：
1. 写测试：`tests/test_hook_markdown_indexer.py`
   - 测试 Markdown 解析：提取 frontmatter（title/tags）、正文摘要
   - 测试路径映射：`/Work/ad_learning/docs/2026/xxx.md` → project=ad_learning
   - 测试空 Markdown、无 frontmatter 的 Markdown
   - 测试中文 Markdown
2. 新建 `scripts/hooks/obsidian_config.yaml`
3. 新建 `scripts/hooks/markdown_indexer.py`（解析部分）
4. 验证测试通过

**obsidian_config.yaml**：
```yaml
vaults:
  - path: /Users/yeshouyou/Work/ad_learning
    project: ad_learning
    index_patterns:
      - "docs/**/*.md"
      - "ad_literacy/**/*.md"
      - "technology/**/*.md"
    exclude_patterns:
      - ".obsidian/**"
      - ".git/**"
      
  - path: /Users/yeshouyou/Work/agent/ai_learning
    project: ai_learning
    index_patterns:
      - "**/*.md"
    exclude_patterns:
      - ".obsidian/**"
      - ".git/**"

  - path: /Users/yeshouyou/Work/agent/team_doc
    project: team_doc
    index_patterns:
      - "plans/**/*.md"
    exclude_patterns:
      - ".obsidian/**"
      - ".git/**"
```

**markdown_indexer.py 核心方法**：
```python
class MarkdownIndexer:
    def parse_file(self, file_path: str) -> dict:
        """解析 Markdown 文件为 Experience 数据"""
        # 1. 读取文件
        # 2. 解析 frontmatter（如果有）
        # 3. title = frontmatter.title 或文件名
        # 4. tags = frontmatter.tags 或目录名推断
        # 5. description = 正文前 500 字
        # 6. solution = 全文
        # 返回 {title, description, solution, tags, file_path}
    
    def resolve_project(self, file_path: str, config: dict) -> str:
        """从文件路径推断 project 名"""
    
    def should_index(self, file_path: str, config: dict) -> bool:
        """判断文件是否应该被索引"""
```

**验收标准**：
- [ ] 带 frontmatter 的 Markdown 正确提取 title 和 tags
- [ ] 不带 frontmatter 的 Markdown 用文件名做 title
- [ ] 中文 Markdown 正确解析
- [ ] 路径映射正确：`/Work/ad_learning/docs/xxx.md` → project=ad_learning
- [ ] exclude_patterns 中的文件不被索引
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 3-2：Git watcher 实现

**描述**：监听 Git 事件（add/commit/rm），触发索引操作。

**TDD 流程**：
1. 写测试：`tests/test_hook_git_watcher.py`
   - 测试 git add 后文件被标记为 staged 索引
   - 测试 git commit 后 staged 升级为 committed 索引
   - 测试 git rm 后索引被移除
   - 测试非 Markdown 文件被忽略
2. 实现 `scripts/hooks/git_watcher.py`
3. 为每个 Obsidian 仓库配置 git post-commit hook
4. 验证测试通过

**监听方式**：
```
方式 1（推荐）：Git post-commit hook
  每个 Obsidian 仓库的 .git/hooks/post-commit
  → 调 git_watcher.py --event=commit --repo=<path>

方式 2（补充）：定时扫描
  每分钟扫描 git status --short
  → 检测 staged 文件
```

**git_watcher.py 伪代码**：
```python
class GitWatcher:
    def __init__(self, config: dict, indexer: MarkdownIndexer, mcp_url: str):
        self._config = config
        self._indexer = indexer
        self._mcp_url = mcp_url
    
    async def on_git_add(self, repo_path: str, files: list[str]):
        """git add 后：为 staged 文件建变更索引"""
        for f in files:
            if self._indexer.should_index(f, self._config):
                data = self._indexer.parse_file(f)
                await call_mcp_tool("memory_draft_save", {
                    "title": data["title"],
                    "content": data["solution"],
                    "tags": data["tags"],
                    "project": data["project"],
                    "group_key": data.get("file_path"),
                })
    
    async def on_git_commit(self, repo_path: str):
        """git commit 后：staged 索引升级为固定索引"""
        # 找到 source=obsidian + exp_status=draft 的 Experience
        # 调 memory_draft_publish 升级
        pass
    
    async def on_git_rm(self, repo_path: str, files: list[str]):
        """git rm 后：移除对应索引"""
        # 找到 source=obsidian + group_key 匹配的 Experience
        # 软删除
        pass
```

**验收标准**：
- [ ] git add 一个 .md 文件后，TM 中出现 draft Experience（source=obsidian）
- [ ] git commit 后，draft → published
- [ ] git rm 后，对应 Experience 被软删除
- [ ] 非 .md 文件不触发索引
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 3-3：TM 新增 source=obsidian 支持

**描述**：TM 的 Experience 模型和 MCP 工具需要支持 source=obsidian 的特殊处理。

**TDD 流程**：
1. 写测试：`tests/test_obsidian_source.py`
   - 测试 source=obsidian 的 Experience 可以通过 MCP 创建
   - 测试 source=obsidian 的 draft 可以升级为 published
   - 测试 source=obsidian 的 Experience 在检索时正常返回
2. 修改 `server.py` 或 `services/memory_operations.py`
3. 验证测试通过

**关键**：
- source=obsidian 是 pipeline 的一种，同样受 draft→published 约束
- 但 source 字段保留 "obsidian" 以区分来源
- 需要微调任务 1-1 的约束：`source in ("pipeline", "obsidian")` 时才能设 draft

**验收标准**：
- [ ] source=obsidian 的 Experience 能正常创建和发布
- [ ] source=obsidian 的 draft 能升级为 published
- [ ] 检索时 source=obsidian 的 Experience 正常返回
- [ ] 所有测试通过

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 3-4：初始全量索引

**描述**：对已有的 Obsidian 仓库进行全量索引，把现有的 Markdown 文件导入 TM。

**TDD 流程**：
1. 写 `scripts/hooks/initial_index.py` — 全量扫描脚本
2. 测试：扫描 ad_learning 仓库，确认文件被正确索引
3. 对各仓库执行全量索引
4. 验证：在 TM 中能搜到已有的 Markdown 内容

**伪代码**：
```python
async def initial_index(config_path: str):
    """对配置中的所有 Obsidian 仓库执行全量索引"""
    config = load_config(config_path)
    indexer = MarkdownIndexer()
    
    for vault in config["vaults"]:
        repo_path = vault["path"]
        project = vault["project"]
        
        # 扫描所有符合 index_patterns 的 .md 文件
        for md_file in glob(os.path.join(repo_path, "**/*.md"), recursive=True):
            if not indexer.should_index(md_file, config):
                continue
            if is_untracked(md_file):  # 跳过 untracked
                continue
            
            data = indexer.parse_file(md_file)
            # 直接创建 published Experience（已有文件视为已确认）
            await call_mcp_tool("memory_save", {
                "title": data["title"],
                "content": data["solution"],
                "tags": data["tags"],
                "project": project,
                "source": "obsidian",
            })
```

**验收标准**：
- [ ] ad_learning 仓库的 .md 文件被导入 TM
- [ ] 搜索 "OMSDK" 或 "programmatic" 能命中 ad_learning 的内容
- [ ] ai_learning 仓库的 .md 文件被导入 TM
- [ ] 搜索 "LLM" 能命中 ai_learning 的内容

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

### 任务 3-5：端到端验证

**描述**：在 Obsidian 中编辑文件，通过 Git 提交，验证 TM 索引更新。

**验收标准**：
- [ ] 在 ad_learning 中新建一个 .md 文件，git add → TM 出现 draft
- [ ] git commit → TM 的 draft → published
- [ ] 修改已有 .md 文件，git commit → TM 的 Experience 内容更新
- [ ] 删除 .md 文件，git rm → TM 的 Experience 被软删除
- [ ] 在 Cursor 中说相关关键词，能检索到 Obsidian 的内容

**Subagent 执行方式**：delegate_task，toolsets=['terminal', 'file']

---

## 阶段 3 人工验收条目

- [ ] 执行全量索引脚本，确认 ad_learning 和 ai_learning 的内容进入 TM
- [ ] 在 Obsidian 中新建一个文件，git add + commit，确认 TM 有新记录
- [ ] 修改 Obsidian 中的文件，git commit，确认 TM 中对应记录已更新
- [ ] 在 Cursor 中问一个和 ad_learning 相关的问题，检查是否有 Obsidian 来源的检索结果
- [ ] 检查 TM Web UI 中 source=obsidian 的 Experience
