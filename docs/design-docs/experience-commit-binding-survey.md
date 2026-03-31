# 经验绑定到代码位置：Commit/版本级方案产品调研

> 调研目标：了解哪些产品使用「Git commit 或代码版本」做记忆/经验/上下文关联，以及 commit 被 amend/rebase/force-push 后如何保证绑定有效或降级。  
> 输出：产品名、方案简述、有效性策略、来源；未找到可靠来源处已标明「未找到」。

---

## 一、产品调研汇总

### 1. GitHub Copilot（Agentic Memory）

| 项 | 内容 |
|----|------|
| **方案简述** | 记忆与**代码位置引用（citations）**关联，citation 为**文件路径 + 行号**（如 `src/client/sdk/constants.ts:12`），**未使用 commit SHA**。记忆按仓库维度存储，在 Coding Agent、Code Review、CLI 间共享。 |
| **有效性策略** | **Just-in-time 校验**：使用记忆前会「检查 citations 与当前代码库是否一致」并确认与当前分支相关，仅校验通过才使用；**28 天自动过期**；若记忆被使用且校验通过，会重新存储以刷新时间戳。未合并/已关闭的 PR 产生的记忆不会影响当前分支行为（因校验时在当前代码库中找不到对应证据）。 |
| **来源** | [About agentic memory for GitHub Copilot](https://docs.github.com/en/copilot/concepts/agents/copilot-memory)；[Building an agentic memory system for GitHub Copilot - The GitHub Blog](https://github.blog/ai-and-ml/github-copilot/building-an-agentic-memory-system-for-github-copilot/) |

**原文摘录（有效性）：**

- *"When Copilot finds a memory that relates to the work it is doing, it checks the citations against the current codebase to validate that the information is still accurate and is relevant to the current branch. The memory is only used if it is successfully validated."*
- *"If the code contradicts the memory, or if the citations are invalid (e.g. point to nonexistent locations), the agent is encouraged to store a corrected version of the memory."*
- *"Memories are automatically deleted after 28 days."*

---

### 2. Git AI（git-ai-project/git-ai）

| 项 | 内容 |
|----|------|
| **方案简述** | 开源 Git 扩展，用 **commit SHA + 行范围** 做 AI 代码归属：Authorship Log 存于 **Git Notes（refs/notes/ai）**，按 commit 挂载，记录每段行范围对应的 agent/session/prompt；`git-ai blame` 与标准 blame 一致并叠加 AI 归属，支持 JSON 输出（含 commit、line、prompt id）。 |
| **有效性策略** | **历史重写时重写 Notes**：在 rebase、merge、squash、stash/pop、cherry-pick、amend 后，Git AI **透明地重写 Authorship Log**，使归属始终跟随「当前历史中的 commit」，故 attribution 不丢失。依赖 Git 的 notes 重写机制（如 `notes.rewriteRef`）或自有逻辑。 |
| **来源** | [Git AI README](https://github.com/git-ai-project/git-ai)（GitHub 仓库）；[AI Blame | Git AI](https://usegitai.com/docs/cli/ai-blame)；Web 搜索摘要中引用的「AI attribution automatically survives rebases, merges...」表述与 README 一致。 |

**原文摘录（有效性）：**

- *"Git AI preserves attribution across rebases, merges, squashes, stash/pops, cherry-picks, and amends by transparently rewriting Authorship Logs whenever history changes."*

---

### 3. Cursor（Codebase Indexing / Memories）

| 项 | 内容 |
|----|------|
| **方案简述** | **Codebase Index**：按文件/目录做 Merkle 树检测变更，对变更文件做语义分块与向量索引，约每 5 分钟同步；**Memories**：工作区级、本地存储的持久上下文。**未找到** 将记忆/经验绑定到「某次 Git commit」或「commit SHA」的官方说明。 |
| **有效性策略** | 索引与当前工作区文件状态同步，文件变更后只更新对应 chunk；记忆为工作区级，**未找到** 对 commit 失效、amend/rebase 后降级或 fallback 的文档。 |
| **来源** | [Codebase indexing \| Cursor Docs](https://cursor.com/docs/context/codebase-indexing)；[Memories \| Cursor Docs](https://docs.cursor.com/context/memories)；[Securely indexing large codebases · Cursor](https://cursor.com/blog/secure-codebase-indexing)。commit 级绑定：**未找到**。 |

---

### 4. Sourcegraph Cody

| 项 | 内容 |
|----|------|
| **方案简述** | Context 来自 Code Graph、Sourcegraph Search、关键词搜索；支持 @ 引用文件、符号、仓库等。与代码宿主（GitHub/GitLab）集成，可多仓库并行取上下文。**未找到** 将上下文或「经验」绑定到具体 Git commit/SHA 的文档。 |
| **有效性策略** | **未找到** 针对 commit 被改写/删除后的校验、过期或 fallback 策略的说明。 |
| **来源** | [Cody Context - Sourcegraph docs](https://sourcegraph.com/docs/cody/core-concepts/context)；[Manage Cody Context - Sourcegraph docs](https://sourcegraph.com/docs/cody/capabilities/ignore-context)。commit 级绑定与有效性：**未找到**。 |

---

### 5. GitLab Semantic Code Search（Duo / Active Context）

| 项 | 内容 |
|----|------|
| **方案简述** | 语义代码检索：按项目索引代码，分块后生成 embedding 存入向量库；**增量索引**在「代码合并到默认分支」时触发，只处理变更文件并删除孤儿数据。文档中有按 `project_id`、`path` 过滤，**未找到** 明确将 embedding/引用与「commit SHA」或「revision」绑定的字段说明。 |
| **有效性策略** | 通过「合并到默认分支」触发增量更新，索引与当前默认分支状态一致；删除/失效时通过 `MarkRepositoryAsPendingDeletionEventWorker` 等清理。**未找到** 对「某 commit 被 force-push 或删除」后对已有绑定做 just-in-time 校验或 fallback 的文档。 |
| **来源** | [Semantic Search \| GitLab Docs](https://docs.gitlab.com/development/ai_features/semantic_search/)（架构、索引与删除流程）。commit 级绑定：**未找到**。 |

---

### 6. Recall（stoodiohq/recall）

| 项 | 内容 |
|----|------|
| **方案简述** | 将 AI 会话上下文写入仓库内 **`.recall/` 目录的 markdown 文件**，随 clone 携带，团队共享；绑定维度是 **仓库 + 文件路径（及目录结构）**，**未使用 commit SHA**。 |
| **有效性策略** | 与 Git 历史无关：内容随当前文件树存在，不依赖某次 commit；若某文件被删或移动，需依赖产品自身是否做路径重定向或检索降级。**未找到** 官方对「文件移动/删除后」的 fallback 说明。 |
| **来源** | [GitHub - stoodiohq/recall](https://github.com/stoodiohq/recall)（README 描述 .recall/ 与 repo 同存）。commit 级：**未使用**。 |

---

### 7. Codeium / Windsurf（Remote Indexing）

| 项 | 内容 |
|----|------|
| **方案简述** | Windsurf 提供 Remote Indexing：按**分支**从 GitHub/GitLab/Bitbucket 拉取并建索引，可配置定期重新索引；记忆为工作区级、本地。**未找到** 将经验/记忆绑定到「某 commit SHA」的文档。 |
| **有效性策略** | 按分支与定时 re-index 更新索引；**未找到** 对 commit 失效、amend/rebase/force-push 后的专门策略或 fallback。 |
| **来源** | [Remote Indexing - Codeium/Windsurf Docs](https://docs.codeium.com/context-awareness/remote-indexing)；[Windsurf Memories](https://docs.windsurf.com/windsurf/cascade/memories)。commit 级绑定：**未找到**。 |

---

### 8. Recallium / OMEGA / Codemem 等（通用 AI 记忆层）

| 项 | 内容 |
|----|------|
| **方案简述** | 多为 MCP 兼容的跨工具记忆：Recallium 本地/自托管、多项目；OMEGA 语义搜索 + 会话决策持久化；Codemem 图+向量、记录文件与编辑。**未找到** 将记忆与「Git commit SHA」或「代码版本」绑定的设计描述。 |
| **有效性策略** | **未找到** 针对 commit 改写/删除的校验或降级策略。 |
| **来源** | 各项目 GitHub README；commit/版本级绑定：**未找到**。 |

---

## 二、Commit 有效性：行业常见做法（归纳）

| 策略 | 代表产品 | 说明 |
|------|----------|------|
| **不绑 commit，绑「文件路径+行号」+ 使用时再校验** | GitHub Copilot Memory | 存 citation 为 path:line；每次使用前对当前代码库做 just-in-time 校验，无效则不用或存修正版；辅以 28 天过期，避免废弃分支/未合并 PR 的记忆影响当前分支。 |
| **绑 commit + 历史重写时重写绑定数据** | Git AI | 用 Git Notes 挂到 commit；在 rebase/amend/squash 等历史重写时重写 Notes，使 attribution 始终对应「当前历史中的 commit」。依赖 Git 的 `notes.rewriteRef` 或自实现重写逻辑。 |
| **绑「分支/当前状态」不绑 commit** | Cursor、GitLab Semantic、Windsurf | 索引或记忆与当前工作区/默认分支状态一致，定时或事件驱动更新；不依赖某次 commit 长期有效。 |
| **仅仓库/路径级，与版本无关** | Recall | 经验存于 `.recall/` 随仓库存在，不涉及 commit；自然无「commit 失效」问题，但也无法按版本精确关联。 |

**关于「commit 被 amend/rebase/force-push/删除」：**

- **Copilot**：不存 commit，只存 path:line，通过「当前代码库校验」自然避免失效 commit 的问题；若位置被删或改，citation 校验失败即不用该记忆。
- **Git AI**：绑定在 commit 上，通过**重写 Notes** 在历史重写后仍对应新 commit，因此「有效性」由「是否执行了 notes 重写」保证；若 force-push 导致旧 commit 在远端消失，本地若已重写则本地仍一致，**未找到** 官方对「远端已无该 commit」时的降级说明。
- **其他产品**：**未找到** 明确针对「commit 失效」的 just-in-time 校验、过期策略或 fallback 到文件级的文档。

---

## 三、结论（3～5 条）

1. **明确使用 commit/版本级关联的产品**：目前调研中仅 **Git AI** 明确用 **commit SHA + 行范围** 做绑定（Authorship Log 存于 Git Notes），用于 AI 代码归属与 blame，而非「经验库」语义；**GitHub Copilot Memory** 使用 **文件路径 + 行号** 作为 citation，**未使用 commit SHA**。

2. **有效性保证的两种典型做法**：  
   - **Just-in-time 校验 + 过期**（Copilot）：不依赖 commit，存 path:line；使用时校验当前代码库与当前分支，无效则不用或更新记忆，并配合 28 天自动过期。  
   - **历史重写时重写绑定**（Git AI）：绑定在 commit 上，在 rebase/amend/squash 等操作后透明重写 Authorship Log，使绑定始终对应新历史中的 commit。

3. **可借鉴点**：  
   - 若采用「文件路径 + commit SHA」绑定：可借鉴 **Copilot** 的 **使用时再校验**：检索到经验后，用当前工作区（或当前分支）的 blame/log 解析出「当前行对应的 commit」，若与存储的 SHA 一致再优先采用，否则降级为「仅按路径/语义」或标记过期。  
   - 若希望「commit 被改写后仍可追踪」：可借鉴 **Git AI** 的 **Notes 随历史重写**：用 Git Notes 或等价存储挂到 commit，并在本地 rebase/amend 时通过 hook 或工具重写绑定（需考虑 force-push 后远端无旧 commit 的可见性策略）。

4. **多数 AI 编程/代码记忆产品**（Cursor、Sourcegraph Cody、GitLab Duo、Windsurf、Recall 等）**未在公开文档中** 将记忆/经验与「Git commit SHA」绑定，多采用**仓库/分支/路径/工作区**维度或**仅路径+行号**；commit 失效后的 **just-in-time 校验、过期策略、fallback 到文件级** 在公开文档中**未找到**除 Copilot 外的系统描述。

5. **若产品采用「文件路径 + Git commit SHA」绑定与检索**：建议至少考虑（1）**检索时用当前 blame 解析出的 commit 与存储 SHA 比对**，不一致时降级或标注；（2）**可选过期或版本窗口**（如仅信任 N 天内或某分支上的 commit）；（3）**fallback**：SHA 不可用时仅按文件路径或语义检索，并可在 UI/API 中标注「未匹配到精确 commit」。

---

## 四、来源一览

| 产品/主题 | 类型 | 链接 |
|-----------|------|------|
| GitHub Copilot Memory | 官方文档 | https://docs.github.com/en/copilot/concepts/agents/copilot-memory |
| GitHub Copilot Memory 设计博客 | 官方博客 | https://github.blog/ai-and-ml/github-copilot/building-an-agentic-memory-system-for-github-copilot/ |
| Git AI | 开源仓库 + 文档 | https://github.com/git-ai-project/git-ai ；https://usegitai.com/docs/cli/ai-blame |
| Cursor Codebase Indexing | 官方文档 | https://cursor.com/docs/context/codebase-indexing |
| Cursor Memories | 官方文档 | https://docs.cursor.com/context/memories |
| Sourcegraph Cody Context | 官方文档 | https://sourcegraph.com/docs/cody/core-concepts/context |
| GitLab Semantic Code Search | 官方文档 | https://docs.gitlab.com/development/ai_features/semantic_search/ |
| Recall (stoodiohq) | GitHub README | https://github.com/stoodiohq/recall |
| Windsurf/Codeium Remote Indexing | 官方文档 | https://docs.codeium.com/context-awareness/remote-indexing |
| Git notes 与 rebase/rewrite | Stack Overflow / Git 文档 | 如 https://stackoverflow.com/questions/79801731/how-to-keep-git-notes-through-a-rewrite-rebase-rename-amend |

---

*调研完成日期：2025-03-13。若某点标注「未找到」，表示未在公开文档/博客/仓库中查到可靠依据。*
