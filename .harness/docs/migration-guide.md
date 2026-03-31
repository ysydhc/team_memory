# Migration Guide — 迁移到新项目

> 将 `.harness/` 框架移植到任何新项目的完整指南。

---

## 前提

- 新项目已有 Git 仓库
- 已确定使用的 AI 编码平台（Claude Code / Cursor / 两者）

---

## 7 步迁移

### Step 1: 复制 `.harness/` 目录

```bash
cp -r /path/to/source/.harness /path/to/new-project/.harness
```

复制后的目录结构：

```
.harness/
├── harness-config.yaml
├── orchestration/
│   ├── task-flow.md
│   ├── context-management.md
│   ├── contracts/
│   │   ├── plan-contract.md
│   │   └── task-contract.md
│   └── extensions/
│       └── README.md
├── failure/
│   └── failure-taxonomy.md
├── hooks/
│   ├── pre-bash-guard.sh
│   ├── post-edit-format.sh
│   └── stop-check.sh
├── plans/
│   ├── progress.md
│   └── completed/
└── docs/
    ├── harness-spec.md
    └── migration-guide.md
```

### Step 2: 修改 `harness-config.yaml`

必须修改的字段：

```yaml
project:
  name: your-project-name          # ← 改为新项目名
  description: your description     # ← 改为新项目描述

quality_gates:
  lint: "your lint command"         # ← 改为项目 lint 命令
  format: "your format command"     # ← 改为项目 format 命令
  test: "your test command"         # ← 改为项目 test 命令
  verify: "your verify command"     # ← 改为项目验收命令
  harness: "your harness command"   # ← 改为项目 harness 检查命令

layering:
  forbidden_imports: [...]          # ← 根据项目架构调整
```

可选调整：

```yaml
timeouts:                           # 根据项目规模调整
context:                            # 根据 Agent 使用习惯调整
failure:                            # 根据项目容错需求调整
security:
  blocked_commands: [...]           # 根据项目需要增删
  blocked_interactive: [...]        # 通常不需要改
```

### Step 3: 调整 Hook 脚本

检查 `.harness/hooks/` 下的三个脚本：

| 脚本 | 需要调整的内容 |
|------|--------------|
| `pre-bash-guard.sh` | 阻止规则列表（通常不需要改） |
| `post-edit-format.sh` | **formatter 命令**（如从 `ruff format` 改为 `prettier`） |
| `stop-check.sh` | **验收命令**（如从 `make verify` 改为 `npm run test`） |

确保可执行：

```bash
chmod +x .harness/hooks/*.sh
```

### Step 4: 创建平台入口文件

#### Claude Code

创建或更新 `CLAUDE.md`：
- 在导航表中添加 `.harness/` 相关链接
- 在完成标准中引用 `harness-config.yaml` 的质量门禁

创建或更新 `.claude/settings.json`：

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": ".harness/hooks/pre-bash-guard.sh",
        "timeout": 5000
      }]
    }],
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": ".harness/hooks/post-edit-format.sh",
        "timeout": 10000
      }]
    }],
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": ".harness/hooks/stop-check.sh",
        "timeout": 15000
      }]
    }]
  }
}
```

#### Cursor

创建 `AGENTS.md`（Cursor 不读 `CLAUDE.md`），内容镜像 `CLAUDE.md` 但：
- 不使用 `@import` 语法
- 包含 `.harness/` 导航

Cursor 自动读取 `.claude/settings.json` 中的 hooks，无需额外配置。

### Step 5: 清空计划状态

```bash
# 清空 progress.md（保留模板头）
cat > .harness/plans/progress.md << 'EOF'
# Active Plans

> 轻量状态文件，追踪所有活跃计划。硬限制：< 100 行。
> 完成的计划移到 `completed/` 目录。
EOF

# 清空 completed 目录
rm -f .harness/plans/completed/*.md
```

### Step 6: 确认 `.gitignore`

确保 `.harness/` **不在** `.gitignore` 中（框架应被版本控制）：

```bash
# 检查
grep -n "\.harness" .gitignore || echo "OK: .harness not in .gitignore"
```

注意：`.claude/` 可能在 `.gitignore` 中，这是正常的——`.claude/` 是平台特有配置。

### Step 7: 验证

```bash
# 1. Hook 可执行
ls -la .harness/hooks/*.sh

# 2. YAML 合法
python3 -c "import yaml; yaml.safe_load(open('.harness/harness-config.yaml')); print('OK')"

# 3. Hook 功能测试
echo '{"tool_input":{"command":"ls"}}' | .harness/hooks/pre-bash-guard.sh && echo "PASS"
echo '{"tool_input":{"command":"rm -rf /"}}' | .harness/hooks/pre-bash-guard.sh 2>/dev/null; [ $? -eq 2 ] && echo "PASS: blocked"

# 4. 文件完整性
find .harness -type f | wc -l   # 应 ≥ 14

# 5. Git 可追踪
git add --dry-run .harness/
```

---

## 验证清单

| 检查项 | 命令 | 预期 |
|--------|------|------|
| 目录存在 | `ls .harness/` | 不报错 |
| 配置合法 | `python3 -c "import yaml; ..."` | 输出 OK |
| Hook 可执行 | `ls -la .harness/hooks/*.sh` | 有 x 权限 |
| 危险命令被阻止 | `echo ... \| pre-bash-guard.sh` | exit 2 |
| 安全命令放行 | `echo ... \| pre-bash-guard.sh` | exit 0 |
| 计划状态干净 | `cat .harness/plans/progress.md` | 无活跃计划 |
| Git 可追踪 | `git add --dry-run .harness/` | 列出所有文件 |
| 平台入口存在 | `ls CLAUDE.md AGENTS.md` | 至少一个存在 |

---

## 常见问题

| 问题 | 解决 |
|------|------|
| Hook 不执行 | 检查 `chmod +x` 和 `settings.json` 路径 |
| Cursor 不执行 Hook | 确认 `.claude/settings.json` 存在且格式正确 |
| `harness-config.yaml` 解析失败 | 检查 YAML 语法（缩进、引号） |
| 新项目的质量门禁不匹配 | 修改 `harness-config.yaml` 的 `quality_gates` |
| 需要项目特有扩展 | 在 `orchestration/extensions/` 下添加 `after-phase-{N}.md` |
