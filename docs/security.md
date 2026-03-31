# 安全约定

## 硬性规则（违反直接打回）

1. **API Key 和密码禁止出现在代码中**
   - 使用环境变量或 `config.development.yaml` / `config.production.yaml`（通过 `${ENV_VAR}` 引用）
   - 仅限本机的覆盖文件可通过 `TEAM_MEMORY_CONFIG_PATH` 指向未入库路径
   - 生产环境禁止使用 `auth.type: none`

2. **日志中禁止打印敏感信息**
   ```python
   # 错误
   logger.info(f"user login: api_key={api_key}")

   # 正确
   logger.info("user_login", extra={"user": username, "key_prefix": api_key[:8]+"..."})
   ```

3. **外部输入必须验证**
   - 所有 MCP 参数通过 Pydantic model 或类型注解校验
   - 数据库查询参数使用 SQLAlchemy 绑定参数，禁止字符串拼接 SQL

## 依赖安全

- 新增依赖前确认：是否有已知 CVE、维护是否活跃
- 禁止 AI 自行添加未经审查的第三方包
- `pyproject.toml` 的变更需要在 PR 说明中解释原因

## 数据隐私

- `project` 参数用于隔离不同项目数据，跨项目查询需要显式授权
- 个人记忆表 `personal_memories`（用户画像源）：仅当前登录用户可 **列表 / 读写 / 删除** 自己的行；团队经验中的 `visibility=private` 等仍按 Experience RBAC，与画像表分离
- 审计日志记录所有写操作（who、what、when），不可删除
