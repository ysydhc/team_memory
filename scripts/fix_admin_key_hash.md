# 修复 API Key 与 admin 账户不匹配

## 问题说明

`0D5007FEF6A98F5A99ED521327C9A698` 是 API Key 的 **SHA256 哈希**（或前 32 位）。认证时用 `hashlib.sha256(api_key.encode()).hexdigest()` 计算，与 `api_keys` 表中的 `key_hash` 比对。

若不匹配 admin，通常有两种情况：

1. **该 key_hash 属于其他用户**：`api_keys` 表中存在 `key_hash = '0D5007FEF6A98F5A698...'` 但 `user_name != 'admin'`
2. **admin 的 key_hash 不是这个**：admin 行用的是另一把 Key 的哈希，你当前用的 Key 对应别的用户

## 排查步骤

```bash
# 连接数据库后执行
psql $DATABASE_URL -c "
  SELECT id, user_name, role, 
         LEFT(key_hash::text, 16) as key_hash_prefix,
         is_active 
  FROM api_keys 
  WHERE key_hash LIKE '0D5007FEF6A98F5A99ED521327C9A698%' 
     OR user_name = 'admin';
"
```

> 注意：完整 SHA256 为 64 位十六进制。若你只有 32 位，用 `LIKE '0D5007FEF6A98F5A99ED521327C9A698%'` 做前缀匹配。

## 修复方式

### 方式 A：把该 key_hash 对应的用户改为 admin

若确认该哈希应属于 admin：

```sql
UPDATE api_keys 
SET user_name = 'admin', role = 'admin' 
WHERE key_hash LIKE '0D5007FEF6A98F5A99ED521327C9A698%';
```

### 方式 B：为 admin 重新生成 API Key

在 Web 设置 → 用户与 Key 管理 中，为 admin 重新生成 Key，或使用「审批通过」为 admin 分配新 Key。

### 方式 C：确认 MCP 使用的 Key

检查环境变量：

- `TEAM_MEMORY_API_KEY`：MCP 使用的 API Key
- `TEAM_MEMORY_API_KEY_NAME`：用于使用统计展示的名称（建议设为 `admin` 等可读名，不要用哈希）

若 `TEAM_MEMORY_API_KEY` 与 admin 在数据库中的 Key 不一致，需要：
- 在 Web 中查看 admin 的 Key（仅显示一次），或
- 为 admin 重新生成 Key 并更新环境变量
