# API Key 与账号系统架构

## 一、整体设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          用户持有的密钥（Raw API Key）                          │
│  格式: 64 位十六进制 (secrets.token_hex(32))                                    │
│  示例: a1b2c3d4e5f6789012345678abcdef...  (共 64 字符)                         │
│  特点: 创建时仅返回一次，之后不可再查看完整值                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          认证时服务端计算                                       │
│  key_hash = SHA256(raw_key)  →  64 位十六进制                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          数据库 api_keys 表存储                                 │
│  key_hash (唯一)  │  user_name  │  role  │  key_prefix  │  key_suffix  │ ...   │
│  不存 raw_key，只存 hash + 前后缀用于展示                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 二、密钥层级说明

| 层级 | 名称 | 长度 | 谁持有 | 存储位置 |
|------|------|------|--------|----------|
| **L1** | Raw API Key（原始密钥） | 64  hex | 用户 | 不存储，创建时仅返回一次 |
| **L2** | key_hash（SHA256） | 64 hex | 服务端 | `api_keys.key_hash` |
| **L3** | key_prefix / key_suffix | 各 4 字符 | 服务端 | 用于展示 `a1b2****c3d4` |

**你拿到的 `0D5007FEF6A98F5A99ED521327C9A698` 属于哪一层？**

- **32 字符** ≠ 标准 Raw Key（应为 64 字符）
- **32 字符** ≠ key_hash（应为 64 字符）
- 可能是：**Raw Key 的前半段**（被截断）、或**旧版/自定义格式**

若你把它当作 Raw Key 使用（`TEAM_MEMORY_API_KEY=0D5007FEF6A98F5A99ED521327C9A698`）：
- 服务端会计算 `SHA256("0D5007FEF6A98F5A99ED521327C9A698")` 得到 64 位 hash
- 用该 hash 在 DB 中查找 → 若 DB 里没有这条记录，则**数据库认证失败**
- 但若设置了该环境变量，**bootstrap 会把它预注册到内存**，认证仍可通过内存路径成功（见下文）

## 三、数据库表结构 (api_keys)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | 主键 |
| key_hash | varchar(256) | SHA256(raw_key)，唯一，可为空（待审批用户无 Key） |
| user_name | varchar(100) | 用户名，唯一 |
| role | varchar(50) | admin / editor / viewer |
| is_active | bool | 是否激活 |
| password_hash | varchar(256) | bcrypt 密码哈希，可为空 |
| key_prefix | varchar(4) | raw_key 前 4 字符，用于展示 |
| key_suffix | varchar(4) | raw_key 后 4 字符，用于展示 |
| created_at | timestamp | 创建时间 |

**不存储**：Raw API Key 的完整值（安全设计）。

## 四、认证流程

```
                     ┌──────────────────┐
                     │  API Key 认证     │
                     └────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ 内存缓存查找      │             │  DB 查找        │
    │ _keys[key_hash]  │  未命中 →   │ api_keys 表     │
    │ (含 env 预注册)   │             │ WHERE key_hash │
    └────────┬────────┘             └────────┬───────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                     ┌──────────────────┐
                     │ 返回 User 或 None │
                     └──────────────────┘
```

**双路径**：
1. **内存**：`DbApiKeyAuth._keys`，包含 DB 查过的 key + **TEAM_MEMORY_API_KEY 预注册**
2. **DB**：`api_keys` 表，持久化用户

## 五、TEAM_MEMORY_API_KEY 的特殊行为

在 `bootstrap._configure_auth` 中：

```python
env_key = (os.environ.get("TEAM_MEMORY_API_KEY") or "").strip()
if env_key:
    user_name = (os.environ.get("TEAM_MEMORY_USER") or "admin").strip()
    auth.register_key(env_key, user_name, "admin")  # 写入内存 _keys
```

因此：
- 设置 `TEAM_MEMORY_API_KEY=0D5007FEF6A98F5A99ED521327C9A698` 时，该值会**直接作为 Raw Key** 注册到内存
- 认证时先查内存 → 命中 → 以 admin 身份通过
- **无需**在 DB 中存在对应 key_hash

所以你能用 32 字符的 key 登录，是因为**环境变量预注册**，而不是 DB 中有这条记录。

## 六、与 admin 不匹配的原因

- DB 中 admin 的 `key_hash` 是另一把 64 字符 Raw Key 的 SHA256
- 你用的 `0D5007FEF6A98F5A99ED521327C9A698` 通过 env 预注册进内存，**绕过了 DB**
- 使用统计等按 `api_key_name`（来自 `TEAM_MEMORY_API_KEY_NAME`）或 DB 用户聚合时，会出现「0D5007FEF6A98F5A99ED521327C9A698 与 admin 不对应」的现象

## 七、建议

1. **统一密钥**：为 admin 在 Web 中重新生成 API Key（64 字符），用该值设置 `TEAM_MEMORY_API_KEY`，并设置 `TEAM_MEMORY_API_KEY_NAME=admin`
2. **或**：在 DB 中为 admin 更新 key_hash，使其等于 `SHA256("0D5007FEF6A98F5A99ED521327C9A698")`（需确认该 32 字符即你希望使用的完整密钥）
