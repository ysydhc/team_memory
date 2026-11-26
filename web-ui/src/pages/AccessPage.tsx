import React, { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  CheckCircle2,
  Copy,
  KeyRound,
  Loader2,
  Plus,
  RefreshCw,
  Shield,
  Terminal,
  Trash2,
  UserPlus,
} from "lucide-react";
import { api, type TenantKeyItem, type TenantMemberItem } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { cn, copyToClipboard, formatRelativeTime } from "@/lib/utils";
import { Badge, Button, Card, Input } from "@/components/ui/primitives";

type ClientPlatform = "antigravity" | "claude" | "cursor" | "codex" | "cli";

export function AccessPage() {
  const { user, isAdmin, currentTenant } = useAuth();
  const queryClient = useQueryClient();
  const tid = currentTenant?.id || "default";

  // Onboarding snippet state
  const [platform, setPlatform] = useState<ClientPlatform>("antigravity");
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [connStatus, setConnStatus] = useState<"idle" | "testing" | "ok" | "fail">("idle");
  const [connDetail, setConnDetail] = useState("");

  // API Key creation & rotation state
  const [newKeyLabel, setNewKeyLabel] = useState("");
  const [creatingKey, setCreatingKey] = useState(false);
  const [rotatingKey, setRotatingKey] = useState(false);

  // Admin reset password modal state
  const [resetPwdUser, setResetPwdUser] = useState<string | null>(null);
  const [resetPwdVal, setResetPwdVal] = useState("");
  const [resetPwdBusy, setResetPwdBusy] = useState(false);
  const [resetPwdMsg, setResetPwdMsg] = useState("");

  // Admin member modal state
  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteUser, setInviteUser] = useState("");
  const [inviteRole, setInviteRole] = useState<"member" | "viewer" | "admin">("member");
  const [invitePwd, setInvitePwd] = useState("");
  const [inviteGenKey, setInviteGenKey] = useState(true);
  const [inviteResultKey, setInviteResultKey] = useState<string | null>(null);
  const [inviteErr, setInviteErr] = useState("");
  const [inviting, setInviting] = useState(false);

  const keysQuery = useQuery({
    queryKey: ["tenant-keys", tid],
    queryFn: () => api.listTenantKeys(),
  });

  const membersQuery = useQuery({
    queryKey: ["tenant-members", tid],
    enabled: isAdmin,
    queryFn: () => api.listMembers(),
  });

  const serverOrigin =
    typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:9111";
  const displayKey = revealedKey || "<YOUR_TENANT_API_KEY>";

  const snippets: Record<ClientPlatform, string> = {
    antigravity: JSON.stringify(
      {
        mcpServers: {
          team_memory: {
            serverUrl: `${serverOrigin}/mcp/mcp`,
            headers: {
              Authorization: `Bearer ${displayKey}`,
            },
          },
        },
      },
      null,
      2,
    ),
    claude: `claude mcp add --transport http team_memory ${serverOrigin}/mcp/mcp \\\n  --header "Authorization: Bearer ${displayKey}"`,
    cursor: JSON.stringify(
      {
        mcpServers: {
          team_memory: {
            url: `${serverOrigin}/mcp/mcp`,
            headers: {
              Authorization: `Bearer ${displayKey}`,
            },
          },
        },
      },
      null,
      2,
    ),
    codex: `# ~/.codex/config.toml\n[mcp_servers.team_memory]\nurl = "${serverOrigin}/mcp/mcp"\nhttp_headers = { Authorization = "Bearer ${displayKey}" }\n\n# 可选：一键配置 Codex CLI Hooks + MCP（本地 Daemon 采集）\n# tm-cli setup --platform codex`,
    cli: `tm-cli config init --url ${serverOrigin} --api-key "${displayKey}"\ntm-cli status`,
  };

  const handleCopySnippet = async () => {
    const ok = await copyToClipboard(snippets[platform]);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleTestConnectivity = async () => {
    setConnStatus("testing");
    setConnDetail("");
    try {
      const [h, me] = await Promise.all([api.health(), api.me()]);
      setConnStatus("ok");
      setConnDetail(
        `连通正常 · 服务状态 ${h.status} · 当前租户 ${me.current_tenant?.name || me.tenant_id}`,
      );
    } catch (err) {
      setConnStatus("fail");
      setConnDetail(err instanceof Error ? err.message : "连通性检查失败");
    }
  };

  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreatingKey(true);
    try {
      const res = await api.createTenantKey({
        label: newKeyLabel.trim() || `${currentTenant?.slug || "default"}-key`,
      });
      setRevealedKey(res.api_key);
      setNewKeyLabel("");
      await queryClient.invalidateQueries({ queryKey: ["tenant-keys", tid] });
    } finally {
      setCreatingKey(false);
    }
  };

  const handleRevokeKey = async (id: number) => {
    await api.revokeTenantKey(id);
    await queryClient.invalidateQueries({ queryKey: ["tenant-keys", tid] });
  };

  const handleRotateKey = async () => {
    setRotatingKey(true);
    try {
      const res = await api.rotateOwnKey();
      if (res.api_key) setRevealedKey(res.api_key);
      await queryClient.invalidateQueries({ queryKey: ["tenant-keys", tid] });
    } finally {
      setRotatingKey(false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resetPwdUser || resetPwdVal.length < 6) return;
    setResetPwdBusy(true);
    setResetPwdMsg("");
    try {
      await api.adminResetPassword(resetPwdUser, resetPwdVal);
      setResetPwdMsg(`已重置 ${resetPwdUser} 的密码`);
      setResetPwdVal("");
      setTimeout(() => {
        setResetPwdUser(null);
        setResetPwdMsg("");
      }, 1200);
    } catch (err) {
      setResetPwdMsg(err instanceof Error ? err.message : "重置失败");
    } finally {
      setResetPwdBusy(false);
    }
  };

  const handleInviteMember = async (e: React.FormEvent) => {
    e.preventDefault();
    setInviting(true);
    setInviteErr("");
    setInviteResultKey(null);
    try {
      const res = await api.addMember({
        user_id: inviteUser.trim(),
        role: inviteRole,
        password: invitePwd.trim() || undefined,
        generate_api_key: inviteGenKey,
      });
      if (res.api_key) setInviteResultKey(res.api_key);
      else setInviteOpen(false);
      setInviteUser("");
      setInvitePwd("");
      await queryClient.invalidateQueries({ queryKey: ["tenant-members", tid] });
    } catch (err) {
      setInviteErr(err instanceof Error ? err.message : "邀请失败");
    } finally {
      setInviting(false);
    }
  };

  const handleRoleChange = async (userId: string, role: string) => {
    await api.updateMember(userId, role);
    await queryClient.invalidateQueries({ queryKey: ["tenant-members", tid] });
  };

  const handleRemoveMember = async (userId: string) => {
    await api.removeMember(userId);
    await queryClient.invalidateQueries({ queryKey: ["tenant-members", tid] });
  };

  const platformTabs: Array<{ id: ClientPlatform; label: string }> = [
    { id: "antigravity", label: "Antigravity" },
    { id: "claude", label: "Claude Code" },
    { id: "cursor", label: "Cursor" },
    { id: "codex", label: "Codex CLI" },
    { id: "cli", label: "CLI (tm-cli)" },
  ];

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">团队与接入</h1>
        <p className="text-xs text-zinc-500">
          10 秒完成当前租户（<strong>{currentTenant?.name || "Default Workspace"}</strong>）的
          MCP / CLI 接入，管理个人 Key 与团队成员角色
        </p>
      </div>

      {/* 1. 10-Second Onboarding Card */}
      <Card className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Terminal className="h-4 w-4 text-zinc-500" />
            <h2 className="text-sm font-semibold">10 秒一键接入向导</h2>
            {revealedKey ? (
              <Badge tone="green">已自动填入新生成的租户 Key</Badge>
            ) : (
              <Badge tone="neutral">下方生成 Key 后自动填入</Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              disabled={connStatus === "testing"}
              onClick={() => void handleTestConnectivity()}
            >
              {connStatus === "testing" ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <RefreshCw className="h-3.5 w-3.5" />
              )}
              测试连通性
            </Button>
            <Button size="sm" onClick={() => void handleCopySnippet()}>
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "已复制" : "一键复制配置"}
            </Button>
          </div>
        </div>

        {/* Platform tabs */}
        <div className="mt-4 inline-flex rounded-lg bg-zinc-100 p-1 dark:bg-zinc-900">
          {platformTabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setPlatform(t.id)}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors cursor-pointer",
                platform === t.id
                  ? "bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>

        <pre className="mt-3 overflow-x-auto rounded-xl border border-zinc-200/80 bg-zinc-950 p-4 font-mono text-xs leading-relaxed text-zinc-100 dark:border-zinc-800">
          {snippets[platform]}
        </pre>

        {connDetail && (
          <div
            className={cn(
              "mt-3 flex items-center gap-2 rounded-lg px-3 py-2 text-xs",
              connStatus === "ok"
                ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
                : "bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300",
            )}
          >
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
            <span>{connDetail}</span>
          </div>
        )}
      </Card>

      {/* 2. My API Keys (Scoped to Current Tenant) */}
      <Card className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <KeyRound className="h-4 w-4 text-zinc-500" />
            <h2 className="text-sm font-semibold">我在本租户的 API Keys</h2>
            <span className="text-xs text-zinc-400">
              每个 Key 严格绑定当前租户（{currentTenant?.slug || "default"}）
            </span>
          </div>

          <form onSubmit={handleCreateKey} className="flex flex-wrap items-center gap-2">
            <Input
              placeholder="备注名，如 macbook-pro"
              value={newKeyLabel}
              onChange={(e) => setNewKeyLabel(e.target.value)}
              className="h-8 w-44 text-xs"
            />
            <Button type="submit" size="sm" disabled={creatingKey}>
              <Plus className="h-3.5 w-3.5" />
              {creatingKey ? "生成中..." : "生成新 Key"}
            </Button>
            {(keysQuery.data?.keys || []).length > 0 && (
              <Button
                type="button"
                size="sm"
                variant="secondary"
                disabled={rotatingKey}
                onClick={() => void handleRotateKey()}
              >
                <RefreshCw className={cn("h-3.5 w-3.5", rotatingKey && "animate-spin")} />
                {rotatingKey ? "轮换中..." : "一键轮换"}
              </Button>
            )}
          </form>
        </div>

        {revealedKey && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-200 bg-emerald-50/70 px-3.5 py-2.5 text-xs text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300">
            <div className="min-w-0 flex-1">
              <div className="font-medium">新 API Key（仅展示一次，已同步填入上方接入配置）</div>
              <code className="mt-1 block truncate font-mono text-[11px]">{revealedKey}</code>
            </div>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void copyToClipboard(revealedKey)}
            >
              <Copy className="h-3.5 w-3.5" />
              复制 Key
            </Button>
          </div>
        )}

        <div className="mt-4 divide-y divide-zinc-100 dark:divide-zinc-800/80">
          {(keysQuery.data?.keys || []).length === 0 ? (
            <div className="py-6 text-center text-xs text-zinc-400">
              当前租户下暂无个人 API Key，点击右上角「生成新 Key」即可创建
            </div>
          ) : (
            (keysQuery.data?.keys || []).map((k: TenantKeyItem) => (
              <div key={k.id} className="flex items-center justify-between py-2.5 text-sm">
                <div className="flex items-center gap-3">
                  <span className="font-medium">{k.label}</span>
                  <code className="rounded bg-zinc-100 px-2 py-0.5 font-mono text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                    {k.masked}
                  </code>
                  <Badge tone="neutral">{k.role}</Badge>
                </div>
                <div className="flex items-center gap-3 text-xs text-zinc-400">
                  <span>{formatRelativeTime(k.created_at)}</span>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40"
                    onClick={() => void handleRevokeKey(k.id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    吊销
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </Card>

      {/* 3. Admin-Only Member Management */}
      {isAdmin && (
        <Card className="p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-zinc-500" />
              <h2 className="text-sm font-semibold">租户成员管理</h2>
              <Badge tone="neutral">仅 Admin 可见</Badge>
            </div>
            <Button
              size="sm"
              onClick={() => {
                setInviteResultKey(null);
                setInviteErr("");
                setInviteOpen(true);
              }}
            >
              <UserPlus className="h-3.5 w-3.5" />
              添加成员
            </Button>
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-zinc-200/80 text-zinc-400 dark:border-zinc-800">
                  <th className="py-2 font-medium">成员</th>
                  <th className="py-2 font-medium">租户角色</th>
                  <th className="py-2 font-medium">活跃 Key</th>
                  <th className="py-2 font-medium">加入时间</th>
                  <th className="py-2 text-right font-medium">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/70">
                {(membersQuery.data?.members || []).map((m: TenantMemberItem) => (
                  <tr key={m.id}>
                    <td className="py-2.5 font-medium text-zinc-900 dark:text-zinc-100">
                      {m.user_id}
                      {m.user_id === user?.user && (
                        <span className="ml-1.5 text-[10px] text-zinc-400">(你)</span>
                      )}
                    </td>
                    <td className="py-2.5">
                      <select
                        value={m.role}
                        onChange={(e) => void handleRoleChange(m.user_id, e.target.value)}
                        className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                      >
                        <option value="admin">Admin</option>
                        <option value="member">Member</option>
                        <option value="viewer">Viewer</option>
                      </select>
                    </td>
                    <td className="py-2.5 text-zinc-500">{m.active_key_count ?? 0}</td>
                    <td className="py-2.5 text-zinc-400">{formatRelativeTime(m.created_at)}</td>
                    <td className="py-2.5 text-right">
                      <div className="inline-flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            setResetPwdUser(m.user_id);
                            setResetPwdVal("");
                            setResetPwdMsg("");
                          }}
                        >
                          重置密码
                        </Button>
                        {m.user_id !== user?.user && (
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-red-600"
                            onClick={() => void handleRemoveMember(m.user_id)}
                          >
                            移除
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Admin Reset Password Modal */}
      {resetPwdUser && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-xs"
          onClick={() => setResetPwdUser(null)}
        >
          <div
            className="w-full max-w-sm rounded-2xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-900"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-base font-semibold">重置成员密码</h3>
            <p className="mt-1 text-xs text-zinc-500">
              为成员 <strong>{resetPwdUser}</strong> 设置新的登录密码（至少 6 位）。
            </p>
            <form onSubmit={handleResetPassword} className="mt-4 space-y-3">
              <Input
                type="password"
                autoFocus
                minLength={6}
                required
                placeholder="输入新密码（至少 6 位）"
                value={resetPwdVal}
                onChange={(e) => setResetPwdVal(e.target.value)}
              />
              {resetPwdMsg && (
                <p className="text-xs text-emerald-600 dark:text-emerald-400">{resetPwdMsg}</p>
              )}
              <div className="flex justify-end gap-2 pt-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => setResetPwdUser(null)}
                >
                  取消
                </Button>
                <Button
                  type="submit"
                  size="sm"
                  disabled={resetPwdBusy || resetPwdVal.length < 6}
                >
                  {resetPwdBusy ? "提交中..." : "确认重置"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Invite Member Modal */}
      {inviteOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-xs"
          onClick={() => setInviteOpen(false)}
        >
          <div
            className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-900"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-base font-semibold">添加租户成员</h3>
            {inviteResultKey ? (
              <div className="mt-4 space-y-3">
                <p className="text-xs text-emerald-600 dark:text-emerald-400">
                  成员已加入当前租户，以下 API Key 仅显示一次：
                </p>
                <code className="block break-all rounded-lg bg-zinc-100 p-3 font-mono text-xs dark:bg-zinc-800">
                  {inviteResultKey}
                </code>
                <div className="flex justify-end gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => void copyToClipboard(inviteResultKey)}
                  >
                    复制 Key
                  </Button>
                  <Button size="sm" onClick={() => setInviteOpen(false)}>
                    完成
                  </Button>
                </div>
              </div>
            ) : (
              <form onSubmit={handleInviteMember} className="mt-4 space-y-3">
                <div>
                  <label className="mb-1 block text-xs font-medium text-zinc-500">用户名</label>
                  <Input
                    required
                    placeholder="alice"
                    value={inviteUser}
                    onChange={(e) => setInviteUser(e.target.value)}
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-zinc-500">角色</label>
                  <select
                    value={inviteRole}
                    onChange={(e) =>
                      setInviteRole(e.target.value as "admin" | "member" | "viewer")
                    }
                    className="h-9 w-full rounded-lg border border-zinc-200 bg-white px-2.5 text-sm dark:border-zinc-800 dark:bg-zinc-900"
                  >
                    <option value="member">Member（读写经验与归档）</option>
                    <option value="viewer">Viewer（仅检索与查看）</option>
                    <option value="admin">Admin（管理成员与系统运维）</option>
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-zinc-500">
                    初始密码（新用户选填，已有用户可留空）
                  </label>
                  <Input
                    type="password"
                    placeholder="至少 6 位"
                    value={invitePwd}
                    onChange={(e) => setInvitePwd(e.target.value)}
                  />
                </div>
                <label className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={inviteGenKey}
                    onChange={(e) => setInviteGenKey(e.target.checked)}
                  />
                  同时生成该租户的专属 API Key
                </label>
                {inviteErr && <p className="text-xs text-red-500">{inviteErr}</p>}
                <div className="flex justify-end gap-2 pt-2">
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onClick={() => setInviteOpen(false)}
                  >
                    取消
                  </Button>
                  <Button type="submit" size="sm" disabled={inviting}>
                    {inviting ? "添加中..." : "确认添加"}
                  </Button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
