import React, { useState } from "react";
import {
  Activity,
  BookOpen,
  Building2,
  Check,
  ChevronsUpDown,
  KeyRound,
  LogOut,
  Moon,
  Plus,
  Search,
  Settings2,
  Sun,
} from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { useTheme } from "@/contexts/ThemeContext";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge, Button, Input } from "@/components/ui/primitives";

export type NavTab = "overview" | "memories" | "access" | "ops";

interface AppShellProps {
  activeTab: NavTab;
  onNavigate: (tab: NavTab) => void;
  onOpenCommandPalette: () => void;
  children: React.ReactNode;
}

export function AppShell({
  activeTab,
  onNavigate,
  onOpenCommandPalette,
  children,
}: AppShellProps) {
  const { user, isAdmin, currentTenant, tenants, switchTenant, refreshMe, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [orgMenuOpen, setOrgMenuOpen] = useState(false);
  const [createOrgOpen, setCreateOrgOpen] = useState(false);
  const [newOrgName, setNewOrgName] = useState("");
  const [creatingOrg, setCreatingOrg] = useState(false);
  const [orgError, setOrgError] = useState("");

  const handleCreateTenant = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newOrgName.trim()) return;
    setCreatingOrg(true);
    setOrgError("");
    try {
      const res = await api.createTenant({ name: newOrgName.trim() });
      setCreateOrgOpen(false);
      setNewOrgName("");
      await switchTenant(res.tenant.id);
      await refreshMe();
    } catch (err) {
      setOrgError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setCreatingOrg(false);
    }
  };

  const navItems: Array<{ id: NavTab; label: string; icon: React.ReactNode; adminOnly?: boolean }> = [
    { id: "overview", label: "驾驶舱", icon: <Activity className="h-4 w-4" /> },
    { id: "memories", label: "记忆库", icon: <BookOpen className="h-4 w-4" /> },
    { id: "access", label: "团队与接入", icon: <KeyRound className="h-4 w-4" /> },
    { id: "ops", label: "系统运维", icon: <Settings2 className="h-4 w-4" />, adminOnly: true },
  ];

  const visibleNav = navItems.filter((item) => !item.adminOnly || isAdmin);

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Left Sidebar */}
      <aside className="flex w-60 shrink-0 flex-col border-r border-zinc-200/80 bg-white dark:border-zinc-800/80 dark:bg-zinc-950">
        {/* Org Switcher */}
        <div className="relative border-b border-zinc-200/70 p-3 dark:border-zinc-800/70">
          <button
            type="button"
            onClick={() => setOrgMenuOpen((v) => !v)}
            className="flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-zinc-100 dark:hover:bg-zinc-900 cursor-pointer"
          >
            <div className="flex min-w-0 items-center gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-zinc-900 text-xs font-semibold text-white dark:bg-zinc-100 dark:text-zinc-900">
                {(currentTenant?.name || "TM").slice(0, 2).toUpperCase()}
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold leading-tight">
                  {currentTenant?.name || "Default Workspace"}
                </div>
                <div className="truncate text-[11px] text-zinc-500">
                  {currentTenant?.slug || "default"} · {user?.role}
                </div>
              </div>
            </div>
            <ChevronsUpDown className="h-4 w-4 shrink-0 text-zinc-400" />
          </button>

          {orgMenuOpen && (
            <>
              <div
                className="fixed inset-0 z-20"
                onClick={() => setOrgMenuOpen(false)}
              />
              <div className="absolute left-3 right-3 top-14 z-30 rounded-xl border border-zinc-200 bg-white p-1.5 shadow-lg dark:border-zinc-800 dark:bg-zinc-900">
                <div className="px-2 py-1 text-[11px] font-medium text-zinc-400">
                  切换工作空间（租户）
                </div>
                <div className="max-h-52 space-y-0.5 overflow-y-auto">
                  {tenants.map((t) => {
                    const active = t.id === currentTenant?.id;
                    return (
                      <button
                        key={t.id}
                        type="button"
                        onClick={async () => {
                          setOrgMenuOpen(false);
                          if (!active) await switchTenant(t.id);
                        }}
                        className={cn(
                          "flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-sm transition-colors cursor-pointer",
                          active
                            ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
                            : "text-zinc-600 hover:bg-zinc-50 dark:text-zinc-300 dark:hover:bg-zinc-800/60",
                        )}
                      >
                        <span className="flex min-w-0 items-center gap-2">
                          <Building2 className="h-3.5 w-3.5 shrink-0 text-zinc-400" />
                          <span className="truncate">{t.name}</span>
                        </span>
                        <span className="flex shrink-0 items-center gap-1.5">
                          <span className="text-[10px] text-zinc-400">{t.role}</span>
                          {active && <Check className="h-3.5 w-3.5 text-emerald-500" />}
                        </span>
                      </button>
                    );
                  })}
                </div>
                <div className="mt-1 border-t border-zinc-100 pt-1 dark:border-zinc-800">
                  <button
                    type="button"
                    onClick={() => {
                      setOrgMenuOpen(false);
                      setCreateOrgOpen(true);
                    }}
                    className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800 cursor-pointer"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    新建租户空间
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-3">
          {visibleNav.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors cursor-pointer",
                  isActive
                    ? "bg-zinc-100 text-zinc-900 dark:bg-zinc-800/90 dark:text-zinc-50"
                    : "text-zinc-600 hover:bg-zinc-100/60 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-100",
                )}
              >
                {item.icon}
                <span>{item.label}</span>
                {item.adminOnly && (
                  <Badge tone="neutral" className="ml-auto text-[10px]">
                    Admin
                  </Badge>
                )}
              </button>
            );
          })}
        </nav>

        {/* Footer: user & theme */}
        <div className="border-t border-zinc-200/70 p-3 dark:border-zinc-800/70">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="truncate text-xs font-medium">{user?.user}</div>
              <div className="truncate text-[11px] text-zinc-400">TeamMemory v2</div>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                title={theme === "dark" ? "切换浅色" : "切换深色"}
                onClick={toggleTheme}
              >
                {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
              <Button variant="ghost" size="icon" title="退出登录" onClick={() => void logout()}>
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Column */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {/* Topbar */}
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-200/80 bg-white/80 px-6 backdrop-blur dark:border-zinc-800/80 dark:bg-zinc-950/80">
          <button
            type="button"
            onClick={onOpenCommandPalette}
            className="flex h-9 w-full max-w-md items-center justify-between rounded-lg border border-zinc-200 bg-zinc-50/80 px-3 text-sm text-zinc-500 transition-colors hover:border-zinc-300 hover:bg-white dark:border-zinc-800 dark:bg-zinc-900/70 dark:text-zinc-400 dark:hover:border-zinc-700 cursor-pointer"
          >
            <span className="flex items-center gap-2">
              <Search className="h-3.5 w-3.5" />
              <span>语义搜索团队经验、归档...</span>
            </span>
            <kbd className="rounded border border-zinc-200 bg-white px-1.5 py-0.5 font-mono text-[10px] text-zinc-400 dark:border-zinc-700 dark:bg-zinc-800">
              ⌘K
            </kbd>
          </button>

          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <span className="hidden items-center gap-1.5 sm:inline-flex">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              <span>租户隔离已激活</span>
            </span>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto px-6 py-6">{children}</main>
      </div>

      {/* Create Tenant Modal */}
      {createOrgOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-xs">
          <div className="w-full max-w-sm rounded-xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
            <h3 className="text-base font-semibold">新建租户工作空间</h3>
            <p className="mt-1 text-xs text-zinc-500">
              每个租户拥有独立的 PostgreSQL RLS 数据隔离与成员权限。
            </p>
            <form onSubmit={handleCreateTenant} className="mt-4 space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-300">
                  空间名称
                </label>
                <Input
                  autoFocus
                  placeholder="例如：ADX-SDK 团队"
                  value={newOrgName}
                  onChange={(e) => setNewOrgName(e.target.value)}
                />
              </div>
              {orgError && <p className="text-xs text-red-500">{orgError}</p>}
              <div className="flex justify-end gap-2 pt-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => setCreateOrgOpen(false)}
                >
                  取消
                </Button>
                <Button type="submit" size="sm" disabled={creatingOrg || !newOrgName.trim()}>
                  {creatingOrg ? "创建中..." : "创建并切换"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
