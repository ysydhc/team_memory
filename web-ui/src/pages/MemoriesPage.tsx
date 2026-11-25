import React, { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive as ArchiveIcon,
  BookOpen,
  Bug,
  CheckCircle2,
  FileEdit,
  Loader2,
  Pin,
  Plus,
  Search,
  Trash2,
  User as UserIcon,
  X,
} from "lucide-react";
import {
  api,
  type ArchiveItem,
  type ExperienceItem,
  type PersonalMemoryItem,
} from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { cn, formatRelativeTime } from "@/lib/utils";
import { Badge, Button, Card, Input, Textarea } from "@/components/ui/primitives";
import type { DrawerTarget } from "@/components/MemoryDrawer";

type MemoryView = "experiences" | "drafts" | "archives" | "personal";

interface MemoriesPageProps {
  initialView?: MemoryView;
  onOpenDrawer: (target: DrawerTarget) => void;
}

export function MemoriesPage({ initialView = "experiences", onOpenDrawer }: MemoriesPageProps) {
  const { currentTenant, canWrite } = useAuth();
  const queryClient = useQueryClient();
  const tid = currentTenant?.id || "default";

  const [view, setView] = useState<MemoryView>(initialView);
  const [project, setProject] = useState<string>("");
  const [tierFilter, setTierFilter] = useState<string>("");
  const [searchInput, setSearchInput] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [debugMode, setDebugMode] = useState(false);

  // Create modal state
  const [createOpen, setCreateOpen] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newProblem, setNewProblem] = useState("");
  const [newSolution, setNewSolution] = useState("");
  const [newTags, setNewTags] = useState("");
  const [newPmContent, setNewPmContent] = useState("");
  const [newPmKind, setNewPmKind] = useState<"static" | "dynamic">("static");
  const [creating, setCreating] = useState(false);
  const [createErr, setCreateErr] = useState("");

  const projectsQuery = useQuery({
    queryKey: ["projects", tid],
    queryFn: () => api.listProjects(),
  });

  const isSearching = submittedQuery.trim().length > 0 && view === "experiences";

  const expQuery = useQuery({
    queryKey: ["experiences", tid, view, project, submittedQuery, debugMode],
    enabled: view === "experiences" || view === "drafts",
    queryFn: async () => {
      if (isSearching) {
        if (debugMode) {
          const res = await api.searchDebug({
            query: submittedQuery.trim(),
            project: project || undefined,
            max_results: 20,
          });
          return {
            items: res.results || [],
            total: res.total || 0,
            duration_ms: res.duration_ms,
            search_type: res.search_type,
            stage_metrics: res.stage_metrics,
          };
        }
        const res = await api.search({
          query: submittedQuery.trim(),
          project: project || undefined,
          max_results: 20,
        });
        return { items: res.results || [], total: res.total || 0 };
      }
      const res = await api.listExperiences({
        limit: 50,
        project: project || undefined,
        status: view === "drafts" ? "draft" : "published",
      });
      return { items: res.items || [], total: res.total || 0 };
    },
  });

  const archivesQuery = useQuery({
    queryKey: ["archives", tid, project, submittedQuery],
    enabled: view === "archives",
    queryFn: () =>
      api.listArchives({
        project: project || undefined,
        q: submittedQuery.trim() || undefined,
        limit: 40,
      }),
  });

  const personalQuery = useQuery({
    queryKey: ["personal-memory", tid],
    enabled: view === "personal",
    queryFn: () => api.listPersonalMemory({ limit: 100 }),
  });

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittedQuery(searchInput.trim());
  };

  const handleQuickPublish = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await api.changeExperienceStatus(id, { status: "published" });
    await queryClient.invalidateQueries({ queryKey: ["experiences", tid] });
  };

  const handleQuickDiscard = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await api.deleteExperience(id, false);
    await queryClient.invalidateQueries({ queryKey: ["experiences", tid] });
  };

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setCreateErr("");
    try {
      if (view === "personal") {
        await api.createPersonalMemory({
          content: newPmContent.trim(),
          profile_kind: newPmKind,
        });
        setNewPmContent("");
        await queryClient.invalidateQueries({ queryKey: ["personal-memory", tid] });
      } else {
        await api.createExperience({
          title: newTitle.trim(),
          problem: newProblem.trim(),
          solution: newSolution.trim() || undefined,
          tags: newTags
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean),
          project: project || undefined,
          status: view === "drafts" ? "draft" : "published",
        });
        setNewTitle("");
        setNewProblem("");
        setNewSolution("");
        setNewTags("");
        await queryClient.invalidateQueries({ queryKey: ["experiences", tid] });
      }
      setCreateOpen(false);
    } catch (err) {
      setCreateErr(err instanceof Error ? err.message : "创建失败");
    } finally {
      setCreating(false);
    }
  };

  const rawExpItems = expQuery.data?.items || [];
  const filteredExpItems = tierFilter
    ? rawExpItems.filter((i) => (i.quality_tier || "").toLowerCase() === tierFilter.toLowerCase())
    : rawExpItems;

  const views: Array<{ id: MemoryView; label: string; icon: React.ReactNode }> = [
    { id: "experiences", label: "团队经验", icon: <BookOpen className="h-3.5 w-3.5" /> },
    { id: "drafts", label: "待审草稿", icon: <FileEdit className="h-3.5 w-3.5" /> },
    { id: "archives", label: "会话归档", icon: <ArchiveIcon className="h-3.5 w-3.5" /> },
    { id: "personal", label: "我的画像", icon: <UserIcon className="h-3.5 w-3.5" /> },
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      {/* Header & View Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">记忆库</h1>
          <p className="text-xs text-zinc-500">
            统一管理当前租户下的团队经验、待审草稿、会话归档与个人画像
          </p>
        </div>

        <div className="flex items-center gap-2">
          {(view === "personal" || canWrite) && (
            <Button size="sm" onClick={() => setCreateOpen(true)}>
              <Plus className="h-3.5 w-3.5" />
              {view === "personal" ? "新增画像" : "新建经验"}
            </Button>
          )}
        </div>
      </div>

      {/* Segmented View Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200/80 pb-3 dark:border-zinc-800">
        <div className="inline-flex rounded-lg bg-zinc-100 p-1 dark:bg-zinc-900">
          {views.map((v) => (
            <button
              key={v.id}
              type="button"
              onClick={() => {
                setView(v.id);
                setSubmittedQuery("");
                setSearchInput("");
              }}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer",
                view === v.id
                  ? "bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200",
              )}
            >
              {v.icon}
              {v.label}
            </button>
          ))}
        </div>

        {/* Filters + Search + Debug */}
        {view !== "personal" && (
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={project}
              onChange={(e) => setProject(e.target.value)}
              className="h-9 rounded-lg border border-zinc-200 bg-white px-2.5 text-xs text-zinc-700 focus:outline-none dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200"
            >
              <option value="">全部项目</option>
              {(projectsQuery.data?.projects || []).map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>

            {view === "experiences" && (
              <select
                value={tierFilter}
                onChange={(e) => setTierFilter(e.target.value)}
                className="h-9 rounded-lg border border-zinc-200 bg-white px-2.5 text-xs text-zinc-700 focus:outline-none dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200"
              >
                <option value="">全部质量等级</option>
                <option value="Gold">Gold</option>
                <option value="Silver">Silver</option>
                <option value="Bronze">Bronze</option>
              </select>
            )}

            <form onSubmit={handleSearchSubmit} className="relative flex items-center">
              <Search className="pointer-events-none absolute left-2.5 h-3.5 w-3.5 text-zinc-400" />
              <Input
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder={view === "archives" ? "筛选归档标题..." : "语义检索经验..."}
                className="w-56 pl-8 pr-7 text-xs"
              />
              {searchInput && (
                <button
                  type="button"
                  onClick={() => {
                    setSearchInput("");
                    setSubmittedQuery("");
                  }}
                  className="absolute right-2 text-zinc-400 hover:text-zinc-600 cursor-pointer"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </form>

            {view === "experiences" && (
              <button
                type="button"
                onClick={() => setDebugMode((d) => !d)}
                title="开启后在搜索结果下方展示 4 路召回得分（Vector / FTS / Tag / Graph）"
                className={cn(
                  "inline-flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-xs font-medium transition-colors cursor-pointer",
                  debugMode
                    ? "border-amber-400/60 bg-amber-50 text-amber-800 dark:border-amber-500/40 dark:bg-amber-950/50 dark:text-amber-300"
                    : "border-zinc-200 bg-white text-zinc-500 hover:text-zinc-800 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400",
                )}
              >
                <Bug className="h-3.5 w-3.5" />
                Debug
              </button>
            )}
          </div>
        )}
      </div>

      {/* Debug Banner */}
      {debugMode && isSearching && expQuery.data && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-200 bg-amber-50/70 px-3.5 py-2 text-xs text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300">
          <span>
            Debug 检索模式已开启 · 模式：
            <strong>{expQuery.data.search_type || "hybrid"}</strong> · 命中{" "}
            <strong>{filteredExpItems.length}</strong> 条
          </span>
          <div className="flex flex-wrap items-center gap-3 font-mono text-[11px]">
            {expQuery.data.stage_metrics &&
              Object.entries(expQuery.data.stage_metrics).map(([k, v]) => (
                <span key={k}>
                  {k}: {String(v)}
                </span>
              ))}
            {expQuery.data.duration_ms != null && <span>总耗时 {expQuery.data.duration_ms} ms</span>}
          </div>
        </div>
      )}

      {/* List Content */}
      {(view === "experiences" || view === "drafts") && (
        <div className="space-y-2">
          {expQuery.isLoading ? (
            <div className="flex items-center justify-center py-16 text-xs text-zinc-400">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              加载中...
            </div>
          ) : filteredExpItems.length === 0 ? (
            <Card className="py-16 text-center text-sm text-zinc-400">
              {view === "drafts" ? "暂无待审核草稿" : "当前筛选条件下暂无经验条目"}
            </Card>
          ) : (
            filteredExpItems.map((item: ExperienceItem) => (
              <Card
                key={item.id}
                className="group p-4 transition-colors hover:border-zinc-300 hover:bg-zinc-50/50 dark:hover:border-zinc-700 dark:hover:bg-zinc-900 cursor-pointer"
              >
                <div onClick={() => onOpenDrawer({ kind: "experience", item })}>
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        {item.is_pinned && <Pin className="h-3.5 w-3.5 shrink-0 text-amber-500" />}
                        <h3 className="truncate text-sm font-semibold">{item.title}</h3>
                        {item.valid_until && <Badge tone="red">已废弃</Badge>}
                      </div>
                      <p className="mt-1 line-clamp-2 text-xs text-zinc-500 dark:text-zinc-400">
                        {item.description}
                      </p>
                    </div>

                    {view === "drafts" && canWrite && (
                      <div className="flex shrink-0 items-center gap-1.5">
                        <Button
                          size="sm"
                          onClick={(e) => void handleQuickPublish(e, item.id)}
                        >
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          发布
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="text-red-600"
                          onClick={(e) => void handleQuickDiscard(e, item.id)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                          丢弃
                        </Button>
                      </div>
                    )}
                  </div>

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-zinc-400">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <Badge tone="neutral">{item.project}</Badge>
                      {item.quality_tier && (
                        <Badge
                          tone={
                            item.quality_tier === "Gold"
                              ? "yellow"
                              : item.quality_tier === "Silver"
                                ? "blue"
                                : "neutral"
                          }
                        >
                          {item.quality_tier}
                        </Badge>
                      )}
                      {(item.tags || []).slice(0, 4).map((t) => (
                        <span key={t}>#{t}</span>
                      ))}
                    </div>
                    <div className="flex items-center gap-3">
                      <span>召回 {item.recall_count ?? 0}</span>
                      <span>采纳 {item.used_count ?? 0}</span>
                      <span>{item.created_by}</span>
                      <span>{formatRelativeTime(item.updated_at)}</span>
                    </div>
                  </div>

                  {/* 4-Way Debug Recall Scores */}
                  {debugMode && isSearching && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-2.5 rounded-lg bg-zinc-100/80 px-2.5 py-1.5 font-mono text-[11px] text-zinc-600 dark:bg-zinc-800/70 dark:text-zinc-300">
                      <span>Score: {(item.score ?? item.rrf_score ?? item.similarity ?? 0).toFixed(4)}</span>
                      {item.similarity != null && (
                        <span>Vector: {Number(item.similarity).toFixed(3)}</span>
                      )}
                      {(item.fts_rank != null || item.fts_score != null) && (
                        <span>FTS: {Number(item.fts_rank ?? item.fts_score).toFixed(3)}</span>
                      )}
                      {item.rrf_score != null && (
                        <span>RRF: {Number(item.rrf_score).toFixed(4)}</span>
                      )}
                      {item.tag_score != null && <span>Tag: {Number(item.tag_score).toFixed(3)}</span>}
                      {item.graph_score != null && (
                        <span>Graph: {Number(item.graph_score).toFixed(3)}</span>
                      )}
                      {item.rerank_score != null && (
                        <span>Rerank: {Number(item.rerank_score).toFixed(3)}</span>
                      )}
                    </div>
                  )}
                </div>
              </Card>
            ))
          )}
        </div>
      )}

      {/* Archives View */}
      {view === "archives" && (
        <div className="space-y-2">
          {archivesQuery.isLoading ? (
            <div className="flex items-center justify-center py-16 text-xs text-zinc-400">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              加载归档中...
            </div>
          ) : (archivesQuery.data?.items || []).length === 0 ? (
            <Card className="py-16 text-center text-sm text-zinc-400">暂无会话归档</Card>
          ) : (
            (archivesQuery.data?.items || []).map((arc: ArchiveItem) => (
              <Card
                key={arc.id}
                className="p-4 transition-colors hover:border-zinc-300 hover:bg-zinc-50/50 dark:hover:border-zinc-700 dark:hover:bg-zinc-900 cursor-pointer"
              >
                <div onClick={() => onOpenDrawer({ kind: "archive", item: arc })}>
                  <div className="flex items-center gap-2">
                    <ArchiveIcon className="h-4 w-4 shrink-0 text-zinc-400" />
                    <h3 className="truncate text-sm font-semibold">{arc.title}</h3>
                  </div>
                  {(arc.overview || arc.value_summary) && (
                    <p className="mt-1 line-clamp-2 text-xs text-zinc-500">
                      {arc.overview || arc.value_summary}
                    </p>
                  )}
                  <div className="mt-2.5 flex items-center justify-between text-[11px] text-zinc-400">
                    <div className="flex items-center gap-1.5">
                      <Badge tone="neutral">{arc.project}</Badge>
                      {arc.scope && <Badge tone="blue">{arc.scope}</Badge>}
                    </div>
                    <span>
                      {arc.created_by} · {formatRelativeTime(arc.updated_at)}
                    </span>
                  </div>
                </div>
              </Card>
            ))
          )}
        </div>
      )}

      {/* Personal Memory View */}
      {view === "personal" && (
        <div className="space-y-2">
          {personalQuery.isLoading ? (
            <div className="flex items-center justify-center py-16 text-xs text-zinc-400">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              加载个人画像中...
            </div>
          ) : (personalQuery.data?.items || []).length === 0 ? (
            <Card className="py-16 text-center text-sm text-zinc-400">
              暂无个人画像记录（点击右上角「新增画像」添加你的偏好或近期上下文）
            </Card>
          ) : (
            (personalQuery.data?.items || []).map((pm: PersonalMemoryItem) => (
              <Card
                key={pm.id}
                className="p-4 transition-colors hover:border-zinc-300 hover:bg-zinc-50/50 dark:hover:border-zinc-700 dark:hover:bg-zinc-900 cursor-pointer"
              >
                <div onClick={() => onOpenDrawer({ kind: "personal", item: pm })}>
                  <div className="flex items-center justify-between gap-2">
                    <Badge tone={pm.profile_kind === "static" ? "purple" : "blue"}>
                      {pm.profile_kind === "static" ? "Static 长期偏好" : "Dynamic 近期上下文"}
                    </Badge>
                    <span className="text-[11px] text-zinc-400">
                      {formatRelativeTime(pm.updated_at)}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-relaxed">{pm.content}</p>
                </div>
              </Card>
            ))
          )}
        </div>
      )}

      {/* Create Modal */}
      {createOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-xs"
          onClick={() => {
            if (!creating) setCreateOpen(false);
          }}
        >
          <div
            className="w-full max-w-lg rounded-2xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-800 dark:bg-zinc-900"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-base font-semibold">
              {view === "personal" ? "新增个人画像" : "新建团队经验"}
            </h3>
            <form onSubmit={handleCreateSubmit} className="mt-4 space-y-3">
              {view === "personal" ? (
                <>
                  <div className="flex gap-2">
                    {(["static", "dynamic"] as const).map((k) => (
                      <Button
                        key={k}
                        type="button"
                        size="sm"
                        variant={newPmKind === k ? "primary" : "secondary"}
                        onClick={() => setNewPmKind(k)}
                      >
                        {k === "static" ? "Static 长期偏好" : "Dynamic 近期上下文"}
                      </Button>
                    ))}
                  </div>
                  <Textarea
                    rows={5}
                    required
                    placeholder="例如：提交 Go 代码前总是先跑 golangci-lint；偏好中文简洁回复。"
                    value={newPmContent}
                    onChange={(e) => setNewPmContent(e.target.value)}
                  />
                </>
              ) : (
                <>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">标题</label>
                    <Input
                      required
                      placeholder="简明描述这条经验的主题"
                      value={newTitle}
                      onChange={(e) => setNewTitle(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">
                      问题 / 背景
                    </label>
                    <Textarea
                      rows={4}
                      required
                      placeholder="发生了什么问题、报错信息或适用场景"
                      value={newProblem}
                      onChange={(e) => setNewProblem(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">
                      解决方案
                    </label>
                    <Textarea
                      rows={5}
                      placeholder="根因分析与标准修复步骤"
                      value={newSolution}
                      onChange={(e) => setNewSolution(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">
                      标签（逗号分隔）
                    </label>
                    <Input
                      placeholder="postgres, rls, migration"
                      value={newTags}
                      onChange={(e) => setNewTags(e.target.value)}
                    />
                  </div>
                </>
              )}
              {createErr && <p className="text-xs text-red-500">{createErr}</p>}
              <div className="flex justify-end gap-2 pt-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => setCreateOpen(false)}
                >
                  取消
                </Button>
                <Button type="submit" size="sm" disabled={creating}>
                  {creating ? "保存中（计算向量）..." : "保存"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
