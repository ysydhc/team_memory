import { useEffect, useState } from "react";
import {
  Archive as ArchiveIcon,
  CheckCircle2,
  Clock,
  Edit3,
  Pin,
  RotateCcw,
  Save,
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
import { formatRelativeTime } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { Badge, Button, Input, Textarea } from "@/components/ui/primitives";
import { MarkdownView } from "@/components/MarkdownView";

export type DrawerTarget =
  | { kind: "experience"; item: ExperienceItem }
  | { kind: "archive"; item: ArchiveItem }
  | { kind: "personal"; item: PersonalMemoryItem }
  | null;

interface MemoryDrawerProps {
  target: DrawerTarget;
  onClose: () => void;
  onMutated: () => void;
  onSwitchTarget?: (target: DrawerTarget) => void;
}

export function MemoryDrawer({
  target,
  onClose,
  onMutated,
  onSwitchTarget,
}: MemoryDrawerProps) {
  const { canWrite } = useAuth();
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");

  // Experience detail & edit state
  const [expDetail, setExpDetail] = useState<ExperienceItem | null>(null);
  const [title, setTitle] = useState("");
  const [problem, setProblem] = useState("");
  const [solution, setSolution] = useState("");
  const [tagsText, setTagsText] = useState("");

  // Archive full detail state
  const [archiveDetail, setArchiveDetail] = useState<ArchiveItem | null>(null);

  // Personal memory edit state
  const [pmContent, setPmContent] = useState("");
  const [pmKind, setPmKind] = useState<"static" | "dynamic">("static");

  useEffect(() => {
    setEditing(false);
    setToast("");
    if (!target) {
      setExpDetail(null);
      setArchiveDetail(null);
      return;
    }
    if (target.kind === "experience") {
      setExpDetail(target.item);
      setTitle(target.item.title || "");
      setProblem(target.item.description || "");
      setSolution(target.item.solution || "");
      setTagsText((target.item.tags || []).join(", "));
      // Fetch fresh full record
      api
        .getExperience(target.item.id)
        .then((fresh) => {
          setExpDetail(fresh);
          setTitle(fresh.title || "");
          setProblem(fresh.description || "");
          setSolution(fresh.solution || "");
          setTagsText((fresh.tags || []).join(", "));
        })
        .catch(() => {});
    } else if (target.kind === "archive") {
      setArchiveDetail(target.item);
      api
        .getArchive(target.item.id, target.item.project)
        .then((fresh) => setArchiveDetail(fresh))
        .catch(() => {});
    } else if (target.kind === "personal") {
      setPmContent(target.item.content || "");
      setPmKind(target.item.profile_kind === "dynamic" ? "dynamic" : "static");
    }
  }, [target]);

  useEffect(() => {
    if (!target) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [target, onClose]);

  if (!target) return null;

  const notify = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  };

  // ---- Experience actions ----
  const handleSaveExp = async () => {
    if (!expDetail) return;
    setBusy(true);
    try {
      const updated = await api.updateExperience(expDetail.id, {
        title: title.trim(),
        problem: problem.trim(),
        solution: solution.trim() || undefined,
        tags: tagsText
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
      });
      setExpDetail(updated);
      setEditing(false);
      notify("已保存修改");
      onMutated();
    } catch (err) {
      notify(err instanceof Error ? err.message : "保存失败");
    } finally {
      setBusy(false);
    }
  };

  const handleTogglePin = async () => {
    if (!expDetail) return;
    setBusy(true);
    try {
      const res = await api.togglePinExperience(expDetail.id);
      setExpDetail({ ...expDetail, is_pinned: res.is_pinned });
      notify(res.is_pinned ? "已置顶保护 (Pin)" : "已取消 Pin");
      onMutated();
    } catch (err) {
      notify(err instanceof Error ? err.message : "操作失败");
    } finally {
      setBusy(false);
    }
  };

  const handlePublish = async () => {
    if (!expDetail) return;
    setBusy(true);
    try {
      const res = await api.changeExperienceStatus(expDetail.id, { status: "published" });
      setExpDetail(res.experience || { ...expDetail, status: "published", exp_status: "published" });
      notify("已发布");
      onMutated();
    } catch (err) {
      notify(err instanceof Error ? err.message : "发布失败");
    } finally {
      setBusy(false);
    }
  };

  const handleSupersede = async () => {
    if (!expDetail) return;
    setBusy(true);
    try {
      const res = await api.supersedeExperience(expDetail.id);
      setExpDetail(res.experience);
      notify("已标记为废弃 (Superseded)");
      onMutated();
    } catch (err) {
      notify(err instanceof Error ? err.message : "废弃失败");
    } finally {
      setBusy(false);
    }
  };

  const handleSoftDelete = async () => {
    if (!expDetail) return;
    setBusy(true);
    try {
      await api.deleteExperience(expDetail.id, false);
      setExpDetail({ ...expDetail, is_deleted: true });
      notify("已软删除");
      onMutated();
    } catch (err) {
      notify(err instanceof Error ? err.message : "删除失败");
    } finally {
      setBusy(false);
    }
  };

  const handleRestore = async () => {
    if (!expDetail) return;
    setBusy(true);
    try {
      await api.restoreExperience(expDetail.id);
      setExpDetail({ ...expDetail, is_deleted: false });
      notify("已恢复");
      onMutated();
    } catch (err) {
      notify(err instanceof Error ? err.message : "恢复失败");
    } finally {
      setBusy(false);
    }
  };

  // ---- Personal memory actions ----
  const handleSavePersonal = async () => {
    if (target.kind !== "personal") return;
    setBusy(true);
    try {
      await api.updatePersonalMemory(target.item.id, {
        content: pmContent.trim(),
        profile_kind: pmKind,
      });
      setEditing(false);
      notify("画像已更新");
      onMutated();
    } catch (err) {
      notify(err instanceof Error ? err.message : "更新失败");
    } finally {
      setBusy(false);
    }
  };

  const handleDeletePersonal = async () => {
    if (target.kind !== "personal") return;
    setBusy(true);
    try {
      await api.deletePersonalMemory(target.item.id);
      onMutated();
      onClose();
    } catch (err) {
      notify(err instanceof Error ? err.message : "删除失败");
    } finally {
      setBusy(false);
    }
  };

  const expStatus = expDetail?.status || expDetail?.exp_status || "published";

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/30 backdrop-blur-[1px]" onClick={onClose}>
      <div
        className="flex h-full w-full max-w-[560px] flex-col border-l border-zinc-200 bg-white shadow-2xl dark:border-zinc-800 dark:bg-zinc-950"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-200/80 px-5 py-3.5 dark:border-zinc-800">
          <div className="flex items-center gap-2 text-xs font-medium text-zinc-500">
            {target.kind === "experience" && <span>经验详情</span>}
            {target.kind === "archive" && (
              <>
                <ArchiveIcon className="h-3.5 w-3.5" />
                <span>会话归档详情</span>
              </>
            )}
            {target.kind === "personal" && (
              <>
                <UserIcon className="h-3.5 w-3.5" />
                <span>个人画像条目</span>
              </>
            )}
            {toast && (
              <Badge tone="green" className="ml-2">
                {toast}
              </Badge>
            )}
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-5">
          {target.kind === "experience" && expDetail && (
            <div className="space-y-5">
              {/* Action bar */}
              {canWrite && (
                <div className="flex flex-wrap items-center gap-2">
                  {!editing ? (
                    <Button size="sm" variant="secondary" onClick={() => setEditing(true)}>
                      <Edit3 className="h-3.5 w-3.5" />
                      编辑
                    </Button>
                  ) : (
                    <>
                      <Button size="sm" disabled={busy} onClick={() => void handleSaveExp()}>
                        <Save className="h-3.5 w-3.5" />
                        保存
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
                        取消
                      </Button>
                    </>
                  )}
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={busy}
                    onClick={() => void handleTogglePin()}
                  >
                    <Pin className="h-3.5 w-3.5" />
                    {expDetail.is_pinned ? "取消 Pin" : "Pin 保护"}
                  </Button>
                  {expStatus === "draft" && (
                    <Button size="sm" disabled={busy} onClick={() => void handlePublish()}>
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      发布
                    </Button>
                  )}
                  {!expDetail.valid_until && (
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={busy}
                      onClick={() => void handleSupersede()}
                    >
                      <Clock className="h-3.5 w-3.5" />
                      废弃 (Supersede)
                    </Button>
                  )}
                  {expDetail.is_deleted ? (
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={busy}
                      onClick={() => void handleRestore()}
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      恢复
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-red-600 hover:bg-red-50 hover:text-red-700 dark:hover:bg-red-950/40"
                      disabled={busy}
                      onClick={() => void handleSoftDelete()}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      软删
                    </Button>
                  )}
                </div>
              )}

              {/* Metadata */}
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge tone="neutral">{expDetail.project}</Badge>
                <Badge tone={expStatus === "draft" ? "yellow" : "green"}>{expStatus}</Badge>
                {expDetail.quality_tier && <Badge tone="blue">{expDetail.quality_tier}</Badge>}
                {expDetail.is_pinned && <Badge tone="purple">Pinned</Badge>}
                {expDetail.valid_until && <Badge tone="red">已废弃</Badge>}
                {expDetail.is_deleted && <Badge tone="red">已软删</Badge>}
              </div>

              {editing ? (
                <div className="space-y-4">
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">标题</label>
                    <Input value={title} onChange={(e) => setTitle(e.target.value)} />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">
                      问题 / 背景
                    </label>
                    <Textarea
                      rows={5}
                      value={problem}
                      onChange={(e) => setProblem(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">
                      解决方案
                    </label>
                    <Textarea
                      rows={8}
                      value={solution}
                      onChange={(e) => setSolution(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">
                      标签（逗号分隔）
                    </label>
                    <Input value={tagsText} onChange={(e) => setTagsText(e.target.value)} />
                  </div>
                </div>
              ) : (
                <>
                  <h2 className="text-lg font-semibold leading-snug">{expDetail.title}</h2>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-500">
                    <span>作者：{expDetail.created_by}</span>
                    <span>召回：{expDetail.recall_count ?? 0}</span>
                    <span>采纳：{expDetail.used_count ?? 0}</span>
                    <span>更新：{formatRelativeTime(expDetail.updated_at)}</span>
                  </div>

                  <section>
                    <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                      问题与上下文
                    </h3>
                    <div className="rounded-xl border border-zinc-200/70 bg-zinc-50/70 p-3.5 dark:border-zinc-800 dark:bg-zinc-900/60">
                      <MarkdownView content={expDetail.description} />
                    </div>
                  </section>

                  {expDetail.solution && (
                    <section>
                      <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                        解决方案
                      </h3>
                      <div className="rounded-xl border border-zinc-200/70 bg-zinc-50/70 p-3.5 dark:border-zinc-800 dark:bg-zinc-900/60">
                        <MarkdownView content={expDetail.solution} />
                      </div>
                    </section>
                  )}

                  {(expDetail.tags || []).length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {expDetail.tags.map((t) => (
                        <Badge key={t} tone="neutral">
                          #{t}
                        </Badge>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {target.kind === "archive" && archiveDetail && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge tone="neutral">{archiveDetail.project}</Badge>
                {archiveDetail.scope && <Badge tone="blue">{archiveDetail.scope}</Badge>}
              </div>
              <h2 className="text-lg font-semibold leading-snug">{archiveDetail.title}</h2>
              <div className="text-xs text-zinc-500">
                作者：{archiveDetail.created_by} · 更新：{formatRelativeTime(archiveDetail.updated_at)}
              </div>
              {archiveDetail.overview && (
                <div className="rounded-xl border border-zinc-200/70 bg-zinc-50/70 p-3.5 dark:border-zinc-800 dark:bg-zinc-900/60">
                  <MarkdownView content={archiveDetail.overview} />
                </div>
              )}
              {archiveDetail.solution_doc && (
                <section>
                  <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                    归档正文
                  </h3>
                  <div className="rounded-xl border border-zinc-200/70 bg-zinc-50/70 p-3.5 dark:border-zinc-800 dark:bg-zinc-900/60">
                    <MarkdownView content={archiveDetail.solution_doc} />
                  </div>
                </section>
              )}
              {(archiveDetail.linked_experiences || []).length > 0 && (
                <section>
                  <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                    关联经验 ({archiveDetail.linked_experiences!.length})
                  </h3>
                  <div className="space-y-1.5">
                    {archiveDetail.linked_experiences!.map((le) => (
                      <button
                        key={le.id}
                        type="button"
                        onClick={async () => {
                          if (!onSwitchTarget) return;
                          try {
                            const fullExp = await api.getExperience(le.id);
                            onSwitchTarget({ kind: "experience", item: fullExp });
                          } catch {
                            notify("加载关联经验失败");
                          }
                        }}
                        className="flex w-full items-center justify-between rounded-lg border border-zinc-200 px-3 py-2 text-left text-xs transition-colors hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900 cursor-pointer"
                      >
                        <span className="truncate font-medium">{le.title || le.id}</span>
                        <Badge tone="blue">查看经验</Badge>
                      </button>
                    ))}
                  </div>
                </section>
              )}
              {(archiveDetail.attachments || []).length > 0 && (
                <section>
                  <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                    附件
                  </h3>
                  <div className="space-y-1.5">
                    {archiveDetail.attachments!.map((att) => (
                      <a
                        key={att.id}
                        href={`/api/v1/archives/${archiveDetail.id}/attachments/${att.id}/file?project=${encodeURIComponent(archiveDetail.project)}`}
                        className="flex items-center justify-between rounded-lg border border-zinc-200 px-3 py-2 text-xs hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900"
                      >
                        <span className="truncate">{att.source_path || att.path || att.id}</span>
                        <Badge tone="neutral">{att.kind}</Badge>
                      </a>
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}

          {target.kind === "personal" && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                {!editing ? (
                  <Button size="sm" variant="secondary" onClick={() => setEditing(true)}>
                    <Edit3 className="h-3.5 w-3.5" />
                    编辑画像
                  </Button>
                ) : (
                  <>
                    <Button size="sm" disabled={busy} onClick={() => void handleSavePersonal()}>
                      <Save className="h-3.5 w-3.5" />
                      保存
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
                      取消
                    </Button>
                  </>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40"
                  disabled={busy}
                  onClick={() => void handleDeletePersonal()}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  删除
                </Button>
              </div>

              {editing ? (
                <div className="space-y-3">
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">画像类型</label>
                    <div className="flex gap-2">
                      {(["static", "dynamic"] as const).map((k) => (
                        <Button
                          key={k}
                          type="button"
                          size="sm"
                          variant={pmKind === k ? "primary" : "secondary"}
                          onClick={() => setPmKind(k)}
                        >
                          {k === "static" ? "Static 长期偏好" : "Dynamic 近期上下文"}
                        </Button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-zinc-500">内容</label>
                    <Textarea
                      rows={6}
                      value={pmContent}
                      onChange={(e) => setPmContent(e.target.value)}
                    />
                  </div>
                </div>
              ) : (
                <>
                  <Badge tone={target.item.profile_kind === "static" ? "purple" : "blue"}>
                    {target.item.profile_kind === "static" ? "Static 长期偏好" : "Dynamic 近期上下文"}
                  </Badge>
                  <div className="whitespace-pre-wrap rounded-xl border border-zinc-200/70 bg-zinc-50/70 p-4 text-sm leading-relaxed dark:border-zinc-800 dark:bg-zinc-900/60">
                    {target.item.content}
                  </div>
                  <div className="text-xs text-zinc-400">
                    仅你本人可见 · 更新于 {formatRelativeTime(target.item.updated_at)}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
