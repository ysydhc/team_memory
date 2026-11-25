import { useEffect, useState } from "react";
import { CornerDownLeft, Loader2, Pin, Search, X } from "lucide-react";
import { api, type ExperienceItem } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/primitives";

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onSelectExperience: (exp: ExperienceItem) => void;
}

export function CommandPalette({ open, onClose, onSelectExperience }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ExperienceItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(0);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setResults([]);
      setSelectedIdx(0);
      return;
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const trimmed = query.trim();
    if (!trimmed) {
      setResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await api.search({ query: trimmed, max_results: 8 });
        setResults(res.results || []);
        setSelectedIdx(0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 220);
    return () => clearTimeout(timer);
  }, [query, open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIdx((i) => Math.min(i + 1, Math.max(results.length - 1, 0)));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIdx((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter" && results[selectedIdx]) {
        e.preventDefault();
        onSelectExperience(results[selectedIdx]);
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, results, selectedIdx, onClose, onSelectExperience]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/45 pt-[12vh] px-4 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-2xl dark:border-zinc-800 dark:bg-zinc-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-zinc-200/80 px-4 py-3 dark:border-zinc-800">
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
          ) : (
            <Search className="h-4 w-4 text-zinc-400" />
          )}
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入关键词或问题描述，按回车打开详情抽屉..."
            className="flex-1 bg-transparent text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none dark:text-zinc-100"
          />
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800 cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="max-h-96 overflow-y-auto p-2">
          {!query.trim() ? (
            <div className="px-3 py-8 text-center text-xs text-zinc-400">
              当前租户实时语义 + 混合召回检索 · 支持 ↑↓ 选择、Enter 打开
            </div>
          ) : results.length === 0 && !loading ? (
            <div className="px-3 py-8 text-center text-xs text-zinc-400">
              未找到匹配记忆
            </div>
          ) : (
            <div className="space-y-1">
              {results.map((item, idx) => (
                <button
                  key={item.id}
                  type="button"
                  onMouseEnter={() => setSelectedIdx(idx)}
                  onClick={() => {
                    onSelectExperience(item);
                    onClose();
                  }}
                  className={cn(
                    "flex w-full items-start justify-between gap-3 rounded-xl px-3 py-2.5 text-left transition-colors cursor-pointer",
                    idx === selectedIdx
                      ? "bg-zinc-100 dark:bg-zinc-800"
                      : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50",
                  )}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 text-sm font-medium">
                      {item.is_pinned && <Pin className="h-3 w-3 shrink-0 text-amber-500" />}
                      <span className="truncate">{item.title}</span>
                    </div>
                    <p className="mt-0.5 line-clamp-1 text-xs text-zinc-500">
                      {item.description}
                    </p>
                    <div className="mt-1.5 flex items-center gap-1.5">
                      <Badge tone="neutral">{item.project}</Badge>
                      {item.quality_tier && <Badge tone="blue">{item.quality_tier}</Badge>}
                      {(item.tags || []).slice(0, 3).map((t) => (
                        <span key={t} className="text-[11px] text-zinc-400">
                          #{t}
                        </span>
                      ))}
                    </div>
                  </div>
                  {idx === selectedIdx && (
                    <CornerDownLeft className="mt-1 h-3.5 w-3.5 shrink-0 text-zinc-400" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
