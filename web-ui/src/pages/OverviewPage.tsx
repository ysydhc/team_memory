import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowRight,
  FileEdit,
  Loader2,
  Play,
  RefreshCw,
} from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import { Badge, Button, Card } from "@/components/ui/primitives";
import { TrendChart } from "@/components/TrendChart";

interface OverviewPageProps {
  onGoToDrafts: () => void;
}

interface SubsystemLamp {
  key: string;
  label: string;
  status: string;
  metric: string;
  reason: string;
}

export function OverviewPage({ onGoToDrafts }: OverviewPageProps) {
  const { currentTenant } = useAuth();
  const tid = currentTenant?.id || "default";

  const [days, setDays] = useState<30 | 180>(30);
  const [scale, setScale] = useState<"linear" | "sqrt">("linear");
  const [evalMessage, setEvalMessage] = useState<string>("");

  const overviewQuery = useQuery({
    queryKey: ["observability-overview", tid, days],
    queryFn: () => api.getObservabilityOverview(days),
  });

  const evalMutation = useMutation({
    mutationFn: () => api.runGoldenEval(),
    onSuccess: (res) => {
      const ev = (res?.eval || {}) as { run_id?: string; graph_lift?: number };
      const lift = Number(ev.graph_lift ?? 0);
      setEvalMessage(
        `评测完成 (${ev.run_id || "ok"}) · Graph Lift ${lift >= 0 ? "+" : ""}${lift.toFixed(4)}`,
      );
      void overviewQuery.refetch();
    },
    onError: (err) => {
      setEvalMessage(`评测失败：${err instanceof Error ? err.message : String(err)}`);
    },
  });

  const draftsQuery = useQuery({
    queryKey: ["drafts-count", tid],
    queryFn: () => api.listExperiences({ status: "draft", limit: 1 }),
  });

  const payload = (overviewQuery.data || {}) as Record<string, unknown>;
  const subs = (payload.subsystems || {}) as Record<string, Record<string, unknown>>;
  const alerts = (payload.alerts || []) as Array<{
    subsystem: string;
    level: string;
    message: string;
  }>;

  const ingestion = subs.ingestion || {};
  const maintenance = subs.maintenance || {};
  const kg = subs.knowledge_graph || {};
  const retrieval = subs.retrieval || {};
  const infra = subs.infra || {};

  interface EvalModeStats {
    recall_at_5?: number;
    mrr_at_5?: number;
    mrr?: number;
    ndcg_at_5?: number;
    p50_latency_ms?: number;
    p95_latency_ms?: number;
    avg_tokens?: number;
    by_bucket?: Record<string, { query_count?: number; recall_at_5?: number; mrr_at_5?: number }>;
  }

  const autodream = (maintenance.autodream || {}) as Record<string, unknown>;
  const latestEval = (retrieval.latest_eval || null) as {
    run_id?: string;
    timestamp?: string;
    golden_count?: number;
    graph_lift?: number | null;
    modes?: Record<string, EvalModeStats>;
  } | null;
  const evalHistory = ((retrieval.eval_history || []) as Array<{
    run_id?: string;
    timestamp?: string;
    graph_lift?: number;
    modes?: Record<string, EvalModeStats>;
  }>).map((item) => ({
    date: String(item.timestamp || item.run_id || "").slice(5, 16).replace("T", " "),
    recall_pct: Number(((item.modes?.full?.recall_at_5 ?? 0) * 100).toFixed(1)),
    mrr_pct: Number(((item.modes?.full?.mrr_at_5 ?? item.modes?.full?.mrr ?? 0) * 100).toFixed(1)),
  }));

  const lamps: SubsystemLamp[] = [
    {
      key: "ingestion",
      label: "写入与精炼",
      status: String(ingestion.status || "green"),
      metric: `${ingestion.avg_7d ?? 0} / 天`,
      reason: String(ingestion.reason || "近7日写入正常"),
    },
    {
      key: "maintenance",
      label: "后台维护",
      status: String(maintenance.status || "green"),
      metric: `产出 ${autodream.total_output ?? 0} 条`,
      reason: String(maintenance.reason || "AutoDream & Janitor"),
    },
    {
      key: "knowledge_graph",
      label: "知识图谱",
      status: String(kg.status || "green"),
      metric: `${kg.entities ?? 0} 点 / ${kg.relationships ?? 0} 边`,
      reason: String(kg.reason || "图谱健康"),
    },
    {
      key: "retrieval",
      label: "检索质量",
      status: String(retrieval.status || "green"),
      metric: `命中 ${Math.round(Number(retrieval.hit_rate_7d ?? 0) * 100)}%`,
      reason: String(retrieval.reason || "近7日检索正常"),
    },
    {
      key: "infra",
      label: "向量底座",
      status: String(infra.status || "green"),
      metric: `${(Number(infra.embedding_coverage ?? 1) * 100).toFixed(1)}%`,
      reason: String(infra.reason || "PostgreSQL & 向量完整"),
    },
  ];

  const writeSeries = (ingestion.daily_series || []) as Array<Record<string, unknown>>;
  const searchSeries = (retrieval.daily_series || []) as Array<Record<string, unknown>>;
  const draftCount = draftsQuery.data?.total ?? 0;

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">驾驶舱</h1>
          <p className="text-xs text-zinc-500">
            租户 <strong>{currentTenant?.name || "Default Workspace"}</strong> · 5 大子系统健康度与全链路趋势
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* 30d / 180d switch */}
          <div className="inline-flex rounded-lg bg-zinc-100 p-0.5 text-xs dark:bg-zinc-900">
            {([30, 180] as const).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDays(d)}
                className={cn(
                  "rounded-md px-2.5 py-1 font-medium transition-colors cursor-pointer",
                  days === d
                    ? "bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100"
                    : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200",
                )}
              >
                近 {d} 天
              </button>
            ))}
          </div>

          {/* Linear / Sqrt Y-axis scale switch */}
          <div className="inline-flex rounded-lg bg-zinc-100 p-0.5 text-xs dark:bg-zinc-900">
            {(
              [
                { id: "linear", label: "线性刻度" },
                { id: "sqrt", label: "平方根 √" },
              ] as const
            ).map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => setScale(s.id)}
                className={cn(
                  "rounded-md px-2.5 py-1 font-medium transition-colors cursor-pointer",
                  scale === s.id
                    ? "bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100"
                    : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200",
                )}
              >
                {s.label}
              </button>
            ))}
          </div>

          <Button
            size="sm"
            variant="secondary"
            onClick={() => void overviewQuery.refetch()}
            disabled={overviewQuery.isFetching}
          >
            <RefreshCw className={cn("h-3.5 w-3.5", overviewQuery.isFetching && "animate-spin")} />
            刷新
          </Button>

          <Button
            size="sm"
            variant="primary"
            onClick={() => {
              setEvalMessage("");
              evalMutation.mutate();
            }}
            disabled={evalMutation.isPending}
          >
            {evalMutation.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            {evalMutation.isPending ? "评测中..." : "运行评测"}
          </Button>
        </div>
      </div>

      {evalMessage && (
        <div className="rounded-lg border border-zinc-200 bg-white px-3.5 py-2 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200">
          {evalMessage}
        </div>
      )}

      {/* Pending Drafts Inbox Strip */}
      {draftCount > 0 && (
        <div className="flex items-center justify-between rounded-xl border border-amber-200/90 bg-amber-50/70 px-4 py-3 dark:border-amber-900/60 dark:bg-amber-950/30">
          <div className="flex items-center gap-2.5 text-sm">
            <FileEdit className="h-4 w-4 text-amber-600 dark:text-amber-400" />
            <span>
              当前有 <strong>{draftCount}</strong> 条待审核草稿等待发布或丢弃
            </span>
          </div>
          <Button size="sm" variant="secondary" onClick={onGoToDrafts}>
            前往处理
            <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}

      {/* Active Alerts Strip */}
      {alerts.length > 0 && (
        <div className="space-y-1.5">
          {alerts.map((a, idx) => (
            <div
              key={idx}
              className={cn(
                "flex items-center gap-2 rounded-lg border px-3.5 py-2 text-xs",
                a.level === "red"
                  ? "border-red-200 bg-red-50/70 text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300"
                  : "border-amber-200 bg-amber-50/70 text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300",
              )}
            >
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              <span className="font-medium uppercase">{a.subsystem}</span>
              <span>{a.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* 5 Subsystem Health Lamp Cards */}
      {overviewQuery.isLoading ? (
        <div className="flex items-center justify-center py-16 text-xs text-zinc-400">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          加载 5 大子系统健康指标...
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {lamps.map((lamp) => (
              <Card key={lamp.key} className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-zinc-500">{lamp.label}</span>
                  <span
                    title={lamp.status}
                    className={cn(
                      "h-2.5 w-2.5 rounded-full",
                      lamp.status === "green" && "bg-emerald-500",
                      lamp.status === "yellow" && "bg-amber-500",
                      lamp.status === "red" && "bg-red-500",
                    )}
                  />
                </div>
                <div className="mt-2 text-lg font-semibold tracking-tight">{lamp.metric}</div>
                <p className="mt-1 line-clamp-2 text-[11px] text-zinc-500">{lamp.reason}</p>
              </Card>
            ))}
          </div>

          {/* 30d / 180d Trend Charts */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <TrendChart
              title={`每日写入趋势（近 ${days} 天）`}
              subtitle="总写入量与经验 / 个人画像 / 会话归档拆分"
              data={writeSeries}
              scale={scale}
              series={[
                { key: "total", label: "总写入", color: "#6366f1" },
                { key: "experiences", label: "经验", color: "#0ea5e9" },
                { key: "personal_memories", label: "画像", color: "#a855f7" },
                { key: "archives", label: "归档", color: "#10b981" },
              ]}
            />
            <TrendChart
              title={`检索延迟 (P50 / P95) 与搜索量（近 ${days} 天）`}
              subtitle={`均 Token: ${Math.round(Number(retrieval.avg_tokens_7d ?? 0))} · 7日图谱辅助占比: ${(Number(kg.graph_assist_rate_7d ?? 0) * 100).toFixed(1)}%`}
              data={searchSeries}
              scale={scale}
              series={[
                { key: "p50_ms", label: "P50 (ms)", color: "#10b981" },
                { key: "p95_ms", label: "P95 (ms)", color: "#f59e0b" },
                { key: "searches", label: "搜索量", color: "#0ea5e9" },
              ]}
            />
          </div>

          {/* Golden Eval Summary + History Trend */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Card className="p-5 lg:col-span-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-sm font-semibold">Golden Eval 三组消融对比</h2>
                  <p className="text-xs text-zinc-500">
                    Vector vs Hybrid vs Full (Adaptive Hybrid+Graph)
                    {latestEval?.golden_count ? ` · ${latestEval.golden_count} 题` : ""}
                    {latestEval?.run_id ? ` · ${latestEval.run_id}` : ""}
                  </p>
                </div>
                {latestEval?.graph_lift != null && (
                  <Badge tone={latestEval.graph_lift >= 0 ? "green" : "red"}>
                    Graph Lift {latestEval.graph_lift >= 0 ? "+" : ""}
                    {latestEval.graph_lift.toFixed(4)}
                  </Badge>
                )}
              </div>

              {latestEval?.modes ? (
                <>
                  <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                    {(
                      [
                        { key: "vector_only", fallbackKey: "vector", label: "A: Vector Only" },
                        { key: "hybrid", fallbackKey: "hybrid", label: "B: Hybrid (Vec+FTS)" },
                        { key: "full", fallbackKey: "full", label: "C: Full (Hybrid+Graph)" },
                      ] as const
                    ).map((m) => {
                      const stats = latestEval.modes?.[m.key] || latestEval.modes?.[m.fallbackKey];
                      const mrrVal = stats?.mrr_at_5 ?? stats?.mrr;
                      return (
                        <div
                          key={m.key}
                          className="rounded-lg border border-zinc-200/70 bg-zinc-50/70 p-3.5 dark:border-zinc-800 dark:bg-zinc-900/50"
                        >
                          <div className="text-xs font-medium text-zinc-500">{m.label}</div>
                          <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1 font-mono text-sm">
                            <span>
                              R@5:{" "}
                              <strong>
                                {stats?.recall_at_5 != null ? stats.recall_at_5.toFixed(3) : "—"}
                              </strong>
                            </span>
                            <span className="text-xs text-zinc-500">
                              MRR: {mrrVal != null ? mrrVal.toFixed(3) : "—"}
                            </span>
                            <span className="text-xs text-zinc-500">
                              NDCG: {stats?.ndcg_at_5 != null ? stats.ndcg_at_5.toFixed(3) : "—"}
                            </span>
                          </div>
                          {stats?.p50_latency_ms != null && (
                            <div className="mt-1.5 font-mono text-[11px] text-zinc-400">
                              P50 {Math.round(stats.p50_latency_ms)}ms · P95{" "}
                              {Math.round(stats.p95_latency_ms ?? 0)}ms
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {latestEval.modes?.full?.by_bucket && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {Object.entries(latestEval.modes.full.by_bucket).map(([bName, bStats]) => (
                        <span
                          key={bName}
                          className="rounded-md border border-zinc-200/80 bg-zinc-50 px-2 py-1 font-mono text-[11px] text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                        >
                          {bName} (n={bStats.query_count ?? 0}): R@5{" "}
                          {((bStats.recall_at_5 ?? 0) * 100).toFixed(0)}% · MRR{" "}
                          {(bStats.mrr_at_5 ?? 0).toFixed(3)}
                        </span>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <p className="mt-3 text-xs text-zinc-400">
                  暂无 Golden Eval 记录，点击右上角「运行评测」即可生成首份三路消融基线。
                </p>
              )}
            </Card>

            <TrendChart
              title="Golden Eval 历史趋势 (%)"
              subtitle={`共 ${evalHistory.length} 次基线评测`}
              data={evalHistory}
              scale="linear"
              series={[
                { key: "recall_pct", label: "Recall@5 %", color: "#10b981" },
                { key: "mrr_pct", label: "MRR@5 %", color: "#6366f1" },
              ]}
            />
          </div>
        </>
      )}
    </div>
  );
}
