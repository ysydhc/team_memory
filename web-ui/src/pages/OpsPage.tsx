import React, { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FlaskConical, Loader2, Play, Save, Settings2, Sliders } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { Badge, Button, Card, Input } from "@/components/ui/primitives";

export function OpsPage() {
  const { currentTenant, refreshMe } = useAuth();
  const tid = currentTenant?.id || "default";

  // Janitor trigger
  const [runningJanitor, setRunningJanitor] = useState(false);
  const [janitorMsg, setJanitorMsg] = useState("");

  // Golden Eval trigger
  const [runningEval, setRunningEval] = useState(false);
  const [evalMsg, setEvalMsg] = useState("");

  // Tenant settings
  const [tenantName, setTenantName] = useState(currentTenant?.name || "");
  const [savingTenant, setSavingTenant] = useState(false);
  const [tenantMsg, setTenantMsg] = useState("");

  // Retrieval config
  const [maxCount, setMaxCount] = useState("10");
  const [minSim, setMinSim] = useState("0.5");
  const [rrfK, setRrfK] = useState("60");
  const [savingCfg, setSavingCfg] = useState(false);
  const [cfgMsg, setCfgMsg] = useState("");

  useEffect(() => {
    setTenantName(currentTenant?.name || "");
  }, [currentTenant?.name]);

  const overviewQuery = useQuery({
    queryKey: ["ops-overview", tid],
    queryFn: () => api.getObservabilityOverview(30),
  });

  const janitorQuery = useQuery({
    queryKey: ["janitor-status", tid],
    queryFn: () => api.getJanitorStatus(),
  });

  const retrievalCfgQuery = useQuery({
    queryKey: ["config-retrieval", tid],
    queryFn: () => api.getRetrievalConfig(),
  });

  const searchCfgQuery = useQuery({
    queryKey: ["config-search", tid],
    queryFn: () => api.getSearchConfig(),
  });

  useEffect(() => {
    if (retrievalCfgQuery.data) {
      if (retrievalCfgQuery.data.max_count != null) {
        setMaxCount(String(retrievalCfgQuery.data.max_count));
      }
      if (retrievalCfgQuery.data.min_similarity != null) {
        setMinSim(String(retrievalCfgQuery.data.min_similarity));
      }
    }
  }, [retrievalCfgQuery.data]);

  useEffect(() => {
    if (searchCfgQuery.data?.rrf_k != null) {
      setRrfK(String(searchCfgQuery.data.rrf_k));
    }
  }, [searchCfgQuery.data]);

  const handleRunJanitor = async () => {
    setRunningJanitor(true);
    setJanitorMsg("");
    try {
      await api.runJanitor();
      setJanitorMsg("Janitor 清理任务执行完成");
      await Promise.all([janitorQuery.refetch(), overviewQuery.refetch()]);
    } catch (err) {
      setJanitorMsg(err instanceof Error ? err.message : "触发失败");
    } finally {
      setRunningJanitor(false);
    }
  };

  const handleRunEval = async () => {
    setRunningEval(true);
    setEvalMsg("");
    try {
      await api.runGoldenEval(20);
      setEvalMsg("Golden Eval 三路消融评测已完成并写入历史");
      await overviewQuery.refetch();
    } catch (err) {
      setEvalMsg(err instanceof Error ? err.message : "评测失败");
    } finally {
      setRunningEval(false);
    }
  };

  const handleSaveTenant = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tenantName.trim()) return;
    setSavingTenant(true);
    setTenantMsg("");
    try {
      await api.updateCurrentTenant({ name: tenantName.trim() });
      await refreshMe();
      setTenantMsg("租户配置已保存");
    } catch (err) {
      setTenantMsg(err instanceof Error ? err.message : "保存失败");
    } finally {
      setSavingTenant(false);
    }
  };

  const handleSaveRetrieval = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingCfg(true);
    setCfgMsg("");
    try {
      await api.putRetrievalConfig({
        max_count: Number(maxCount),
        min_similarity: Number(minSim),
      });
      await api.putSearchConfig({ rrf_k: Number(rrfK) });
      setCfgMsg("检索与 RRF 参数已热更新并清空缓存");
    } catch (err) {
      setCfgMsg(err instanceof Error ? err.message : "更新失败");
    } finally {
      setSavingCfg(false);
    }
  };

  const subs = ((overviewQuery.data as Record<string, unknown>)?.subsystems || {}) as Record<
    string,
    Record<string, unknown>
  >;
  const maint = subs.maintenance || {};
  const dream = (maint.autodream || {}) as Record<string, unknown>;
  const evalHistory = ((subs.retrieval?.eval_history || []) as Array<{
    timestamp?: string;
    graph_lift?: number | null;
    modes?: Record<string, { recall_at_5?: number; mrr?: number }>;
  }>).slice(-5);

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">系统运维</h1>
          <p className="text-xs text-zinc-500">
            仅 Tenant Admin 可见 · AutoDream / Janitor / Golden Eval 消融 / 租户与检索策略配置
          </p>
        </div>
        <Badge tone="neutral">Admin Only</Badge>
      </div>

      {/* 1. AutoDream & Janitor */}
      <Card className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-zinc-500" />
            <h2 className="text-sm font-semibold">AutoDream & Janitor 后台维护</h2>
          </div>
          <Button size="sm" disabled={runningJanitor} onClick={() => void handleRunJanitor()}>
            {runningJanitor ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            手动触发 Janitor
          </Button>
        </div>
        {janitorMsg && <p className="mt-2 text-xs text-emerald-600">{janitorMsg}</p>}

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
            <div className="text-[11px] text-zinc-400">AutoDream 合并</div>
            <div className="mt-1 font-mono text-base font-semibold">{String(dream.merged ?? 0)}</div>
          </div>
          <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
            <div className="text-[11px] text-zinc-400">双时态废弃</div>
            <div className="mt-1 font-mono text-base font-semibold">
              {String(dream.superseded ?? 0)}
            </div>
          </div>
          <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
            <div className="text-[11px] text-zinc-400">低分清理</div>
            <div className="mt-1 font-mono text-base font-semibold">{String(dream.pruned ?? 0)}</div>
          </div>
          <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
            <div className="text-[11px] text-zinc-400">Janitor 调度器</div>
            <div className="mt-1 text-sm font-semibold">
              {janitorQuery.data?.scheduler_running ? "运行中" : "空闲/未启用"}
            </div>
          </div>
        </div>
      </Card>

      {/* 2. Golden Eval Ablation */}
      <Card className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4 text-zinc-500" />
            <h2 className="text-sm font-semibold">Golden Eval 消融回归（Vector vs Hybrid vs Full）</h2>
          </div>
          <Button size="sm" variant="secondary" disabled={runningEval} onClick={() => void handleRunEval()}>
            {runningEval ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            运行消融评测
          </Button>
        </div>
        {evalMsg && <p className="mt-2 text-xs text-emerald-600">{evalMsg}</p>}

        {evalHistory.length === 0 ? (
          <p className="mt-3 text-xs text-zinc-400">暂无历史评测记录</p>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-zinc-200/80 text-zinc-400 dark:border-zinc-800">
                  <th className="py-2 font-medium">时间</th>
                  <th className="py-2 font-medium">Vector R@5</th>
                  <th className="py-2 font-medium">Hybrid R@5</th>
                  <th className="py-2 font-medium">Full R@5</th>
                  <th className="py-2 font-medium">Graph Lift</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 font-mono dark:divide-zinc-800/70">
                {evalHistory.map((row, idx) => (
                  <tr key={idx}>
                    <td className="py-2 text-zinc-500">
                      {row.timestamp ? row.timestamp.slice(0, 16).replace("T", " ") : "—"}
                    </td>
                    <td className="py-2">
                      {row.modes?.vector?.recall_at_5?.toFixed(3) ?? "—"}
                    </td>
                    <td className="py-2">
                      {row.modes?.hybrid?.recall_at_5?.toFixed(3) ?? "—"}
                    </td>
                    <td className="py-2 font-semibold">
                      {row.modes?.full?.recall_at_5?.toFixed(3) ?? "—"}
                    </td>
                    <td className="py-2">
                      {row.graph_lift != null
                        ? `${row.graph_lift >= 0 ? "+" : ""}${row.graph_lift.toFixed(4)}`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* 3. Tenant & Retrieval Tuning */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card className="p-5">
          <h2 className="text-sm font-semibold">租户工作空间配置</h2>
          <form onSubmit={handleSaveTenant} className="mt-4 space-y-3">
            <div>
              <label className="mb-1 block text-xs font-medium text-zinc-500">租户名称</label>
              <Input value={tenantName} onChange={(e) => setTenantName(e.target.value)} />
            </div>
            {tenantMsg && <p className="text-xs text-emerald-600">{tenantMsg}</p>}
            <Button type="submit" size="sm" disabled={savingTenant}>
              <Save className="h-3.5 w-3.5" />
              保存租户信息
            </Button>
          </form>
        </Card>

        <Card className="p-5">
          <div className="flex items-center gap-2">
            <Sliders className="h-4 w-4 text-zinc-500" />
            <h2 className="text-sm font-semibold">检索高级参数（热更新）</h2>
          </div>
          <form onSubmit={handleSaveRetrieval} className="mt-4 space-y-3">
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="mb-1 block text-xs text-zinc-500">Top-K (max_count)</label>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={maxCount}
                  onChange={(e) => setMaxCount(e.target.value)}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-zinc-500">最低相似度</label>
                <Input
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={minSim}
                  onChange={(e) => setMinSim(e.target.value)}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-zinc-500">RRF k</label>
                <Input
                  type="number"
                  min={1}
                  max={200}
                  value={rrfK}
                  onChange={(e) => setRrfK(e.target.value)}
                />
              </div>
            </div>
            {cfgMsg && <p className="text-xs text-emerald-600">{cfgMsg}</p>}
            <Button type="submit" size="sm" disabled={savingCfg}>
              <Save className="h-3.5 w-3.5" />
              应用检索参数
            </Button>
          </form>
        </Card>
      </div>
    </div>
  );
}
