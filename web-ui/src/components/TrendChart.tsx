import { useMemo, useState } from "react";

export interface SeriesConfig {
  key: string;
  label: string;
  color: string;
}

interface TrendChartProps {
  title: string;
  subtitle?: string;
  data: Array<Record<string, unknown>>;
  xKey?: string;
  series: SeriesConfig[];
  scale: "linear" | "sqrt";
  unit?: string;
}

export function TrendChart({
  title,
  subtitle,
  data,
  xKey = "date",
  series,
  scale,
  unit = "",
}: TrendChartProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const width = 640;
  const height = 210;
  const pad = { top: 16, right: 16, bottom: 28, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const transformY = (v: number) => (scale === "sqrt" ? Math.sqrt(Math.max(0, v)) : Math.max(0, v));

  const { maxRaw, maxTransformed, yTicks } = useMemo(() => {
    let m = 0;
    for (const row of data) {
      for (const s of series) {
        const val = Number(row[s.key] ?? 0);
        if (Number.isFinite(val) && val > m) m = val;
      }
    }
    const safeMax = m > 0 ? m : 4;
    const maxT = transformY(safeMax);
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
      const tVal = maxT * ratio;
      const rawVal = scale === "sqrt" ? tVal * tVal : tVal;
      return {
        y: pad.top + plotH - ratio * plotH,
        label: rawVal >= 100 ? Math.round(rawVal).toString() : rawVal.toFixed(rawVal >= 10 ? 0 : 1),
      };
    });
    return { maxRaw: safeMax, maxTransformed: maxT || 1, yTicks: ticks };
  }, [data, series, scale, plotH, pad.top]);

  const xPos = (idx: number) => {
    if (data.length <= 1) return pad.left + plotW / 2;
    return pad.left + (idx / (data.length - 1)) * plotW;
  };

  const yPos = (raw: number) => {
    const t = transformY(raw);
    const ratio = Math.min(1, Math.max(0, t / maxTransformed));
    return pad.top + plotH - ratio * plotH;
  };

  const labelStep = Math.max(1, Math.ceil(data.length / 6));
  const hoveredRow = hoverIdx !== null ? data[hoverIdx] : null;

  return (
    <div className="rounded-xl border border-zinc-200/80 bg-white p-4 dark:border-zinc-800/80 dark:bg-zinc-900/60">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          {subtitle && <p className="text-[11px] text-zinc-400">{subtitle}</p>}
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
          {series.map((s) => (
            <span key={s.key} className="inline-flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              {s.label}
            </span>
          ))}
        </div>
      </div>

      {data.length === 0 ? (
        <div className="flex h-44 items-center justify-center text-xs text-zinc-400">
          暂无时间序列数据
        </div>
      ) : (
        <div className="relative">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="h-52 w-full overflow-visible"
            onMouseLeave={() => setHoverIdx(null)}
          >
            {/* Horizontal grid lines + Y labels */}
            {yTicks.map((t, i) => (
              <g key={i}>
                <line
                  x1={pad.left}
                  y1={t.y}
                  x2={width - pad.right}
                  y2={t.y}
                  stroke="currentColor"
                  className="text-zinc-200/80 dark:text-zinc-800"
                  strokeDasharray={i === 0 ? undefined : "3 3"}
                />
                <text
                  x={pad.left - 8}
                  y={t.y + 3}
                  textAnchor="end"
                  className="fill-zinc-400 text-[10px]"
                >
                  {t.label}
                </text>
              </g>
            ))}

            {/* X-axis date labels */}
            {data.map((row, idx) => {
              if (idx % labelStep !== 0 && idx !== data.length - 1) return null;
              const rawDate = String(row[xKey] || "");
              const shortDate = rawDate.length >= 10 ? rawDate.slice(5, 10) : rawDate;
              return (
                <text
                  key={idx}
                  x={xPos(idx)}
                  y={height - 6}
                  textAnchor="middle"
                  className="fill-zinc-400 text-[10px]"
                >
                  {shortDate}
                </text>
              );
            })}

            {/* Lines */}
            {series.map((s) => {
              const pts = data
                .map((row, idx) => `${xPos(idx).toFixed(1)},${yPos(Number(row[s.key] ?? 0)).toFixed(1)}`)
                .join(" ");
              return (
                <polyline
                  key={s.key}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  points={pts}
                />
              );
            })}

            {/* Hover vertical guide + dots */}
            {hoverIdx !== null && hoveredRow && (
              <g>
                <line
                  x1={xPos(hoverIdx)}
                  y1={pad.top}
                  x2={xPos(hoverIdx)}
                  y2={pad.top + plotH}
                  stroke="currentColor"
                  className="text-zinc-400/60"
                  strokeDasharray="2 2"
                />
                {series.map((s) => (
                  <circle
                    key={s.key}
                    cx={xPos(hoverIdx)}
                    cy={yPos(Number(hoveredRow[s.key] ?? 0))}
                    r={3.5}
                    fill={s.color}
                    stroke="#fff"
                    strokeWidth={1.5}
                  />
                ))}
              </g>
            )}

            {/* Invisible hover hit bands */}
            {data.map((_, idx) => {
              const bandW = plotW / Math.max(1, data.length);
              return (
                <rect
                  key={idx}
                  x={xPos(idx) - bandW / 2}
                  y={pad.top}
                  width={Math.max(6, bandW)}
                  height={plotH}
                  fill="transparent"
                  onMouseEnter={() => setHoverIdx(idx)}
                />
              );
            })}
          </svg>

          {/* Floating tooltip */}
          {hoverIdx !== null && hoveredRow && (
            <div className="pointer-events-none absolute right-2 top-1 rounded-lg border border-zinc-200 bg-white/95 px-2.5 py-1.5 text-[11px] shadow-md backdrop-blur dark:border-zinc-700 dark:bg-zinc-900/95">
              <div className="font-mono font-medium text-zinc-500">
                {String(hoveredRow[xKey] || "")}
              </div>
              <div className="mt-1 space-y-0.5">
                {series.map((s) => (
                  <div key={s.key} className="flex items-center justify-between gap-3">
                    <span className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300">
                      <span
                        className="h-1.5 w-1.5 rounded-full"
                        style={{ backgroundColor: s.color }}
                      />
                      {s.label}
                    </span>
                    <span className="font-mono font-semibold">
                      {Number(hoveredRow[s.key] ?? 0)}
                      {unit}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scale hint */}
          <div className="mt-1 flex justify-end text-[10px] text-zinc-400">
            峰值 {maxRaw}
            {unit} · {scale === "sqrt" ? "平方根刻度 (√Y)" : "线性刻度 (Y)"}
          </div>
        </div>
      )}
    </div>
  );
}
