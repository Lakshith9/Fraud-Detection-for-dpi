import clsx from "clsx";
import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Empty, ErrorBox, KV, Kpi, Pill, Spinner } from "../components/ui";
import { Metrics } from "../lib/api";
import { useApi } from "../lib/hooks";
import { fmtMs, fmtPct, fmtTime } from "../lib/format";

interface LatPoint {
  t: number;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  e2e99: number | null;
}

export default function SystemHealth() {
  const m = useApi<Metrics>("/v1/metrics", 2000);
  const [hist, setHist] = useState<LatPoint[]>([]);
  useEffect(() => {
    if (!m.data) return;
    const d = m.data;
    setHist((h) => [...h, { t: d.time * 1000, p50: d.latency_ms.p50, p95: d.latency_ms.p95, p99: d.latency_ms.p99, e2e99: d.e2e_latency_ms.p99 }].slice(-150));
  }, [m.data]);

  if (m.loading && !m.data) return <Spinner label="Loading metrics…" />;
  if (m.error && !m.data) return <ErrorBox error={m.error} onRetry={m.reload} />;
  const d = m.data!;
  const p99 = d.latency_ms.p99;
  const tps = d.throughput.series.map((v, i) => ({ t: (d.time - 60 + i) * 1000, tps: v }));
  const degraded = d.degraded_rate > 0 || d.graph.stale || d.workers.some((w) => !w.alive || !w.db_healthy);
  return (
    <div className="space-y-4 p-6">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">System & Model Health</h1>
          <p className="text-sm text-slate-400">Refreshes every 2 s · latency = in-worker scoring (feature fetch → decision → state write)</p>
        </div>
        {degraded ? <Pill tone="amber">degraded components present</Pill> : <Pill tone="emerald">all components healthy</Pill>}
      </div>
      {m.error && <ErrorBox error={`metrics refresh failed: ${m.error}`} />}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
        <Kpi label="Throughput" value={`${d.throughput.tps_last_10s}`} sub={`TPS (10 s) · ${d.throughput.tps_last_60s} avg 60 s`} />
        <Kpi label="p50 latency" value={fmtMs(d.latency_ms.p50)} sub={`n=${d.latency_ms.n}`} />
        <Kpi label="p95 latency" value={fmtMs(d.latency_ms.p95)} />
        <Kpi label="p99 latency" value={fmtMs(p99)} tone={p99 == null ? undefined : p99 < 100 ? "ok" : "bad"} sub="budget 100 ms" />
        <Kpi label="E2E p99" value={fmtMs(d.e2e_latency_ms.p99)} sub="publish → scored (incl. queue)" />
        <Kpi label="Alert rate" value={fmtPct(d.alert_rate, 2)} sub={`block ${fmtPct(d.block_rate, 2)}`} />
        <Kpi label="Degraded" value={fmtPct(d.degraded_rate, 2)} tone={d.degraded_rate > 0.01 ? "warn" : "ok"} />
        <Kpi label="Model" value={d.model.version} sub={`${(d.counts.total ?? 0).toLocaleString()} scored`} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="card">
          <div className="card-h">Throughput (scored tx / second, last 60 s)</div>
          <div className="h-56 p-2">
            <ResponsiveContainer>
              <AreaChart data={tps}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="t" tickFormatter={fmtTime} stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} labelFormatter={(v) => fmtTime(Number(v))} />
                <Area type="monotone" dataKey="tps" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.15} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card">
          <div className="card-h">Latency percentiles (rolling window, ms)</div>
          <div className="h-56 p-2">
            {hist.length < 2 ? (
              <Empty title="Collecting…" />
            ) : (
              <ResponsiveContainer>
                <LineChart data={hist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="t" tickFormatter={fmtTime} stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={10} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} labelFormatter={(v) => fmtTime(Number(v))} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="p50" stroke="#10b981" dot={false} isAnimationActive={false} />
                  <Line dataKey="p95" stroke="#f59e0b" dot={false} isAnimationActive={false} />
                  <Line dataKey="p99" stroke="#f43f5e" dot={false} isAnimationActive={false} />
                  <Line dataKey="e2e99" name="e2e p99" stroke="#a78bfa" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="card">
          <div className="card-h">Scoring workers</div>
          {!d.workers.length ? (
            <Empty title="No workers" />
          ) : (
            <table className="w-full">
              <thead>
                <tr>
                  <th className="th">Part.</th>
                  <th className="th">State</th>
                  <th className="th">Processed</th>
                  <th className="th">Pending</th>
                  <th className="th">DB</th>
                </tr>
              </thead>
              <tbody>
                {d.workers.map((w) => (
                  <tr key={w.partition} className="border-t border-slate-800/60">
                    <td className="td">{w.partition}</td>
                    <td className="td">{w.alive ? <Pill tone="emerald">alive</Pill> : <Pill tone="rose">down</Pill>}</td>
                    <td className="td tabular-nums">{w.processed.toLocaleString()}</td>
                    <td className="td tabular-nums">{w.pending}</td>
                    <td className="td">{w.db_healthy ? <Pill tone="emerald">ok · {w.db_backlog}</Pill> : <Pill tone="amber">buffering</Pill>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="border-t border-slate-800 px-4 py-2">
            <div className="mb-1 text-xs uppercase text-slate-500">Graph worker</div>
            <KV k="Last run" v={d.graph.last_run_age_s != null ? `${d.graph.last_run_age_s.toFixed(0)} s ago` : "never"} />
            <KV k="Status" v={d.graph.stale ? <Pill tone="amber">stale → non-graph combo</Pill> : <Pill tone="emerald">fresh</Pill>} />
            <KV k="Rings / scored nodes" v={`${d.graph.rings} / ${d.graph.nodes}`} />
            <KV k="Window rows / run time" v={`${d.graph.window_rows.toLocaleString()} / ${d.graph.run_ms.toFixed(0)} ms`} />
            <div className="mb-1 mt-3 text-xs uppercase text-slate-500">Persister (Redis stream → PostgreSQL)</div>
            <KV k="Status" v={d.persister?.alive ? <Pill tone="emerald">alive</Pill> : <Pill tone="rose">down</Pill>} />
            <KV k="Lag (rows)" v={d.persister?.lag ?? "—"} />
            <KV k="Rows written / async SHAP" v={`${(d.persister?.written ?? 0).toLocaleString()} / ${(d.persister?.explained_async ?? 0).toLocaleString()}`} />
          </div>
        </div>
        <div className="card">
          <div className="card-h">
            Feature drift (PSI vs training) <span className="text-xs font-normal text-slate-500">last {d.drift.n} scored</span>
          </div>
          {!Object.keys(d.drift.features).length ? (
            <Empty title="No scored traffic yet" />
          ) : (
            <div className="px-4 py-2">
              {Object.entries(d.drift.features)
                .sort((a, b) => b[1].psi - a[1].psi)
                .map(([k, v]) => (
                  <div key={k} className="flex items-center gap-2 border-b border-slate-800/60 py-1.5 text-sm last:border-0">
                    <span className="flex-1 text-slate-300">{k}</span>
                    <div className="h-1.5 w-24 overflow-hidden rounded bg-slate-800">
                      <div
                        className={clsx("h-full", v.status === "stable" ? "bg-emerald-500" : v.status === "moderate" ? "bg-amber-400" : "bg-rose-500")}
                        style={{ width: `${Math.min(100, (v.psi / 0.3) * 100)}%` }}
                      />
                    </div>
                    <span className="w-14 text-right font-mono text-xs">{v.psi.toFixed(3)}</span>
                  </div>
                ))}
              <div className="pt-2 text-xs text-slate-500">PSI &lt; 0.1 stable · 0.1–0.25 moderate · &gt; 0.25 shift</div>
            </div>
          )}
        </div>
        <div className="card">
          <div className="card-h">Decisions & feedback</div>
          <div className="px-4 py-2">
            {["ALLOW", "REVIEW", "BLOCK", "degraded", "total"].map((k) => (
              <KV key={k} k={k} v={(d.counts[k] ?? 0).toLocaleString()} />
            ))}
            <div className="mt-3 mb-1 text-xs uppercase text-slate-500">Operator feedback</div>
            {Object.keys(d.feedback).length ? Object.entries(d.feedback).map(([k, v]) => <KV key={k} k={k.replace("_", " ")} v={v} />) : <div className="text-xs text-slate-500">none yet</div>}
            <div className="mt-3 mb-1 text-xs uppercase text-slate-500">Thresholds ({d.model.version})</div>
            {Object.entries(d.model.thresholds).map(([k, v]) => (
              <KV key={k} k={k} v={`review ≥ ${v.review.toFixed(4)} · block ≥ ${v.block.toFixed(4)}`} mono />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
