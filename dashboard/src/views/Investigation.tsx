import clsx from "clsx";
import { Briefcase, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { FeedbackButtons } from "../components/FeedbackButtons";
import { DecisionBadge, Empty, ErrorBox, KV, Pill, Spinner } from "../components/ui";
import { Explanation, HistRow, TxDetail } from "../lib/api";
import { useApi } from "../lib/hooks";
import { fmtAmount, fmtMs, fmtTs, short, typeLabel } from "../lib/format";
import { CreateCaseModal } from "./LiveAlerts";

export default function Investigation() {
  const { id } = useParams();
  const nav = useNavigate();
  const [q, setQ] = useState(id ?? "");
  return (
    <div className="p-6">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Transaction Investigation</h1>
          <p className="text-sm text-slate-400">Risk score, SHAP attribution, history and linked entities</p>
        </div>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (q.trim()) nav(`/investigate/${q.trim()}`);
          }}
        >
          <input className="input w-80 font-mono text-xs" placeholder="transfer id" value={q} onChange={(e) => setQ(e.target.value)} />
          <button className="btn-primary">
            <Search className="h-4 w-4" /> Open
          </button>
        </form>
      </div>
      {id ? <Detail id={id} key={id} /> : <Empty title="Pick an alert or enter a transfer id" hint="Click any row on the Live Alerts page to investigate it." />}
    </div>
  );
}

function Detail({ id }: { id: string }) {
  const tx = useApi<TxDetail>(`/v1/transactions/${id}`);
  const ex = useApi<Explanation>(`/v1/transactions/${id}/explanation`);
  const [caseOpen, setCaseOpen] = useState(false);
  const [attempts, setAttempts] = useState(0);
  const nav = useNavigate();
  // Alerts arrive over the WebSocket the moment they are scored; the row reaches PostgreSQL via the asynchronous
  // persister a moment later. A fresh alert can therefore 404 briefly: retry instead of showing an error.
  const notYetPersisted = tx.error === "transaction not found" && attempts < 15;
  const { reload: reloadTx } = tx;
  const { reload: reloadEx } = ex;
  useEffect(() => {
    if (!notYetPersisted) return;
    const t = window.setTimeout(() => {
      setAttempts((a) => a + 1);
      reloadTx();
      reloadEx();
    }, 2000);
    return () => window.clearTimeout(t);
  }, [notYetPersisted, attempts, reloadTx, reloadEx]);
  if (tx.loading && !tx.data) return <Spinner />;
  if (notYetPersisted) return <Spinner label="Transaction is still being written to the database — retrying…" />;
  if (tx.error) return <ErrorBox error={tx.error} onRetry={tx.reload} />;
  if (!tx.data) return null;
  const t = tx.data.transaction;
  const p = (t.payload ?? {}) as Record<string, unknown>;
  const reasons = ex.data?.top_reasons ?? [];
  const degraded = t.components?.degraded_reasons ?? [];
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
      <div className="card p-4 xl:col-span-3">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <div className="text-xs uppercase text-slate-500">Risk score</div>
            <div className={clsx("text-3xl font-bold tabular-nums", (t.risk_score ?? 0) >= 0.5 ? "text-rose-300" : (t.risk_score ?? 0) >= 0.1 ? "text-amber-300" : "text-emerald-300")}>
              {t.risk_score != null ? t.risk_score.toFixed(4) : "—"}
            </div>
          </div>
          <DecisionBadge d={t.decision} />
          <Pill tone="sky">model {t.model_version ?? "—"}</Pill>
          <Pill>combo {t.combo ?? "—"}</Pill>
          <Pill>{fmtMs(t.latency_ms)} scoring</Pill>
          {t.degraded && <Pill tone="amber">degraded: {degraded.join(", ")}</Pill>}
          {tx.data.alert?.predicted_type && <Pill tone="violet">likely {typeLabel(tx.data.alert.predicted_type)}</Pill>}
          <div className="ml-auto flex items-center gap-2">
            {tx.data.feedback.length ? (
              <Pill tone={tx.data.feedback[0].label === "fraud" ? "rose" : "emerald"}>
                feedback: {tx.data.feedback[0].kind.replace("_", " ")} by {tx.data.feedback[0].operator_id}
              </Pill>
            ) : (
              <FeedbackButtons transferId={t.transfer_id} onDone={() => tx.reload()} />
            )}
            <button className="btn-ghost" onClick={() => setCaseOpen(true)}>
              <Briefcase className="h-4 w-4" /> Open case
            </button>
          </div>
        </div>
        {tx.data.cases.length > 0 && (
          <div className="mt-2 text-xs text-slate-400">
            In case(s):{" "}
            {tx.data.cases.map((c) => (
              <Link key={c.case_id} className="mr-2 text-sky-400 hover:underline" to={`/cases/${c.case_id}`}>
                #{c.case_id} {c.title} ({c.status})
              </Link>
            ))}
          </div>
        )}
      </div>

      <div className="card xl:col-span-2">
        <div className="card-h">
          Why this score? <span className="text-xs font-normal text-slate-500">SHAP contributions in ensemble log-odds · {ex.data?.computed?.replace("_", " ")}</span>
        </div>
        <div className="p-4">
          {ex.loading ? <Spinner /> : ex.error ? <ErrorBox error={ex.error} /> : ex.data ? <ShapChart ex={ex.data} /> : null}
          {reasons.length > 0 && (
            <ol className="mt-4 space-y-1.5">
              {reasons.map((r, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <span className={clsx("mt-0.5 w-12 shrink-0 text-right font-mono text-xs", r.contribution > 0 ? "text-rose-300" : "text-emerald-300")}>
                    {r.contribution > 0 ? "+" : ""}
                    {r.contribution.toFixed(2)}
                  </span>
                  <span className="text-slate-200">{r.reason}</span>
                  <Pill>{r.source}</Pill>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-h">Transfer details</div>
        <div className="px-4 py-2">
          <KV k="Transfer id" v={short(t.transfer_id, 18)} mono />
          <KV k="Event time" v={fmtTs(t.ts)} mono />
          <KV k="Amount" v={fmtAmount(t.amount, t.currency)} />
          <KV k="Scenario" v={String(p.scenario ?? "—")} />
          <KV k="Channel" v={String(p.channel ?? "—")} />
          <KV k="Payer" v={<Link className="text-sky-400 hover:underline" to={`/investigate/${t.transfer_id}`}>{t.payer_id}</Link>} mono />
          <KV k="Payer FSP" v={String(p.payer_fsp ?? "—")} />
          <KV k="Payee" v={`${t.payee_id} (${String(p.payee_type ?? "")})`} mono />
          <KV k="Payee FSP" v={String(p.payee_fsp ?? "—")} />
          <KV k="Merchant category" v={String(p.merchant_category || "—")} />
          <KV k="Device" v={String(p.device_id ?? "—")} mono />
          <KV k="IP" v={String(p.ip ?? "—")} mono />
          <KV k="Geo" v={p.lat != null ? `${Number(p.lat).toFixed(3)}, ${Number(p.lon).toFixed(3)}` : "—"} mono />
          <KV k="Sequence model logit" v={ex.data?.sequence_logit != null ? ex.data.sequence_logit.toFixed(3) : "—"} mono />
        </div>
      </div>

      <div className="card xl:col-span-2">
        <div className="card-h">Payer history timeline</div>
        <div className="p-2">
          <Timeline rows={tx.data.payer_history} focus={t.transfer_id} onPick={(tid) => nav(`/investigate/${tid}`)} />
        </div>
      </div>

      <div className="card">
        <div className="card-h">Linked entities & graph evidence</div>
        <div className="space-y-3 p-4 text-sm">
          <div>
            <div className="mb-1 text-xs uppercase text-slate-500">Fraud rings</div>
            {tx.data.linked_entities.rings.length ? (
              tx.data.linked_entities.rings.map((r) => (
                <Link key={r.ring_id} to={`/rings/${r.ring_id}`} className="mb-1 flex items-center justify-between rounded border border-slate-800 px-2 py-1 hover:bg-slate-800/50">
                  <span className="font-mono text-xs text-sky-300">{r.ring_id}</span>
                  <span className="text-xs text-slate-400">
                    {typeLabel(r.ring_type)} · {r.size} accts · score {r.ring_score.toFixed(2)}
                  </span>
                </Link>
              ))
            ) : (
              <div className="text-xs text-slate-500">No ring membership for payer or payee</div>
            )}
          </div>
          <div>
            <div className="mb-1 text-xs uppercase text-slate-500">Accounts on this device ({tx.data.linked_entities.device_accounts.length})</div>
            <div className="flex flex-wrap gap-1">
              {tx.data.linked_entities.device_accounts.slice(0, 24).map((a) => (
                <Pill key={a}>{a}</Pill>
              ))}
            </div>
          </div>
          {t.typology && (
            <div>
              <div className="mb-1 text-xs uppercase text-slate-500">Typology probabilities</div>
              {Object.entries(t.typology)
                .sort((a, b) => b[1] - a[1])
                .map(([k, v]) => (
                  <KV key={k} k={typeLabel(k)} v={`${(v * 100).toFixed(1)}%`} />
                ))}
            </div>
          )}
          <div>
            <div className="mb-1 text-xs uppercase text-slate-500">Latency breakdown</div>
            {Object.entries(t.components?.timings_ms ?? {}).map(([k, v]) => (
              <KV key={k} k={k.replace("_", " ")} v={fmtMs(v)} />
            ))}
          </div>
        </div>
      </div>

      <div className="card xl:col-span-3">
        <div className="card-h">Payee activity (last 30)</div>
        <HistTable rows={tx.data.payee_history} focus={t.transfer_id} />
      </div>
      {caseOpen && <CreateCaseModal transferIds={[t.transfer_id]} onClose={() => setCaseOpen(false)} onCreated={(c) => nav(`/cases/${c.case_id}`)} />}
    </div>
  );
}

function ShapChart({ ex }: { ex: Explanation }) {
  const data = ex.contributions
    .slice()
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 12)
    .map((c) => ({ name: c.feature, value: +c.contribution.toFixed(4), raw: c.value, source: c.source }))
    .reverse();
  if (!data.length) return <Empty title="No contributions" />;
  return (
    <div className="h-80">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
          <XAxis type="number" stroke="#64748b" fontSize={11} />
          <YAxis type="category" dataKey="name" width={190} stroke="#94a3b8" fontSize={11} />
          <ReferenceLine x={0} stroke="#475569" />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
            formatter={(v: number, _n, item) => [`${v > 0 ? "+" : ""}${v} (value ${item.payload.raw ?? "n/a"})`, item.payload.source]}
          />
          <Bar dataKey="value">
            {data.map((d, i) => (
              <Cell key={i} fill={d.value > 0 ? "#f43f5e" : "#10b981"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="text-right text-xs text-slate-500">
        base log-odds {ex.base_logit.toFixed(2)} · combo {ex.combo} · red increases risk, green decreases
      </div>
    </div>
  );
}

function Timeline({ rows, focus, onPick }: { rows: HistRow[]; focus: string; onPick: (id: string) => void }) {
  if (!rows.length) return <Empty title="No history" />;
  const series = (d: string | null) =>
    rows.filter((r) => (r.decision ?? "ALLOW") === (d ?? "ALLOW")).map((r) => ({ x: r.ts, y: r.amount, id: r.transfer_id, focus: r.transfer_id === focus }));
  return (
    <div className="h-56">
      <ResponsiveContainer>
        <ScatterChart margin={{ left: 10, right: 20, top: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis type="number" dataKey="x" domain={["auto", "auto"]} tickFormatter={(v) => fmtTs(v).slice(5, 16)} stroke="#64748b" fontSize={10} />
          <YAxis type="number" dataKey="y" scale="log" domain={["auto", "auto"]} stroke="#64748b" fontSize={10} tickFormatter={(v) => `${Math.round(v)}`} />
          <ZAxis range={[40, 40]} />
          <Tooltip
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
            formatter={(v: number, n) => (n === "x" ? fmtTs(v) : fmtAmount(v))}
          />
          <Scatter name="ALLOW" data={series("ALLOW")} fill="#10b981" onClick={(d) => onPick((d as unknown as { id: string }).id)} />
          <Scatter name="REVIEW" data={series("REVIEW")} fill="#f59e0b" onClick={(d) => onPick((d as unknown as { id: string }).id)} />
          <Scatter name="BLOCK" data={series("BLOCK")} fill="#f43f5e" onClick={(d) => onPick((d as unknown as { id: string }).id)} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function HistTable({ rows, focus }: { rows: HistRow[]; focus: string }) {
  if (!rows.length) return <Empty title="No activity" />;
  return (
    <div className="max-h-72 overflow-auto">
      <table className="w-full">
        <thead className="sticky top-0 bg-slate-900">
          <tr>
            <th className="th">Time</th>
            <th className="th">Payer → Payee</th>
            <th className="th">Scenario</th>
            <th className="th text-right">Amount</th>
            <th className="th">Decision</th>
            <th className="th">Risk</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.transfer_id} className={clsx("border-t border-slate-800/60", r.transfer_id === focus && "bg-sky-500/10")}>
              <td className="td font-mono text-xs">{fmtTs(r.ts)}</td>
              <td className="td font-mono text-xs">
                <Link className="hover:text-sky-300" to={`/investigate/${r.transfer_id}`}>
                  {short(r.payer_id, 12)} → {short(r.payee_id, 12)}
                </Link>
              </td>
              <td className="td text-xs">{r.scenario}</td>
              <td className="td text-right tabular-nums">{fmtAmount(r.amount)}</td>
              <td className="td">
                <DecisionBadge d={r.decision} />
              </td>
              <td className="td font-mono text-xs">{r.risk_score != null ? r.risk_score.toFixed(3) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
