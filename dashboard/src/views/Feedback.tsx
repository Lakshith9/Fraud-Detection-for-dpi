import clsx from "clsx";
import { RefreshCw, RotateCcw, Send } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { Empty, ErrorBox, Field, Kpi, Pill, Spinner } from "../components/ui";
import { FeedbackRow, ModelsInfo, post } from "../lib/api";
import { useApi } from "../lib/hooks";
import { short } from "../lib/format";

const KIND_TONE: Record<string, "rose" | "emerald" | "amber" | "slate"> = {
  true_positive: "rose",
  false_positive: "emerald",
  false_negative: "amber",
  true_negative: "slate",
};

function metric(m: Record<string, unknown> | null | undefined, k: string): string {
  const v = m?.[k];
  return typeof v === "number" ? v.toFixed(4) : "—";
}

export default function Feedback() {
  const fb = useApi<{ counts: Record<string, number>; items: FeedbackRow[] }>("/v1/feedback?limit=200", 10000);
  const models = useApi<ModelsInfo>("/v1/models", 8000);
  const [tid, setTid] = useState("");
  const [notes, setNotes] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const reportFN = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await post<{ kind: string }>("/v1/feedback", { transfer_id: tid.trim(), label: "fraud", notes: notes || null });
      setMsg({ ok: true, text: `Recorded as ${r.kind.replace("_", " ")}` });
      setTid("");
      setNotes("");
      fb.reload();
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };
  const act = async (path: string) => {
    setMsg(null);
    try {
      const r = await post<Record<string, unknown>>(path, {});
      setMsg({ ok: true, text: JSON.stringify(r) });
      models.reload();
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    }
  };
  const counts = fb.data?.counts ?? {};

  return (
    <div className="space-y-4 p-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Feedback & Model Lifecycle</h1>
        <p className="text-sm text-slate-400">Operator labels retrain the models; a candidate is promoted only if it does not degrade the held-out metrics</p>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi label="True positives" value={counts.true_positive ?? 0} tone="bad" />
        <Kpi label="False positives" value={counts.false_positive ?? 0} tone="ok" />
        <Kpi label="Missed fraud (FN)" value={counts.false_negative ?? 0} tone="warn" />
        <Kpi label="Production model" value={models.data?.current ?? "—"} sub={`loaded by API: ${models.data?.loaded ?? "—"}`} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="card p-4">
          <div className="mb-3 text-sm font-semibold text-slate-200">Report missed fraud (false negative)</div>
          <Field label="Transfer id">
            <input className="input w-full font-mono text-xs" value={tid} onChange={(e) => setTid(e.target.value)} placeholder="transfer id that was ALLOWed" />
          </Field>
          <Field label="Notes">
            <textarea className="input h-20 w-full" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="e.g. customer complaint, chargeback" />
          </Field>
          <button className="btn-danger" disabled={!tid.trim() || busy} onClick={reportFN}>
            <Send className="h-4 w-4" /> Report fraud
          </button>
          {msg && <div className={clsx("mt-3 break-words text-xs", msg.ok ? "text-emerald-400" : "text-rose-400")}>{msg.text}</div>}
          <div className="mt-6 border-t border-slate-800 pt-4">
            <div className="mb-2 text-sm font-semibold text-slate-200">Model actions</div>
            <div className="flex flex-wrap gap-2">
              <button className="btn-primary" onClick={() => act("/v1/models/retrain")}>
                <RefreshCw className="h-4 w-4" /> Retrain now
              </button>
              <button className="btn-ghost" onClick={() => act("/v1/models/rollback")}>
                <RotateCcw className="h-4 w-4" /> Roll back
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-500">Retraining runs in a separate process; the scoring workers hot-reload a promoted model without downtime.</p>
          </div>
        </div>

        <div className="card xl:col-span-2">
          <div className="card-h">Recent feedback</div>
          {fb.loading && !fb.data ? (
            <Spinner />
          ) : fb.error ? (
            <ErrorBox error={fb.error} onRetry={fb.reload} />
          ) : !fb.data?.items.length ? (
            <Empty title="No feedback yet" hint="Mark alerts as True Fraud / False Positive on the Live Alerts page." />
          ) : (
            <div className="max-h-96 overflow-auto">
              <table className="w-full">
                <thead className="sticky top-0 bg-slate-900">
                  <tr>
                    <th className="th">When</th>
                    <th className="th">Transfer</th>
                    <th className="th">Kind</th>
                    <th className="th">Operator</th>
                    <th className="th">Used in</th>
                  </tr>
                </thead>
                <tbody>
                  {fb.data.items.map((f) => (
                    <tr key={f.id} className="border-t border-slate-800/60">
                      <td className="td font-mono text-xs">{f.created_at.slice(0, 19).replace("T", " ")}</td>
                      <td className="td font-mono text-xs">
                        <Link to={`/investigate/${f.transfer_id}`} className="hover:text-sky-300">
                          {short(f.transfer_id, 12)}
                        </Link>
                      </td>
                      <td className="td">
                        <Pill tone={KIND_TONE[f.kind]}>{f.kind.replace("_", " ")}</Pill>
                      </td>
                      <td className="td text-xs">{f.operator_id}</td>
                      <td className="td font-mono text-xs">{f.used_in_version ?? "pending"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-h">Model registry</div>
        {models.loading && !models.data ? (
          <Spinner />
        ) : models.error ? (
          <ErrorBox error={models.error} onRetry={models.reload} />
        ) : (
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">Version</th>
                <th className="th">Status</th>
                <th className="th">Parent</th>
                <th className="th">Test PR-AUC</th>
                <th className="th">Recall@1%FPR</th>
                <th className="th">Notes</th>
                <th className="th" />
              </tr>
            </thead>
            <tbody>
              {models.data?.versions.map((v) => (
                <tr key={v.version} className="border-t border-slate-800/60">
                  <td className="td font-mono">{v.version}</td>
                  <td className="td">
                    <Pill tone={v.status === "production" ? "emerald" : v.status === "rejected" ? "rose" : "slate"}>{v.status}</Pill>
                  </td>
                  <td className="td font-mono text-xs">{v.parent_version ?? "—"}</td>
                  <td className="td font-mono text-xs">{metric(v.test_full, "pr_auc")}</td>
                  <td className="td font-mono text-xs">{metric(v.test_full, "recall_at_1pct_fpr")}</td>
                  <td className="td max-w-md truncate text-xs text-slate-400" title={v.notes ?? ""}>
                    {v.notes}
                  </td>
                  <td className="td">
                    {v.status !== "production" && (
                      <button className="btn-ghost !py-0.5 text-xs" onClick={() => act(`/v1/models/${v.version}/promote`)}>
                        Promote
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <div className="card-h">Retraining runs (gate: held-out window, observed labels)</div>
        {!models.data?.retrain_runs.length ? (
          <Empty title="No retraining runs yet" />
        ) : (
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">#</th>
                <th className="th">Trigger</th>
                <th className="th">Status</th>
                <th className="th">Base → candidate</th>
                <th className="th">Feedback</th>
                <th className="th">PR-AUC before → after</th>
                <th className="th">Decision</th>
              </tr>
            </thead>
            <tbody>
              {models.data.retrain_runs.map((r) => (
                <tr key={r.id} className="border-t border-slate-800/60">
                  <td className="td">{r.id}</td>
                  <td className="td text-xs">{r.trigger}</td>
                  <td className="td text-xs">{r.status}</td>
                  <td className="td font-mono text-xs">
                    {r.base_version} → {r.candidate_version ?? "…"}
                  </td>
                  <td className="td text-xs">{r.n_feedback ?? "—"}</td>
                  <td className="td font-mono text-xs">
                    {metric(r.before_metrics?.observed, "pr_auc")} → {metric(r.after_metrics?.observed, "pr_auc")}
                  </td>
                  <td className="td max-w-md text-xs" title={r.reason ?? ""}>
                    {r.promoted == null ? "—" : r.promoted ? <Pill tone="emerald">promoted</Pill> : <Pill tone="rose">kept current</Pill>}{" "}
                    <span className="text-slate-400">{r.reason}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
