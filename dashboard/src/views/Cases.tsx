import clsx from "clsx";
import { Plus } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { DecisionBadge, Empty, ErrorBox, Field, KV, Pill, Spinner } from "../components/ui";
import { Case, patch } from "../lib/api";
import { useApi } from "../lib/hooks";
import { fmtAmount, short, typeLabel } from "../lib/format";
import { CreateCaseModal } from "./LiveAlerts";

const STATUS_TONE = { open: "sky", investigating: "amber", closed: "emerald" } as const;

export default function Cases() {
  const { id } = useParams();
  const nav = useNavigate();
  const [status, setStatus] = useState("");
  const [creating, setCreating] = useState(false);
  const list = useApi<{ items: Case[] }>(`/v1/cases${status ? `?status=${status}` : ""}`, 15000);
  return (
    <div className="p-6">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Case Management</h1>
          <p className="text-sm text-slate-400">Group alerts into investigations, assign, annotate and close — every change is audited</p>
        </div>
        <button className="btn-primary" onClick={() => setCreating(true)}>
          <Plus className="h-4 w-4" /> New case
        </button>
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[380px_1fr]">
        <div className="card">
          <div className="flex gap-1 border-b border-slate-800 p-2">
            {["", "open", "investigating", "closed"].map((s) => (
              <button key={s} onClick={() => setStatus(s)} className={clsx("rounded px-2.5 py-1 text-xs", status === s ? "bg-sky-600 text-white" : "text-slate-400 hover:bg-slate-800")}>
                {s || "all"}
              </button>
            ))}
          </div>
          {list.loading && !list.data ? (
            <Spinner />
          ) : list.error ? (
            <ErrorBox error={list.error} onRetry={list.reload} />
          ) : !list.data?.items.length ? (
            <Empty title="No cases" hint="Select alerts on the Live Alerts page and click Create case." />
          ) : (
            <div className="max-h-[calc(100vh-230px)] overflow-auto">
              {list.data.items.map((c) => (
                <button
                  key={c.case_id}
                  onClick={() => nav(`/cases/${c.case_id}`)}
                  className={clsx("block w-full border-b border-slate-800/70 px-4 py-2.5 text-left hover:bg-slate-800/40", String(c.case_id) === id && "bg-sky-500/10")}
                >
                  <div className="flex items-center justify-between">
                    <span className="truncate text-sm text-slate-200">
                      #{c.case_id} {c.title}
                    </span>
                    <Pill tone={STATUS_TONE[c.status]}>{c.status}</Pill>
                  </div>
                  <div className="mt-0.5 text-xs text-slate-500">
                    {c.priority} · {c.n_alerts ?? 0} alerts · {c.assignee ?? "unassigned"} · updated {c.updated_at.slice(0, 16).replace("T", " ")}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
        <div>{id ? <CaseDetail id={id} key={id} onChanged={list.reload} /> : <div className="card"><Empty title="Select a case" /></div>}</div>
      </div>
      {creating && (
        <CreateCaseModal
          transferIds={[]}
          onClose={() => setCreating(false)}
          onCreated={(c) => {
            setCreating(false);
            list.reload();
            nav(`/cases/${c.case_id}`);
          }}
        />
      )}
    </div>
  );
}

function CaseDetail({ id, onChanged }: { id: string; onChanged: () => void }) {
  const c = useApi<Case>(`/v1/cases/${id}`);
  const [note, setNote] = useState("");
  const [assignee, setAssignee] = useState<string | null>(null);
  const [resolution, setResolution] = useState("");
  const [addIds, setAddIds] = useState("");
  const [err, setErr] = useState("");
  const update = async (body: Record<string, unknown>) => {
    setErr("");
    try {
      await patch(`/v1/cases/${id}`, body);
      c.reload();
      onChanged();
    } catch (e) {
      setErr((e as Error).message);
    }
  };
  if (c.loading && !c.data) return <Spinner />;
  if (c.error) return <ErrorBox error={c.error} onRetry={c.reload} />;
  if (!c.data) return null;
  const d = c.data;
  return (
    <div className="space-y-4">
      <div className="card p-4">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-lg font-semibold text-slate-100">
            #{d.case_id} {d.title}
          </h2>
          <Pill tone={STATUS_TONE[d.status]}>{d.status}</Pill>
          <Pill>{d.priority}</Pill>
          {d.ring_id && (
            <Link to={`/rings/${d.ring_id}`} className="text-xs text-sky-400 hover:underline">
              ring {d.ring_id}
            </Link>
          )}
        </div>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <Field label="Status">
            <select className="input" value={d.status} onChange={(e) => update({ status: e.target.value })}>
              {["open", "investigating", "closed"].map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="Priority">
            <select className="input" value={d.priority} onChange={(e) => update({ priority: e.target.value })}>
              {["low", "medium", "high", "critical"].map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="Assignee">
            <div className="flex gap-1">
              <input className="input w-36" value={assignee ?? d.assignee ?? ""} onChange={(e) => setAssignee(e.target.value)} />
              <button className="btn-ghost" disabled={assignee == null} onClick={() => update({ assignee })}>
                Save
              </button>
            </div>
          </Field>
          <Field label="Resolution">
            <div className="flex gap-1">
              <input className="input w-56" placeholder={d.resolution ?? "e.g. confirmed mule network"} value={resolution} onChange={(e) => setResolution(e.target.value)} />
              <button className="btn-ghost" disabled={!resolution} onClick={() => update({ resolution, status: "closed" })}>
                Close
              </button>
            </div>
          </Field>
        </div>
        {err && <div className="text-xs text-rose-400">{err}</div>}
        <div className="mt-1 text-xs text-slate-500">
          created by {d.created_by} · {d.created_at.slice(0, 19).replace("T", " ")}
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="card">
          <div className="card-h">Linked alerts ({d.alerts?.length ?? 0})</div>
          {d.alerts?.length ? (
            <div className="max-h-80 overflow-auto">
              <table className="w-full">
                <tbody>
                  {d.alerts.map((a) => (
                    <tr key={a.transfer_id} className="border-b border-slate-800/60">
                      <td className="td">
                        <DecisionBadge d={a.decision} />
                      </td>
                      <td className="td font-mono text-xs">
                        <Link to={`/investigate/${a.transfer_id}`} className="hover:text-sky-300">
                          {short(a.transfer_id, 10)}
                        </Link>
                      </td>
                      <td className="td text-xs">{typeLabel(a.predicted_type)}</td>
                      <td className="td text-right tabular-nums">{fmtAmount(a.amount)}</td>
                      <td className="td text-xs text-slate-400">{a.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty title="No alerts linked" />
          )}
          <div className="flex gap-2 border-t border-slate-800 p-3">
            <input className="input flex-1 font-mono text-xs" placeholder="add transfer ids (comma separated)" value={addIds} onChange={(e) => setAddIds(e.target.value)} />
            <button
              className="btn-ghost"
              disabled={!addIds.trim()}
              onClick={() => {
                update({ add_transfer_ids: addIds.split(",").map((s) => s.trim()).filter(Boolean) });
                setAddIds("");
              }}
            >
              Add
            </button>
          </div>
        </div>
        <div className="card">
          <div className="card-h">Notes</div>
          <div className="max-h-64 space-y-2 overflow-auto p-4">
            {d.notes?.length ? (
              d.notes.map((n) => (
                <div key={n.id} className="rounded-lg bg-slate-800/50 p-2 text-sm">
                  <div className="text-xs text-slate-400">
                    {n.author} · {n.created_at.slice(0, 16).replace("T", " ")}
                  </div>
                  <div className="whitespace-pre-wrap text-slate-200">{n.body}</div>
                </div>
              ))
            ) : (
              <div className="text-xs text-slate-500">No notes yet</div>
            )}
          </div>
          <div className="flex gap-2 border-t border-slate-800 p-3">
            <textarea className="input h-16 flex-1" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Add an investigation note" />
            <button
              className="btn-primary self-end"
              disabled={!note.trim()}
              onClick={() => {
                update({ note });
                setNote("");
              }}
            >
              Add
            </button>
          </div>
        </div>
      </div>
      <div className="card">
        <div className="card-h">Audit trail</div>
        <div className="px-4 py-2">
          {d.audit?.map((a) => <KV key={a.id} k={`${a.created_at.slice(0, 19).replace("T", " ")} · ${a.actor} · ${a.action}`} v={JSON.stringify(a.detail)} mono />)}
        </div>
      </div>
    </div>
  );
}
