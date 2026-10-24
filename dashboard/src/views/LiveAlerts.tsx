import clsx from "clsx";
import { Briefcase, Volume2, VolumeX } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FeedbackButtons } from "../components/FeedbackButtons";
import { DecisionBadge, Empty, Field, Modal, Pill, ScoreBar, Spinner } from "../components/ui";
import { Alert, Case, post } from "../lib/api";
import type { WsStatus } from "../lib/hooks";
import { FRAUD_TYPES, fmtAmount, fmtTs, short, typeLabel } from "../lib/format";

interface Props {
  stream: { alerts: Alert[]; status: WsStatus };
  sound: boolean;
  setSound: (v: boolean) => void;
}

export default function LiveAlerts({ stream, sound, setSound }: Props) {
  const nav = useNavigate();
  const [decision, setDecision] = useState<"ALL" | "REVIEW" | "BLOCK">("ALL");
  const [ftype, setFtype] = useState("");
  const [fsp, setFsp] = useState("");
  const [minScore, setMinScore] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [caseOpen, setCaseOpen] = useState(false);
  const [paused, setPaused] = useState(false);
  const [frozen, setFrozen] = useState<Alert[]>([]);
  const seen = useRef<Set<string>>(new Set());
  const [fresh, setFresh] = useState<Set<string>>(new Set());

  const source = paused ? frozen : stream.alerts;

  // flash newly arrived BLOCK alerts
  useEffect(() => {
    const newBlocks = stream.alerts.filter((a) => a.decision === "BLOCK" && !seen.current.has(a.transfer_id));
    stream.alerts.forEach((a) => seen.current.add(a.transfer_id));
    if (newBlocks.length && seen.current.size > newBlocks.length) {
      setFresh(new Set(newBlocks.map((a) => a.transfer_id)));
      const t = window.setTimeout(() => setFresh(new Set()), 2800);
      return () => window.clearTimeout(t);
    }
  }, [stream.alerts]);

  const fsps = useMemo(() => {
    const s = new Set<string>();
    stream.alerts.forEach((a) => {
      if (a.payer_fsp) s.add(a.payer_fsp);
      if (a.payee_fsp) s.add(a.payee_fsp);
    });
    return [...s].sort();
  }, [stream.alerts]);

  const rows = useMemo(
    () =>
      source.filter(
        (a) =>
          (decision === "ALL" || a.decision === decision) &&
          (!ftype || a.predicted_type === ftype) &&
          (!fsp || a.payer_fsp === fsp || a.payee_fsp === fsp) &&
          a.risk_score >= minScore,
      ),
    [source, decision, ftype, fsp, minScore],
  );
  const nBlock = source.filter((a) => a.decision === "BLOCK").length;

  const toggle = (id: string) =>
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  return (
    <div className="p-6">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Live Alerts</h1>
          <p className="text-sm text-slate-400">
            Real-time REVIEW / BLOCK decisions from the scoring pipeline · {source.length} alerts in view ·{" "}
            <span className="text-rose-300">{nBlock} blocked</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn-ghost" onClick={() => setSound(!sound)} title="Audible cue for BLOCK alerts">
            {sound ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />} Sound {sound ? "on" : "off"}
          </button>
          <button
            className="btn-ghost"
            onClick={() => {
              if (!paused) setFrozen(stream.alerts);
              setPaused(!paused);
            }}
          >
            {paused ? "Resume stream" : "Pause stream"}
          </button>
          <button className="btn-primary" disabled={!selected.size} onClick={() => setCaseOpen(true)}>
            <Briefcase className="h-4 w-4" /> Create case ({selected.size})
          </button>
        </div>
      </div>

      <div className="card mb-4 flex flex-wrap items-center gap-3 p-3">
        <div className="flex overflow-hidden rounded-lg border border-slate-700">
          {(["ALL", "REVIEW", "BLOCK"] as const).map((d) => (
            <button
              key={d}
              onClick={() => setDecision(d)}
              className={clsx("px-3 py-1.5 text-xs font-semibold", decision === d ? "bg-sky-600 text-white" : "text-slate-400 hover:bg-slate-800")}
            >
              {d}
            </button>
          ))}
        </div>
        <select className="input" value={ftype} onChange={(e) => setFtype(e.target.value)}>
          <option value="">All fraud types</option>
          {FRAUD_TYPES.map((t) => (
            <option key={t} value={t}>
              {typeLabel(t)}
            </option>
          ))}
        </select>
        <select className="input" value={fsp} onChange={(e) => setFsp(e.target.value)}>
          <option value="">All FSPs</option>
          {fsps.map((f) => (
            <option key={f}>{f}</option>
          ))}
        </select>
        <label className="flex items-center gap-2 text-xs text-slate-400">
          Min risk
          <input type="range" min={0} max={1} step={0.01} value={minScore} onChange={(e) => setMinScore(+e.target.value)} />
          <span className="w-10 font-mono text-slate-300">{minScore.toFixed(2)}</span>
        </label>
        {paused && <Pill tone="amber">paused</Pill>}
      </div>

      <div className="card overflow-hidden">
        {stream.status === "connecting" && !source.length ? (
          <Spinner label="Connecting to live alert stream…" />
        ) : !rows.length ? (
          <Empty title="No alerts match" hint={source.length ? "Adjust the filters above." : "Start the simulator (make demo) to see live alerts."} />
        ) : (
          <div className="max-h-[calc(100vh-260px)] overflow-auto">
            <table className="w-full">
              <thead className="sticky top-0 z-10 bg-slate-900">
                <tr className="border-b border-slate-800">
                  <th className="th w-8" />
                  <th className="th">Event time</th>
                  <th className="th">Decision</th>
                  <th className="th">Risk</th>
                  <th className="th">Payer → Payee</th>
                  <th className="th text-right">Amount</th>
                  <th className="th">Likely type</th>
                  <th className="th">Top reason</th>
                  <th className="th">Feedback</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((a) => (
                  <tr
                    key={a.transfer_id}
                    onClick={() => nav(`/investigate/${a.transfer_id}`)}
                    className={clsx(
                      "cursor-pointer border-b border-slate-800/60 hover:bg-slate-800/40",
                      a.decision === "BLOCK" && "bg-rose-950/20",
                      fresh.has(a.transfer_id) && "animate-flash",
                    )}
                  >
                    <td className="td" onClick={(e) => e.stopPropagation()}>
                      <input type="checkbox" checked={selected.has(a.transfer_id)} onChange={() => toggle(a.transfer_id)} />
                    </td>
                    <td className="td whitespace-nowrap font-mono text-xs text-slate-400">{fmtTs(a.ts)}</td>
                    <td className="td">
                      <DecisionBadge d={a.decision} />
                      {a.degraded && <span className="ml-1 text-[10px] text-amber-400" title="scored in degraded mode">●</span>}
                    </td>
                    <td className="td">
                      <ScoreBar score={a.risk_score} />
                    </td>
                    <td className="td whitespace-nowrap font-mono text-xs">
                      {short(a.payer_id, 12)} <span className="text-slate-500">→</span> {short(a.payee_id, 12)}
                      <div className="text-[10px] text-slate-500">
                        {a.payer_fsp} → {a.payee_fsp}
                      </div>
                    </td>
                    <td className="td whitespace-nowrap text-right tabular-nums">{fmtAmount(a.amount)}</td>
                    <td className="td whitespace-nowrap text-xs">{typeLabel(a.predicted_type)}</td>
                    <td className="td max-w-xs truncate text-xs text-slate-300" title={(a.top_reasons ?? []).map((r) => r.reason).join("\n")}>
                      {a.top_reasons?.[0]?.reason ?? "—"}
                    </td>
                    <td className="td" onClick={(e) => e.stopPropagation()}>
                      {a.status && a.status !== "open" && a.status !== "in_case" ? (
                        <Pill tone={a.status === "confirmed_fraud" ? "rose" : "emerald"}>{a.status.replace("_", " ")}</Pill>
                      ) : (
                        <FeedbackButtons transferId={a.transfer_id} compact />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      {caseOpen && (
        <CreateCaseModal
          transferIds={[...selected]}
          onClose={() => setCaseOpen(false)}
          onCreated={(c) => {
            setSelected(new Set());
            setCaseOpen(false);
            nav(`/cases/${c.case_id}`);
          }}
        />
      )}
    </div>
  );
}

export function CreateCaseModal({ transferIds, ringId, onClose, onCreated }: { transferIds: string[]; ringId?: string; onClose: () => void; onCreated: (c: Case) => void }) {
  const [title, setTitle] = useState(ringId ? `Ring ${ringId}` : `Investigation of ${transferIds.length} alert(s)`);
  const [priority, setPriority] = useState("high");
  const [assignee, setAssignee] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try {
      const c = await post<Case>("/v1/cases", { title, transfer_ids: transferIds, ring_id: ringId, priority, assignee: assignee || null, note: note || null });
      onCreated(c);
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  };
  return (
    <Modal title="Create case" onClose={onClose}>
      <Field label="Title">
        <input className="input w-full" value={title} onChange={(e) => setTitle(e.target.value)} />
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Priority">
          <select className="input w-full" value={priority} onChange={(e) => setPriority(e.target.value)}>
            {["low", "medium", "high", "critical"].map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </Field>
        <Field label="Assignee">
          <input className="input w-full" value={assignee} onChange={(e) => setAssignee(e.target.value)} placeholder="operator id" />
        </Field>
      </div>
      <Field label="Initial note">
        <textarea className="input h-20 w-full" value={note} onChange={(e) => setNote(e.target.value)} />
      </Field>
      <div className="mb-3 text-xs text-slate-400">{transferIds.length} linked transfer(s)</div>
      {err && <div className="mb-2 text-xs text-rose-400">{err}</div>}
      <div className="flex justify-end gap-2">
        <button className="btn-ghost" onClick={onClose}>
          Cancel
        </button>
        <button className="btn-primary" disabled={busy || !title} onClick={submit}>
          Create case
        </button>
      </div>
    </Modal>
  );
}
