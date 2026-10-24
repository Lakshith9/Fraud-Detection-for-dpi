import clsx from "clsx";
import { Briefcase } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { Link, useNavigate, useParams } from "react-router-dom";
import { DecisionBadge, Empty, ErrorBox, KV, Pill, Spinner } from "../components/ui";
import { Ring } from "../lib/api";
import { useApi } from "../lib/hooks";
import { fmtAmount, fmtTs, typeLabel } from "../lib/format";
import { CreateCaseModal } from "./LiveAlerts";

const TYPE_TONE: Record<string, "rose" | "amber" | "violet" | "sky" | "emerald"> = {
  cycle: "rose",
  mule_network: "amber",
  dense_subgraph: "violet",
  shared_device: "sky",
  reciprocal_burst: "emerald",
};

export default function Rings({ ringEvents }: { ringEvents: number }) {
  const { id } = useParams();
  const nav = useNavigate();
  const [type, setType] = useState("");
  const [minScore, setMinScore] = useState(0.5);
  const list = useApi<{ items: Ring[] }>(`/v1/rings?min_score=${minScore}${type ? `&ring_type=${type}` : ""}&limit=200`);
  const { reload } = list;
  useEffect(() => {
    if (ringEvents) reload();
  }, [ringEvents, reload]);
  const items = list.data?.items ?? [];

  return (
    <div className="p-6">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Fraud Ring Graph</h1>
          <p className="text-sm text-slate-400">Rings detected by the asynchronous graph worker on the trailing 24h event window</p>
        </div>
        <div className="flex items-center gap-2">
          <select className="input" value={type} onChange={(e) => setType(e.target.value)}>
            <option value="">All ring types</option>
            {Object.keys(TYPE_TONE).map((t) => (
              <option key={t} value={t}>
                {typeLabel(t)}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-xs text-slate-400">
            Min score
            <input type="range" min={0.35} max={1} step={0.05} value={minScore} onChange={(e) => setMinScore(+e.target.value)} />
            <span className="w-8 font-mono">{minScore.toFixed(2)}</span>
          </label>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[360px_1fr]">
        <div className="card max-h-[calc(100vh-170px)] overflow-auto">
          <div className="card-h">
            Active rings <span className="text-xs font-normal text-slate-500">{items.length}</span>
          </div>
          {list.loading && !list.data ? (
            <Spinner />
          ) : list.error ? (
            <ErrorBox error={list.error} onRetry={list.reload} />
          ) : !items.length ? (
            <Empty title="No rings at this threshold" hint="Rings appear once the graph worker has processed live traffic." />
          ) : (
            items.map((r) => (
              <button
                key={r.ring_id}
                onClick={() => nav(`/rings/${r.ring_id}`)}
                className={clsx("block w-full border-b border-slate-800/70 px-4 py-2.5 text-left hover:bg-slate-800/40", id === r.ring_id && "bg-sky-500/10")}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-sky-300">{r.ring_id}</span>
                  <span className="font-mono text-xs text-slate-300">{r.ring_score.toFixed(2)}</span>
                </div>
                <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
                  <Pill tone={TYPE_TONE[r.ring_type] ?? "sky"}>{typeLabel(r.ring_type)}</Pill>
                  {r.members.length} accounts · {fmtTs(r.last_seen).slice(5, 16)}
                </div>
              </button>
            ))
          )}
        </div>
        <div>{id ? <RingDetail id={id} key={id} /> : <div className="card"><Empty title="Select a ring" hint="Pick a ring on the left to see its transaction graph and evidence." /></div>}</div>
      </div>
    </div>
  );
}

interface GNode {
  id: string;
  member: boolean;
  cycle: boolean;
  x?: number;
  y?: number;
}
interface GLink {
  source: string;
  target: string;
  amount: number;
  ts: number;
  cycle: boolean;
}

function RingDetail({ id }: { id: string }) {
  const r = useApi<Ring>(`/v1/rings/${id}`);
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const [caseOpen, setCaseOpen] = useState(false);
  const nav = useNavigate();
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver((e) => setW(e[0].contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, [r.data]);

  const graph = useMemo(() => {
    if (!r.data) return { nodes: [] as GNode[], links: [] as GLink[] };
    const members = new Set(r.data.members);
    const path = (r.data.evidence?.cycle_path as string[] | undefined) ?? [];
    const cycEdges = new Set(path.slice(0, -1).map((a, i) => `${a}>${path[i + 1]}`));
    const nodes = new Map<string, GNode>();
    const links: GLink[] = [];
    for (const [s, t, amt, ts] of r.data.edges ?? []) {
      for (const n of [s, t]) if (!nodes.has(n)) nodes.set(n, { id: n, member: members.has(n), cycle: path.includes(n) });
      links.push({ source: s, target: t, amount: amt, ts, cycle: cycEdges.has(`${s}>${t}`) });
    }
    for (const m of r.data.members) if (!nodes.has(m)) nodes.set(m, { id: m, member: true, cycle: path.includes(m) });
    return { nodes: [...nodes.values()], links };
  }, [r.data]);

  if (r.loading) return <Spinner />;
  if (r.error) return <ErrorBox error={r.error} onRetry={r.reload} />;
  if (!r.data) return null;
  const ev = r.data.evidence ?? {};
  return (
    <div className="space-y-4">
      <div className="card">
        <div className="card-h">
          <span>
            {r.data.ring_id} <Pill tone={TYPE_TONE[r.data.ring_type] ?? "sky"}>{typeLabel(r.data.ring_type)}</Pill>
          </span>
          <button className="btn-ghost !py-1" onClick={() => setCaseOpen(true)}>
            <Briefcase className="h-4 w-4" /> Open case
          </button>
        </div>
        <div ref={box} className="h-[460px] bg-slate-950/60">
          <ForceGraph2D
            width={w}
            height={460}
            graphData={graph}
            backgroundColor="rgba(0,0,0,0)"
            nodeRelSize={5}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={0.9}
            linkDirectionalParticles={(l: object) => ((l as GLink).cycle ? 3 : 0)}
            linkColor={(l: object) => ((l as GLink).cycle ? "#f59e0b" : "rgba(148,163,184,0.35)")}
            linkWidth={(l: object) => Math.max(0.5, Math.log10((l as GLink).amount + 1) - 2.5)}
            nodeLabel={(n: object) => (n as GNode).id}
            nodeCanvasObject={(n: object, ctx, scale) => {
              const node = n as GNode;
              const r0 = node.member ? 5 : 3;
              ctx.beginPath();
              ctx.arc(node.x ?? 0, node.y ?? 0, r0, 0, 2 * Math.PI);
              ctx.fillStyle = node.cycle ? "#f59e0b" : node.member ? "#f43f5e" : "#64748b";
              ctx.fill();
              if (scale > 1.6) {
                ctx.font = `${10 / scale}px Inter`;
                ctx.fillStyle = "#cbd5e1";
                ctx.fillText(node.id, (node.x ?? 0) + 6, (node.y ?? 0) + 3);
              }
            }}
            onNodeClick={(n: object) => navigator.clipboard?.writeText((n as GNode).id)}
            cooldownTicks={120}
          />
        </div>
        <div className="flex gap-4 border-t border-slate-800 px-4 py-2 text-xs text-slate-400">
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-rose-500" />ring member</span>
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-500" />cycle path</span>
          <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-slate-500" />counterparty</span>
          <span>edge width ∝ log(amount)</span>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="card">
          <div className="card-h">Evidence</div>
          <div className="px-4 py-2">
            <KV k="Ring score" v={r.data.ring_score.toFixed(3)} />
            <KV k="First seen" v={fmtTs(r.data.first_seen)} mono />
            <KV k="Last seen" v={fmtTs(r.data.last_seen)} mono />
            {Object.entries(ev).map(([k, v]) => (
              <KV key={k} k={k.replace(/_/g, " ")} v={Array.isArray(v) ? (v as string[]).join(" → ") : k.includes("amount") ? fmtAmount(Number(v)) : String(v)} mono={Array.isArray(v)} />
            ))}
            {r.data.devices?.length > 0 && <KV k="Devices" v={r.data.devices.slice(0, 8).join(", ")} mono />}
          </div>
        </div>
        <div className="card">
          <div className="card-h">Alerts involving ring members</div>
          {r.data.alerts?.length ? (
            <div className="max-h-80 overflow-auto">
              <table className="w-full">
                <tbody>
                  {r.data.alerts.map((a) => (
                    <tr key={a.transfer_id} className="border-b border-slate-800/60">
                      <td className="td">
                        <DecisionBadge d={a.decision} />
                      </td>
                      <td className="td font-mono text-xs">
                        <Link to={`/investigate/${a.transfer_id}`} className="hover:text-sky-300">
                          {a.payer_id} → {a.payee_id}
                        </Link>
                      </td>
                      <td className="td text-right tabular-nums">{fmtAmount(a.amount)}</td>
                      <td className="td font-mono text-xs">{a.risk_score.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty title="No alerts on ring members yet" />
          )}
        </div>
      </div>
      {caseOpen && (
        <CreateCaseModal
          ringId={r.data.ring_id}
          transferIds={(r.data.alerts ?? []).slice(0, 50).map((a) => a.transfer_id)}
          onClose={() => setCaseOpen(false)}
          onCreated={(c) => nav(`/cases/${c.case_id}`)}
        />
      )}
    </div>
  );
}
