import clsx from "clsx";
import { AlertTriangle, Inbox, Loader2, X } from "lucide-react";
import { ReactNode } from "react";
import type { Decision } from "../lib/api";

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-sm text-slate-400">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="m-4 flex items-start gap-3 rounded-lg border border-rose-900 bg-rose-950/40 p-3 text-sm text-rose-200">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1 break-words">{error}</div>
      {onRetry && (
        <button className="btn-ghost !py-0.5" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1 py-12 text-center text-slate-500">
      <Inbox className="h-7 w-7" />
      <div className="text-sm font-medium text-slate-400">{title}</div>
      {hint && <div className="max-w-sm text-xs">{hint}</div>}
    </div>
  );
}

export function DecisionBadge({ d }: { d: Decision | null | undefined }) {
  if (!d) return <span className="text-slate-500">—</span>;
  return (
    <span
      className={clsx(
        "inline-flex rounded-md px-2 py-0.5 text-xs font-bold tracking-wide",
        d === "BLOCK" && "bg-rose-600/20 text-rose-300 ring-1 ring-rose-600/50",
        d === "REVIEW" && "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/40",
        d === "ALLOW" && "bg-emerald-500/10 text-emerald-300 ring-1 ring-emerald-500/30",
      )}
    >
      {d}
    </span>
  );
}

export function Pill({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "sky" | "amber" | "rose" | "emerald" | "violet" }) {
  const tones = {
    slate: "bg-slate-800 text-slate-300",
    sky: "bg-sky-500/15 text-sky-300",
    amber: "bg-amber-500/15 text-amber-300",
    rose: "bg-rose-500/15 text-rose-300",
    emerald: "bg-emerald-500/15 text-emerald-300",
    violet: "bg-violet-500/15 text-violet-300",
  };
  return <span className={clsx("inline-flex rounded px-1.5 py-0.5 text-xs", tones[tone])}>{children}</span>;
}

export function ScoreBar({ score }: { score: number }) {
  const pct = Math.max(2, Math.min(100, score * 100));
  const color = score >= 0.5 ? "bg-rose-500" : score >= 0.1 ? "bg-amber-400" : "bg-emerald-500";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded bg-slate-800">
        <div className={clsx("h-full", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-12 font-mono text-xs tabular-nums text-slate-300">{score.toFixed(3)}</span>
    </div>
  );
}

export function Kpi({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "ok" | "warn" | "bad" }) {
  return (
    <div className="card p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div
        className={clsx(
          "mt-1 text-2xl font-semibold tabular-nums",
          tone === "ok" && "text-emerald-300",
          tone === "warn" && "text-amber-300",
          tone === "bad" && "text-rose-300",
        )}
      >
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="card w-full max-w-lg" onClick={(e) => e.stopPropagation()}>
        <div className="card-h">
          {title}
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200" aria-label="close">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="mb-3 block">
      <span className="mb-1 block text-xs font-medium text-slate-400">{label}</span>
      {children}
    </label>
  );
}

export function KV({ k, v, mono }: { k: string; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex justify-between gap-4 border-b border-slate-800/60 py-1.5 text-sm last:border-0">
      <span className="text-slate-400">{k}</span>
      <span className={clsx("text-right text-slate-200", mono && "font-mono text-xs")}>{v}</span>
    </div>
  );
}
