import clsx from "clsx";
import { Activity, Bell, Briefcase, MessageSquareWarning, Network, Search, ShieldCheck } from "lucide-react";
import { useRef, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import type { Alert } from "./lib/api";
import { beep, useAlertStream, useApi } from "./lib/hooks";
import Cases from "./views/Cases";
import Feedback from "./views/Feedback";
import Investigation from "./views/Investigation";
import LiveAlerts from "./views/LiveAlerts";
import Rings from "./views/Rings";
import SystemHealth from "./views/SystemHealth";

const NAV = [
  { to: "/alerts", label: "Live Alerts", icon: Bell },
  { to: "/investigate", label: "Investigation", icon: Search },
  { to: "/rings", label: "Fraud Rings", icon: Network },
  { to: "/cases", label: "Cases", icon: Briefcase },
  { to: "/feedback", label: "Feedback & Models", icon: MessageSquareWarning },
  { to: "/health", label: "System Health", icon: Activity },
];

interface Health {
  status: string;
  checks: Record<string, { ok: boolean; version?: string }>;
}

export default function App() {
  const [sound, setSound] = useState(false);
  const soundRef = useRef(sound);
  soundRef.current = sound;
  const onAlert = (a: Alert) => {
    if (a.decision === "BLOCK" && soundRef.current) beep();
  };
  const stream = useAlertStream(onAlert);
  const health = useApi<Health>("/healthz", 10000);
  const ok = health.data?.status === "ok";

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-950 md:flex">
        <div className="flex items-center gap-2 px-5 py-5">
          <ShieldCheck className="h-6 w-6 text-sky-400" />
          <div>
            <div className="text-sm font-bold leading-tight text-slate-100">DPI Fraud Ops</div>
            <div className="text-[11px] text-slate-500">Mojaloop-style payment rails</div>
          </div>
        </div>
        <nav className="flex-1 space-y-0.5 px-3">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition",
                  isActive ? "bg-sky-500/10 text-sky-300" : "text-slate-400 hover:bg-slate-900 hover:text-slate-200",
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="space-y-2 border-t border-slate-800 p-4 text-xs text-slate-500">
          <div className="flex items-center justify-between">
            <span>Live stream</span>
            <span className={clsx("flex items-center gap-1.5", stream.status === "open" ? "text-emerald-400" : "text-amber-400")}>
              <span className={clsx("h-2 w-2 rounded-full", stream.status === "open" ? "bg-emerald-400" : "bg-amber-400")} />
              {stream.status}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Backend</span>
            <span className={ok ? "text-emerald-400" : "text-rose-400"}>{health.data ? health.data.status : health.error ? "unreachable" : "…"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Model</span>
            <span className="font-mono text-slate-300">{health.data?.checks?.model?.version ?? "—"}</span>
          </div>
          <div className="pt-1 text-[10px] leading-snug text-slate-600">Simulated Mojaloop-compatible events — not a live Mojaloop switch.</div>
        </div>
      </aside>
      <main className="min-w-0 flex-1">
        <div className="flex gap-1 overflow-x-auto border-b border-slate-800 px-3 py-2 md:hidden">
          {NAV.map(({ to, label }) => (
            <NavLink key={to} to={to} className={({ isActive }) => clsx("whitespace-nowrap rounded px-2 py-1 text-xs", isActive ? "bg-sky-500/15 text-sky-300" : "text-slate-400")}>
              {label}
            </NavLink>
          ))}
        </div>
        <Routes>
          <Route path="/" element={<Navigate to="/alerts" replace />} />
          <Route path="/alerts" element={<LiveAlerts stream={stream} sound={sound} setSound={setSound} />} />
          <Route path="/investigate" element={<Investigation />} />
          <Route path="/investigate/:id" element={<Investigation />} />
          <Route path="/rings" element={<Rings ringEvents={stream.ringEvents} />} />
          <Route path="/rings/:id" element={<Rings ringEvents={stream.ringEvents} />} />
          <Route path="/cases" element={<Cases />} />
          <Route path="/cases/:id" element={<Cases />} />
          <Route path="/feedback" element={<Feedback />} />
          <Route path="/health" element={<SystemHealth />} />
          <Route path="*" element={<Navigate to="/alerts" replace />} />
        </Routes>
      </main>
    </div>
  );
}
