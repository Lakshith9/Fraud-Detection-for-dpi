export type Decision = "ALLOW" | "REVIEW" | "BLOCK";

export interface Reason {
  feature: string;
  value: number | null;
  contribution: number;
  direction: string;
  source: string;
  reason: string;
}

export interface Alert {
  alert_id?: number;
  transfer_id: string;
  ts: number;
  risk_score: number;
  decision: Decision;
  predicted_type: string | null;
  payer_id: string;
  payee_id: string;
  payer_fsp: string | null;
  payee_fsp: string | null;
  amount: number;
  top_reasons: Reason[];
  model_version: string;
  degraded: boolean;
  status?: string;
  created_at?: string;
  case_id?: number | null;
}

export interface Contribution {
  feature: string;
  value: number | null;
  contribution: number;
  source: string;
}

export interface Explanation {
  transfer_id: string;
  risk_score: number;
  decision: Decision;
  model_version: string;
  computed: string;
  combo: string;
  base_logit: number;
  sequence_logit: number | null;
  contributions: Contribution[];
  top_reasons: Reason[];
  graph: Record<string, unknown> | null;
}

export interface HistRow {
  transfer_id: string;
  ts: number;
  payer_id: string;
  payee_id: string;
  amount: number;
  scenario: string;
  risk_score: number | null;
  decision: Decision | null;
  device_id: string | null;
}

export interface TxDetail {
  transaction: Record<string, unknown> & {
    transfer_id: string;
    ts: number;
    payer_id: string;
    payee_id: string;
    amount: number;
    currency: string;
    risk_score: number | null;
    decision: Decision | null;
    model_version: string | null;
    combo: string | null;
    degraded: boolean;
    latency_ms: number | null;
    typology: Record<string, number> | null;
    components: { degraded_reasons?: string[]; timings_ms?: Record<string, number>; seq_logit?: number | null } | null;
  };
  alert: Alert | null;
  feedback: FeedbackRow[];
  cases: { case_id: number; title: string; status: string }[];
  payer_history: HistRow[];
  payee_history: HistRow[];
  linked_entities: { device_accounts: string[]; rings: RingSummary[] };
}

export interface RingSummary {
  ring_id: string;
  ring_type: string;
  ring_score: number;
  size?: number;
  evidence: Record<string, unknown>;
}

export interface Ring extends RingSummary {
  members: string[];
  devices: string[];
  first_seen: number;
  last_seen: number;
  active: boolean;
  edges?: [string, string, number, number][];
  alerts?: { transfer_id: string; decision: Decision; risk_score: number; payer_id: string; payee_id: string; amount: number }[];
}

export interface FeedbackRow {
  id: number;
  transfer_id: string;
  label: "fraud" | "legit";
  kind: string;
  operator_id: string;
  notes: string | null;
  created_at: string;
  used_in_version: string | null;
}

export interface Case {
  case_id: number;
  title: string;
  status: "open" | "investigating" | "closed";
  priority: string;
  assignee: string | null;
  created_by: string;
  resolution: string | null;
  ring_id: string | null;
  created_at: string;
  updated_at: string;
  n_alerts?: number;
  alerts?: (Partial<Alert> & { transfer_id: string })[];
  notes?: { id: number; author: string; body: string; created_at: string }[];
  audit?: { id: number; actor: string; action: string; detail: Record<string, unknown>; created_at: string }[];
}

export interface Pcts {
  n: number;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  max: number | null;
}

export interface Metrics {
  time: number;
  throughput: { tps_last_10s: number; tps_last_60s: number; series: number[] };
  latency_ms: Pcts;
  e2e_latency_ms: Pcts;
  counts: Record<string, number>;
  alert_rate: number;
  block_rate: number;
  degraded_rate: number;
  model: { version: string; thresholds: Record<string, { review: number; block: number }> };
  graph: { last_run_age_s: number | null; stale: boolean; rings: number; nodes: number; run_ms: number; window_rows: number };
  workers: { partition: number; alive: boolean; processed: number; errors: number; model: string; db_backlog: number; db_healthy: boolean; stream_len: number; pending: number }[];
  persister: { alive: boolean; written: number; explained_async: number; explain_skipped: number; lag: number | null };
  feedback: Record<string, number>;
  drift: { n: number; max_psi?: number; features: Record<string, { psi: number; status: string }> };
}

export interface ModelsInfo {
  current: string | null;
  loaded: string;
  versions: { version: string; status: string; created_at: string; promoted_at: string | null; data_hash: string; parent_version: string | null; notes: string | null; test_full: Record<string, unknown> | null }[];
  retrain_runs: {
    id: number; started_at: string; finished_at: string | null; trigger: string; status: string; base_version: string;
    candidate_version: string | null; promoted: boolean | null; reason: string | null; n_feedback: number | null;
    before_metrics: Record<string, Record<string, unknown>> | null; after_metrics: Record<string, Record<string, unknown>> | null;
  }[];
}

const BASE: string = import.meta.env.VITE_API_URL ?? "";
export const API_KEY: string = import.meta.env.VITE_API_KEY ?? "";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY, ...(init.headers ?? {}) },
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

export const post = <T,>(path: string, body: unknown) => api<T>(path, { method: "POST", body: JSON.stringify(body) });
export const patch = <T,>(path: string, body: unknown) => api<T>(path, { method: "PATCH", body: JSON.stringify(body) });

export function wsUrl(path: string): string {
  if (BASE) return BASE.replace(/^http/, "ws") + `${path}?api_key=${encodeURIComponent(API_KEY)}`;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}?api_key=${encodeURIComponent(API_KEY)}`;
}
