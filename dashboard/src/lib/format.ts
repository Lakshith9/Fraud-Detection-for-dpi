export const fmtAmount = (x: number | null | undefined, ccy = "INR") =>
  x == null ? "—" : `${ccy === "INR" ? "₹" : ccy + " "}${x.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;

export const fmtTs = (ms: number | null | undefined) =>
  ms == null ? "—" : new Date(ms).toISOString().replace("T", " ").slice(0, 19);

export const fmtTime = (ms: number) => new Date(ms).toISOString().slice(11, 19);

export const fmtPct = (x: number | null | undefined, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);

export const fmtMs = (x: number | null | undefined) => (x == null ? "—" : `${x.toFixed(x < 10 ? 2 : 1)} ms`);

export const short = (s: string | null | undefined, n = 8) => (!s ? "—" : s.length > n + 2 ? s.slice(0, n) + "…" : s);

export const typeLabel = (t: string | null | undefined) =>
  !t ? "—" : t.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export const FRAUD_TYPES = ["identity_spoofing", "settlement_manipulation", "merchant_fraud_ring", "transaction_laundering"];
