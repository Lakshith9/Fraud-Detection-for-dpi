import { KeyRound, ShieldCheck } from "lucide-react";
import { FormEvent, useState } from "react";
import { verifyApiKey } from "../lib/api";
import { Field } from "../components/ui";

/** Operator sign-in: the key is checked against the API, then kept for this browser session only. */
export default function SignIn({ onSignIn, expired }: { onSignIn: (key: string) => void; expired?: boolean }) {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(expired ? "Your key was rejected. Sign in again." : null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const k = key.trim();
    if (!k) return;
    setBusy(true);
    setError(null);
    try {
      if (await verifyApiKey(k)) onSignIn(k);
      else setError("That API key is not valid.");
    } catch (err) {
      setError(`Could not reach the API (${err instanceof Error ? err.message : "network error"}).`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={submit} className="card w-full max-w-sm p-6">
        <div className="mb-5 flex items-center gap-2">
          <ShieldCheck className="h-6 w-6 text-sky-400" />
          <div>
            <div className="text-sm font-bold leading-tight text-slate-100">DPI Fraud Ops</div>
            <div className="text-[11px] text-slate-500">Operator sign-in</div>
          </div>
        </div>
        <Field label="Operator API key">
          <input
            className="input w-full font-mono"
            type="password"
            autoComplete="current-password"
            autoFocus
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="from OPERATOR_API_KEYS"
          />
        </Field>
        {error && <div className="mb-3 text-xs text-rose-300">{error}</div>}
        <button className="btn-primary w-full justify-center" disabled={busy || !key.trim()}>
          <KeyRound className="h-4 w-4" />
          {busy ? "Checking…" : "Sign in"}
        </button>
        <p className="mt-4 text-[11px] leading-snug text-slate-500">
          The key is kept in this tab's session storage and cleared when you sign out or close the browser.
        </p>
      </form>
    </div>
  );
}
