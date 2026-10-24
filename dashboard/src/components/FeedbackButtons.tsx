import { Check, ShieldAlert } from "lucide-react";
import { useState } from "react";
import { post } from "../lib/api";

/** True Fraud / False Positive buttons used on every alert. */
export function FeedbackButtons({ transferId, compact, onDone }: { transferId: string; compact?: boolean; onDone?: (label: string) => void }) {
  const [state, setState] = useState<"idle" | "busy" | "fraud" | "legit" | "error">("idle");
  const [err, setErr] = useState<string>("");

  const send = async (label: "fraud" | "legit") => {
    setState("busy");
    try {
      await post("/v1/feedback", { transfer_id: transferId, label });
      setState(label);
      onDone?.(label);
    } catch (e) {
      setErr((e as Error).message);
      setState("error");
    }
  };

  if (state === "fraud") return <span className="text-xs font-medium text-rose-300">✓ True fraud</span>;
  if (state === "legit") return <span className="text-xs font-medium text-emerald-300">✓ False positive</span>;
  return (
    <div className="flex items-center gap-1.5">
      <button
        className="btn-danger !px-2 !py-1 text-xs"
        disabled={state === "busy"}
        onClick={(e) => {
          e.stopPropagation();
          send("fraud");
        }}
        title="Confirm as true fraud"
      >
        <ShieldAlert className="h-3.5 w-3.5" />
        {!compact && "True fraud"}
      </button>
      <button
        className="btn-ghost !px-2 !py-1 text-xs"
        disabled={state === "busy"}
        onClick={(e) => {
          e.stopPropagation();
          send("legit");
        }}
        title="Mark as false positive"
      >
        <Check className="h-3.5 w-3.5" />
        {!compact && "False positive"}
      </button>
      {state === "error" && <span className="text-xs text-rose-400" title={err}>failed</span>}
    </div>
  );
}
