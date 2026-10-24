import { useCallback, useEffect, useRef, useState } from "react";
import { api, Alert, wsUrl } from "./api";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Fetch `path` on mount / when deps change, optionally polling every `intervalMs`. */
export function useApi<T>(path: string | null, intervalMs?: number): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }
    let alive = true;
    const run = (first: boolean) => {
      if (first) setLoading(true);
      api<T>(path)
        .then((d) => {
          if (!alive) return;
          setData(d);
          setError(null);
        })
        .catch((e: Error) => alive && setError(e.message))
        .finally(() => alive && setLoading(false));
    };
    run(true);
    const id = intervalMs ? window.setInterval(() => run(false), intervalMs) : undefined;
    return () => {
      alive = false;
      if (id) window.clearInterval(id);
    };
  }, [path, intervalMs, tick]);

  return { data, error, loading, reload };
}

export type WsStatus = "connecting" | "open" | "closed";

/** Live alert stream over WebSocket with automatic reconnect and keep-alive. */
export function useAlertStream(onAlert?: (a: Alert) => void, max = 500) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [ringEvents, setRingEvents] = useState(0);
  const cb = useRef(onAlert);
  cb.current = onAlert;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let stopped = false;
    let retry = 0;
    let ping: number | undefined;
    let timer: number | undefined;

    const connect = () => {
      setStatus("connecting");
      ws = new WebSocket(wsUrl("/ws/alerts"));
      ws.onopen = () => {
        retry = 0;
        setStatus("open");
        ping = window.setInterval(() => ws?.readyState === WebSocket.OPEN && ws.send("ping"), 20000);
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "snapshot") setAlerts(msg.data as Alert[]);
        else if (msg.type === "alert") {
          const a = msg.data as Alert;
          setAlerts((prev) => [a, ...prev.filter((p) => p.transfer_id !== a.transfer_id)].slice(0, max));
          cb.current?.(a);
        } else if (msg.type === "ring") setRingEvents((n) => n + 1);
      };
      ws.onclose = () => {
        window.clearInterval(ping);
        setStatus("closed");
        if (!stopped) timer = window.setTimeout(connect, Math.min(10000, 500 * 2 ** retry++));
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      stopped = true;
      window.clearInterval(ping);
      window.clearTimeout(timer);
      ws?.close();
    };
  }, [max]);

  return { alerts, setAlerts, status, ringEvents };
}

let audioCtx: AudioContext | null = null;
export function beep(freq = 880, ms = 160) {
  try {
    audioCtx = audioCtx ?? new AudioContext();
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.frequency.value = freq;
    o.type = "square";
    g.gain.value = 0.04;
    o.connect(g).connect(audioCtx.destination);
    o.start();
    o.stop(audioCtx.currentTime + ms / 1000);
  } catch {
    /* audio not available */
  }
}
