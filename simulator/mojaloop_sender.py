"""Replay the labelled synthetic window through a live Mojaloop switch as real FSPIOP transfers.

Where ``simulator.simulator`` writes events straight onto the scoring stream, this sender makes every transfer
travel the real path:

    sender (as payer FSP) --POST /transfers--> ml-api-adapter --> Kafka topic-transfer-prepare
        --> central-ledger (positions) --> payee FSP callback (this process: fulfils) --> COMMITTED
                                  \\--> pipeline.mojaloop_consumer --> scoring workers --> alerts / dashboard

1. Onboards the dataset's FSPs (dfsp-01 ... dfsp-12) on the switch through the central-ledger admin API:
   currency, net-debit-cap, settlement funds and callback endpoints pointing at this process.
2. Serves those callbacks as a mock FSP: as payee it fulfils every transfer (fulfilment derived from the
   transfer id, condition = SHA-256 of it); as payer it records the switch's final state.
3. Sends each transfer as a ``POST /transfers`` with a real ILPv4 packet carrying the FSPIOP ``Transaction``.
   ``--no-device-signals`` leaves device, IP, geo and channel out of the ``extensionList`` (plain FSPIOP).
   The original event time travels in the ``eventTime`` extension; the engine must run with
   ``TRUST_FSP_EVENT_TIME=true`` to use it (replay only).
4. Fraud-registry reports are not FSPIOP traffic: they go straight onto the scoring stream as in the simulator.
5. Writes the ground truth (under the switch's transfer ids) for ``scripts.evaluate_live``.

Pacing: at most ``--tps``, and never more than ``--max-in-flight`` transfers unfinished in the switch, so the
replay runs at the switch's own speed and every transfer can complete before it expires.

Usage:
    python -m simulator.mojaloop_sender --from-day 53 --to-day 60
    python -m simulator.mojaloop_sender --no-device-signals --callback-host 172.18.0.1   # Linux Docker host
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from email.utils import formatdate
from pathlib import Path

import httpx
import orjson
import uvicorn
from fastapi import FastAPI, Request, Response

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.redis_client import get_redis, partition_of, stream_key
from pipeline.ilp import b64url, encode_ilp_prepare
from pipeline.mojaloop_adapter import EVENT_TIME_EXT, _iso, fspiop_extensions, fspiop_transaction
from simulator.simulator import TX_COLS, build_timeline, write_ground_truth

log = get_logger("mojaloop_sender")
CT = "application/vnd.interoperability.transfers+json;version=1.1"
ACCEPT = "application/vnd.interoperability.transfers+json;version=1"
NS = uuid.UUID("6f1c2f7e-3c2b-4f5e-9a51-6d1f0c0b7a10")
HUB_OPERATOR = "NOT_APPLICABLE"


def fulfilment_for(transfer_id: str, secret: str) -> bytes:
    return hashlib.sha256(f"{secret}:{transfer_id}".encode()).digest()


def condition_for(fulfilment: bytes) -> bytes:
    return hashlib.sha256(fulfilment).digest()


def switch_id(run_id: str, transfer_id: str) -> str:
    """Switch transfer ids must be UUIDs and unique per run; derived deterministically for evaluation."""
    return str(uuid.uuid5(NS, f"{run_id}:{transfer_id}"))


def _fspiop_headers(source: str, destination: str) -> dict:
    return {
        "Content-Type": CT,
        "Accept": ACCEPT,
        "Date": formatdate(usegmt=True),
        "FSPIOP-Source": source,
        "FSPIOP-Destination": destination,
    }


# --- onboarding --------------------------------------------------------------------------------------------


class Hub:
    """Central-ledger admin API client (the hub operator's onboarding calls, as in Mojaloop's TTK provisioning)."""

    def __init__(self, admin_url: str):
        self.c = httpx.Client(base_url=admin_url, timeout=30, headers={"Content-Type": "application/json"})

    def _post(self, path: str, body: dict, ok_if_exists: bool = True, **headers) -> httpx.Response:
        r = self.c.post(path, json=body, headers=headers or None)
        if r.status_code >= 400 and not (ok_if_exists and "already" in r.text.lower()):
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
        return r

    def ensure_currency(self, currency: str) -> None:
        for kind in ("HUB_MULTILATERAL_SETTLEMENT", "HUB_RECONCILIATION"):
            self._post("/participants/Hub/accounts", {"type": kind, "currency": currency})

    def onboard(self, fsp: str, currency: str, callback_base: str, net_debit_cap: float) -> None:
        self._post("/participants", {"name": fsp, "currency": currency})
        self._post(
            f"/participants/{fsp}/initialPositionAndLimits",
            {"currency": currency, "limit": {"type": "NET_DEBIT_CAP", "value": net_debit_cap}, "initialPosition": 0},
        )
        for kind, path in (
            ("FSPIOP_CALLBACK_URL_TRANSFER_POST", "/transfers"),
            ("FSPIOP_CALLBACK_URL_TRANSFER_PUT", "/transfers/{{transferId}}"),
            ("FSPIOP_CALLBACK_URL_TRANSFER_ERROR", "/transfers/{{transferId}}/error"),
        ):
            self._post(f"/participants/{fsp}/endpoints", {"type": kind, "value": callback_base + path}, False)
        accounts = self.c.get(f"/participants/{fsp}/accounts", headers={"FSPIOP-Source": HUB_OPERATOR}).json()
        settlement = next(a for a in accounts if a["ledgerAccountType"] == "SETTLEMENT" and a["currency"] == currency)
        if abs(float(settlement["value"])) < net_debit_cap / 2:  # fund once: settlement balance covers the cap
            self._post(
                f"/participants/{fsp}/accounts/{settlement['id']}",
                {
                    "transferId": str(uuid.uuid4()),
                    "externalReference": "fraud-engine-replay",
                    "action": "recordFundsIn",
                    "reason": "liquidity for replayed traffic",
                    "amount": {"amount": f"{net_debit_cap:.0f}", "currency": currency},
                },
                False,
                **{"FSPIOP-Source": HUB_OPERATOR},
            )


# --- mock FSP callbacks --------------------------------------------------------------------------------------


class MockFsp:
    """Callback endpoints for every onboarded FSP: payee side fulfils, payer side records the outcome."""

    def __init__(self, switch_url: str, secret: str):
        self.switch_url, self.secret = switch_url, secret
        self.counts = {"prepares_received": 0, "fulfils_sent": 0, "fulfil_errors": 0, "committed": 0, "errors": 0}
        self.done = {}  # switch transfer id -> final state seen by the payer
        self.app = FastAPI()
        self.client: httpx.AsyncClient | None = None
        self.app.add_api_route("/transfers", self.on_prepare, methods=["POST"])
        self.app.add_api_route("/transfers/{tid}", self.on_result, methods=["PUT", "PATCH"])
        self.app.add_api_route("/transfers/{tid}/error", self.on_error, methods=["PUT"])

    async def on_prepare(self, request: Request) -> Response:
        body = orjson.loads(await request.body())
        self.counts["prepares_received"] += 1
        asyncio.get_running_loop().create_task(self._fulfil(body["transferId"], body["payerFsp"], body["payeeFsp"]))
        return Response(status_code=202)

    async def _fulfil(self, tid: str, payer: str, payee: str) -> None:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=64))
        body = {
            "fulfilment": b64url(fulfilment_for(tid, self.secret)),
            "completedTimestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "transferState": "COMMITTED",
        }
        try:
            r = await self.client.put(
                f"{self.switch_url}/transfers/{tid}", json=body, headers=_fspiop_headers(payee, payer)
            )
            ok = r.status_code < 300
        except httpx.HTTPError:
            ok = False
        self.counts["fulfils_sent" if ok else "fulfil_errors"] += 1

    async def on_result(self, tid: str, request: Request) -> Response:
        body = orjson.loads(await request.body() or b"{}")
        if body.get("transferState") == "COMMITTED" and tid not in self.done:
            self.done[tid] = "COMMITTED"
            self.counts["committed"] += 1
        return Response(status_code=200)

    async def on_error(self, tid: str, request: Request) -> Response:
        body = orjson.loads(await request.body() or b"{}")
        if tid not in self.done:
            self.done[tid] = (body.get("errorInformation") or {}).get("errorDescription", "error")
            self.counts["errors"] += 1
        return Response(status_code=200)

    def serve(self, port: int) -> uvicorn.Server:
        server = uvicorn.Server(uvicorn.Config(self.app, host="0.0.0.0", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started:
                return server
            time.sleep(0.05)
        raise RuntimeError(f"callback server did not start on port {port}")


# --- transfers -----------------------------------------------------------------------------------------------


def build_transfer(t: dict, run_id: str, secret: str, device_signals: bool) -> dict:
    """POST /transfers body for one internal transaction (ids mapped to switch UUIDs)."""
    tid = switch_id(run_id, t["transfer_id"])
    t = {**t, "transfer_id": tid, "quote_id": switch_id(run_id, t.get("quote_id") or t["transfer_id"])}
    if t.get("original_transfer_id"):
        t["original_transfer_id"] = switch_id(run_id, t["original_transfer_id"])
    transaction = fspiop_transaction(t)
    now = datetime.now(UTC)
    expires = now + timedelta(minutes=5)  # the wire expiry is real time; the event time travels separately
    condition = condition_for(fulfilment_for(tid, secret))
    alias = "".join(ch for ch in str(t.get("payee_alias") or t["payee_id"]) if ch.isalnum())
    ilp = encode_ilp_prepare(
        transaction,
        amount=int(round(float(t["amount"]) * 100)),
        destination=f"g.{t['payee_fsp']}.msisdn.{alias}",
        expires_at=expires.strftime("%Y%m%d%H%M%S") + f"{expires.microsecond // 1000:03d}",
        condition=condition,
    )
    ext = fspiop_extensions(t, device_signals=device_signals)
    ext.append({"key": EVENT_TIME_EXT, "value": _iso(int(t["ts"]))})
    return {
        "transferId": tid,
        "payerFsp": t["payer_fsp"],
        "payeeFsp": t["payee_fsp"],
        "amount": transaction["amount"],
        "ilpPacket": ilp,
        "condition": b64url(condition),
        "expiration": expires.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "extensionList": {"extension": ext},
    }


async def _send_all(events, colv, a, run_id: str, secret: str, mock: MockFsp) -> dict:
    s = get_settings()
    r = get_redis(timeout=10)
    counts = {"sent": 0, "accepted": 0, "rejected": 0, "http_errors": 0, "reports": 0}
    rejections: dict[str, int] = {}
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=a.concurrency)) as client:

        async def post(body: dict) -> None:
            async with sem:
                try:
                    resp = await client.post(
                        f"{a.switch_url}/transfers",
                        content=orjson.dumps(body),
                        headers=_fspiop_headers(body["payerFsp"], body["payeeFsp"]),
                    )
                except httpx.HTTPError:
                    counts["http_errors"] += 1
                    return
                if resp.status_code == 202:
                    counts["accepted"] += 1
                else:
                    counts["rejected"] += 1
                    key = f"{resp.status_code} {resp.text[:120]}"
                    rejections[key] = rejections.get(key, 0) + 1

        def in_flight() -> int:  # accepted by the switch but not yet committed or failed
            return counts["accepted"] - mock.counts["committed"] - mock.counts["errors"]

        tasks = set()
        t0 = next_at = time.perf_counter()
        for _ts, kind, ev in events:
            while kind == 1 and a.max_in_flight and in_flight() >= a.max_in_flight:
                await asyncio.sleep(0.02)  # closed loop: never outrun the switch's ledger
            now = time.perf_counter()
            if next_at > now:
                await asyncio.sleep(next_at - now)
            next_at = max(next_at, now) + 1 / a.tps
            if kind == 0:  # fraud-registry report: not FSPIOP traffic, straight onto the stream
                part = partition_of(ev["account_id"], s.stream_partitions)
                fields = {"event": orjson.dumps(ev), "sent_at": f"{time.time() * 1000:.3f}"}
                r.xadd(stream_key(part), fields, maxlen=s.stream_maxlen, approximate=True)
                counts["reports"] += 1
                continue
            t = {c: colv[c][ev] for c in TX_COLS}
            task = asyncio.create_task(post(build_transfer(t, run_id, secret, not a.no_device_signals)))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            counts["sent"] += 1
            if counts["sent"] % max(1, int(a.tps * 30)) == 0:
                el = time.perf_counter() - t0
                log.info("sender progress", rate=round(counts["sent"] / el, 1), in_flight=in_flight(), **counts)
        await asyncio.gather(*tasks)
        counts["seconds"] = round(time.perf_counter() - t0, 1)
    counts["rejections"] = rejections
    return counts


def run(a) -> dict:
    s = get_settings()
    setup_logging(s.log_level)
    run_id = a.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    secret = uuid.uuid4().hex
    tx, rep = build_timeline(a.data or s.data_dir, a.from_day, a.to_day, a.limit)
    fsps = sorted(set(tx["payer_fsp"]) | set(tx["payee_fsp"]))
    callback_base = f"http://{a.callback_host}:{a.callback_port}"
    log.info("onboarding FSPs", fsps=len(fsps), currency=a.currency, callbacks=callback_base)
    hub = Hub(a.admin_url)
    hub.ensure_currency(a.currency)
    for f in fsps:
        hub.onboard(f, a.currency, callback_base, a.net_debit_cap)
    mock = MockFsp(a.switch_url, secret)
    server = mock.serve(a.callback_port)

    gt = tx[["transfer_id", "is_fraud", "fraud_type", "case_id"]].copy()
    gt["transfer_id"] = [switch_id(run_id, x) for x in gt["transfer_id"]]
    write_ground_truth(gt)

    colv = {c: tx[c].tolist() for c in TX_COLS}
    colv["currency"] = [a.currency] * len(tx)
    events = [(colv["ts"][i], 1, i) for i in range(len(tx))] + [
        (int(t), 0, {"type": "fraud_report", "account_id": acc}) for acc, t in zip(rep["account_id"], rep["report_ts"])
    ]
    events.sort(key=lambda e: (e[0], e[1]))
    log.info(
        "sending", run_id=run_id, transfers=len(tx), reports=len(rep), tps=a.tps, device_signals=not a.no_device_signals
    )
    counts = asyncio.run(_send_all(events, colv, a, run_id, secret, mock))

    deadline = time.time() + a.settle_timeout  # let the switch finish fulfils and notifications
    while time.time() < deadline and mock.counts["committed"] + mock.counts["errors"] < counts["accepted"]:
        time.sleep(1)
    server.should_exit = True
    switch_errors: dict[str, int] = {}
    for state in mock.done.values():
        if state != "COMMITTED":
            switch_errors[state] = switch_errors.get(state, 0) + 1
    summary = {
        "run_id": run_id,
        "window_days": [a.from_day, a.to_day],
        "device_signals": not a.no_device_signals,
        "transfers": len(tx),
        "frauds": int(tx["is_fraud"].sum()),
        **counts,
        "switch": {**mock.counts, "final_errors": switch_errors},
    }
    log.info("sender finished", **{k: v for k, v in summary.items() if k != "rejections"})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-day", type=int, default=53)
    ap.add_argument("--to-day", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tps", type=float, default=40, help="maximum send rate")
    ap.add_argument(
        "--max-in-flight",
        type=int,
        default=250,
        help="pause while this many transfers are unfinished in the switch (0 = send at --tps regardless)",
    )
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--no-device-signals", action="store_true", help="plain FSPIOP: no device/IP/geo extensions")
    ap.add_argument("--switch-url", default="http://localhost:3000", help="ml-api-adapter (FSPIOP API)")
    ap.add_argument("--admin-url", default="http://localhost:3001", help="central-ledger admin API")
    ap.add_argument(
        "--callback-host",
        default="host.docker.internal",
        help="how the switch's containers reach this process (Linux Docker: the network gateway IP)",
    )
    ap.add_argument("--callback-port", type=int, default=4041)
    ap.add_argument("--currency", default="INR")
    ap.add_argument("--net-debit-cap", type=float, default=10_000_000_000)
    ap.add_argument("--settle-timeout", type=float, default=120)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the run summary JSON here")
    a = ap.parse_args(argv)
    print(orjson.dumps(run(a), option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
