# Live Mojaloop integration

The engine can score the transfers of a real Mojaloop switch. It reads the switch's Kafka topic
`topic-transfer-prepare` with its own consumer group and feeds each transfer into the existing scoring pipeline.

## How it works

```
Payer FSP ──POST /transfers──► ml-api-adapter
                                    │
                                    ▼
                    Kafka: topic-transfer-prepare
                       │                      │
         group: central-ledger         group: fraud-engine
                       ▼                      ▼
              transfer processing    pipeline/mojaloop_consumer.py
              (unchanged)                     │  decode data-URI payload + ILP packet,
                                              │  validate via from_fspiop()
                                              ▼
                               Redis stream fsp:transfers:<payer partition>
                                              ▼
                        scoring workers → alerts → dashboard   (unchanged)
```

Kafka gives every consumer group its own copy of the topic, so the engine sees every transfer while the switch
processes it exactly as before.

| File | Role |
|---|---|
| `pipeline/mojaloop_consumer.py` | Kafka consumer → Redis stream. Commits offsets only after the batch is on the stream (at-least-once; scoring is idempotent by transferId). Bad events → `mojaloop:dlq`; non-prepare events skipped. Counters in the Redis hash `mojaloop_consumer`. |
| `pipeline/ilp.py` | Decodes the binary ILP packet in `ilpPacket` (ILPv4 Prepare and ILPv1 Payment, OER encoding) to get the FSPIOP `Transaction`: payer, payee, transaction type. Also accepts the simulator's mocked packet. |
| `pipeline/mojaloop_adapter.py` | `decode_payload` (data-URI payloads), `is_transfer_prepare`, `normalize_message`, `from_fspiop` → internal schema. |
| `docker-compose.mojaloop.yml` | Adds the `mojaloop-consumer` service on the switch's `mojaloop-net` network. |
| `tests/fixtures/mojaloop_transfer_prepare.json` | A transfer-prepare event captured from a live switch, used by the unit tests. |

## What a real event looks like

Captured from `ml-core-test-harness` (FSPIOP 2.0):

* `metadata.event` is `{"type": "prepare", "action": "prepare", "createdAt": ...}`.
* `content.payload` is a data URI:
  `data:application/vnd.interoperability.transfers+json;version=2.0;base64,<POST /transfers body>`.
* The body's `ilpPacket` is an ILPv1 Payment packet (type 1). Its data field is the base64url JSON `Transaction`,
  which gives the MSISDN identities of payer and payee.
* There is no device, IP or geo data. These features stay empty unless FSPs put them in the transfer's
  `extensionList` (keys `deviceId`, `ipAddress`, `geoLat`, `geoLon`, `channel`, `deviceBoundAt`,
  `merchantCategory`).

## Running it against a Mojaloop sandbox

Needs Docker and about 12 GB of free RAM for the switch.

1. Start the switch and onboard the test FSPs:
   ```bash
   git clone https://github.com/mojaloop/ml-core-test-harness.git
   ```
   ```bash
   cd ml-core-test-harness && docker compose --profile all-services --profile ttk-provisioning up -d
   ```
   The harness binds host ports such as 6379 (its Redis), so give the engine's own Redis another port if both run
   on one machine.
2. Start the engine with the overlay (it joins the `mojaloop-net` network and reads `kafka:29092`):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.mojaloop.yml up -d --build
   ```
   Running natively instead: map `kafka` to the Kafka container's IP in `/etc/hosts` (the broker advertises
   `kafka:29092`), then run `python -m pipeline.mojaloop_consumer --bootstrap kafka:29092`.
3. Send transfers. Mojaloop's standard P2P test, from the harness directory:
   ```bash
   docker compose --profile all-services --profile ttk-provisioning --profile ttk-tests up --no-deps ttk-tests
   ```
   or use the testing toolkit UI at http://localhost:9660 (mobile simulator: `/mobilesimulator`).
4. Check: the transfers appear on the dashboard's Live Alerts / Investigation views and in
   `redis-cli hgetall mojaloop_consumer`. Rejected events are in `redis-cli xrange mojaloop:dlq - +`.

Settings (`.env`): `KAFKA_BOOTSTRAP_SERVERS`, `MOJALOOP_PREPARE_TOPIC` (default `topic-transfer-prepare`),
`MOJALOOP_CONSUMER_GROUP` (default `fraud-engine`), `MOJALOOP_OFFSET_RESET` (`latest` | `earliest`, where a new
group starts). Captured events can be replayed without Kafka:
`python -m pipeline.mojaloop_consumer --from-file events.jsonl` (one Kafka message value per line).

## Live test (2026-10-25)

Run against `mojaloop/ml-core-test-harness` (all-services + TTK provisioning: 100/100 assertions), with the
engine running natively (models trained on a 200k-transaction synthetic dataset) and the consumer reading
`kafka:29092`:

| Check | Result |
|---|---|
| Mojaloop P2P test (`ttk-tests`, 31 assertions each) | 4 runs, all 100% passed: transfers fulfilled by the switch while the engine read alongside it |
| Events consumed | 4 forwarded, 0 invalid, 0 skipped; consumer-group lag 0 |
| Scored and stored | 4/4 in `transactions`, payer `MSISDN:44123456789` → payee `MSISDN:27713803912` (from the real ILP packet), decision ALLOW, risk ≤ 0.0001, scoring latency 4.5–9.4 ms, SHAP explanation stored |
| API | `GET /v1/transactions/{id}` returns the live transfers |
| Restart | Consumer stopped, one transfer sent (lag 1), consumer restarted: the transfer was picked up and scored, none lost |

Found by the live test and fixed: ml-api-adapter tags prepare events `type: "prepare"`, not `"transfer"`; the
first version of the event filter would have skipped every real transfer.

Not tested here: the `docker-compose.mojaloop.yml` overlay (validated with `docker compose config`; the app
image was not built in this environment). The low risk scores show the model working on thin signals: with no
device or history data, an ordinary 100-unit transfer looks safe; this says nothing about how well it would catch
real fraud. Fraud detection through the switch, with and without device extensions and at volume, is measured in
[mojaloop_fraud_test.md](mojaloop_fraud_test.md).

## Operating notes

* **Shadow mode.** The engine scores and alerts but never answers the switch, so it cannot delay or block a
  transfer. To block, a payer FSP calls `POST /v1/score/fspiop` before it sends, or a payee FSP rejects the
  transfer with an error callback.
* **Retrain before trusting scores.** The models were trained on synthetic data with device/geo signals that real
  FSPIOP traffic does not carry. Keep it in shadow mode, label alerts through the dashboard and let the feedback
  loop retrain on real transfers before any blocking decision relies on it.
* **Identities.** Accounts are keyed `MSISDN:<number>` (from the ILP Transaction). If a packet cannot be decoded
  the adapter falls back to FSP-level identities (`fsp:<fspId>`), which weakens per-account features.
* **Not covered yet:** bulk transfers (`topic-bulk-prepare`), FX transfers (`fx-prepare`), and settlement-window
  events are skipped.
