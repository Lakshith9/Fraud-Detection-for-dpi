# Fraud test through a live Mojaloop switch

The integration test in [mojaloop_integration.md](mojaloop_integration.md) proved the plumbing with Mojaloop's
own test transfers (all legitimate). This test measures **fraud detection on traffic that travels through the
switch**: the labelled synthetic test window, including its frauds, is replayed as real FSPIOP transfers.

## How it works

```
simulator/mojaloop_sender.py                         Mojaloop (ml-core-test-harness)
  as payer FSP:  POST /transfers  ─────────────────►  ml-api-adapter ──► Kafka topic-transfer-prepare
                                                                          │                  │
  as payee FSP:  PUT /transfers/{id} (fulfil) ◄── callbacks ── central-ledger      pipeline/mojaloop_consumer.py
                                                  (positions,                      │
                                                   COMMITTED)                      ▼
                                                                    scoring workers → alerts → dashboard
scripts/evaluate_live.py: stored decisions vs ground truth  →  docs/live_eval/<run>.json
```

* **Onboarding.** The sender registers the dataset's 12 FSPs (`dfsp-01` … `dfsp-12`) with the switch through the
  central-ledger admin API: INR hub accounts, net-debit cap, settlement funds and callback endpoints.
* **Real transfers.** Each synthetic transaction becomes a `POST /transfers` with a real ILPv4 packet carrying the
  FSPIOP `Transaction`, a SHA-256 condition and FSPIOP headers. The sender also answers the switch's callbacks
  as the payee FSP (fulfilment), so transfers complete as `COMMITTED`.
* **Identities.** Payer and payee keep the dataset's account ids (`partySubIdOrType`), so the engine continues
  each account's history from the day-46 bootstrap snapshot. The dashboard therefore shows `cust-…` ids, as with
  the direct simulator.
* **Event time.** The switch stamps the current time on every transfer. The original event time travels in an
  `eventTime` extension, which the engine uses only with `TRUST_FSP_EVENT_TIME=true` (replay only; a live
  deployment keeps it off so an FSP cannot shift its own time features).
* **Device signals.** By default device, IP, geo and channel travel in the FSPIOP `extensionList`, as an FSP that
  shares them would send. `--no-device-signals` sends plain FSPIOP (what a switch sees when FSPs share nothing).
* **Fraud reports** (registry feed) are not FSPIOP traffic; they go straight onto the scoring stream, as in the
  simulator.
* **Pacing.** At most `--tps`, and never more than `--max-in-flight` transfers unfinished in the switch, so the
  replay adapts to the switch's speed and no transfer expires. `--max-in-flight 0` sends at `--tps` regardless
  (the engine keeps up; the test switch's ledger may not, see the results).

## Results (2026-10-28)

Setup: `ml-core-test-harness` (FSPIOP 2.0 switch, 28 containers) and the engine running natively on one 4-CPU,
15 GB machine; model v1 trained on the 200k-transaction synthetic dataset (days before 46), engine state
bootstrapped at day 46. Every run starts from a fresh bootstrap. "Direct" is the same window replayed by
`simulator.simulator` straight onto the scoring stream, the baseline the switch runs are compared with. Full JSON
per run is in [live_eval/](live_eval/).

### 1. Every transfer completed by the switch (days 46–51)

The sender paced itself to the switch (`--max-in-flight 250`, about 14 transfers/s): **15,182 of 15,182
transfers COMMITTED** (0 rejected, 0 expired, 0 HTTP errors), with fraud scoring identical to the direct path:

| | Direct | Through the switch (all committed) |
|---|---|---|
| Transfers scored | 15,182 / 15,182 | 15,182 / 15,182 |
| Frauds in window | 57 | 57 |
| PR-AUC | 0.8722 | 0.8654 |
| ROC-AUC | 0.9919 | 0.9925 |
| Recall at 1% false-positive rate | 94.7% | 94.7% |
| Alerts (REVIEW + BLOCK) | 124 | 125 |
| Frauds caught (REVIEW or BLOCK) | 51 (89.5%) | 51 (89.5%) |
| False alarms | 73 (0.5% of legit) | 74 (0.5% of legit) |
| Alert precision | 41.1% | 40.8% |
| Blocked frauds | 43 (75.4%) | 42 (73.7%) |
| Block precision | 84.3% | 84.0% |
| Caught: identity spoofing | 27/28 (96.4%) | 27/28 (96.4%) |
| Caught: transaction laundering | 24/29 (82.8%) | 24/29 (82.8%) |
| Scoring latency p50 / p99 | 4.25 / 8.9 ms | 7.08 / 30.81 ms |

### 2. Device signals shared vs plain FSPIOP (days 46–60, 60 transfers/s)

| | Direct | Through the switch, device extensions |
|---|---|---|
| Transfers scored | 54,124 / 54,124 | 54,095 / 54,124 |
| Frauds in window | 144 | 144 |
| PR-AUC | 0.7765 | 0.7524 |
| ROC-AUC | 0.9842 | 0.9843 |
| Recall at 1% false-positive rate | 89.6% | 86.8% |
| Alerts (REVIEW + BLOCK) | 538 | 525 |
| Frauds caught (REVIEW or BLOCK) | 124 (86.1%) | 120 (83.3%) |
| False alarms | 414 (0.8% of legit) | 405 (0.8% of legit) |
| Alert precision | 23.1% | 22.9% |
| Blocked frauds | 104 (72.2%) | 103 (71.5%) |
| Block precision | 72.7% | 73.6% |
| Caught: identity spoofing | 59/74 (79.7%) | 58/74 (78.4%) |
| Caught: settlement manipulation | 7/7 (100.0%) | 7/7 (100.0%) |
| Caught: transaction laundering | 58/63 (92.1%) | 55/63 (87.3%) |
| Scoring latency p50 / p99 | 4.33 / 10.01 ms | 8.86 / 49.62 ms |

* **With device extensions the switch path matches the direct path**: 120 vs 124 of 144 frauds caught at the same
  0.8 % false-alarm rate and the same block precision. The small gap likely comes from the reordering of
  concurrent sends and graph-refresh timing (run-to-run variance was not measured); the switch adds latency (p99 50 ms vs 10 ms, on a machine
  also running the switch).
* **The switch's ledger was the bottleneck, not the engine.** At 60 transfers/s the fraud engine's consumer lag
  stayed at 0–4 messages, while the harness's single central-ledger prepare handler sustains about 12 transfers/s
  on this machine: most of these transfers expired in the switch's queue before the ledger reached them
  (scoring is unaffected, since the engine reads each transfer as it enters the switch). Run 1 shows that paced
  traffic completes in full. 29 of 54,124 POSTs timed out under that load and never entered the switch.
* **Plain FSPIOP (no device extensions): not measured to completion.** After the two fixes below, a partial run
  scored without errors but flagged about two thirds of transfers (REVIEW/BLOCK) in its first 2,000. The model
  was trained only on data where every transfer has a device, so it never learned what a missing device means.
  For switches whose FSPs share no device data, train a model without device features (the same need as for
  PaySim). The run then stopped because the test machine ran out of disk (Kafka log growth over the earlier
  runs).

### Bugs found by this test (fixed)

1. **Missing device data looked like a device farm.** Without device extensions every transfer carried device id
   `""`, which the feature store treated as one device shared by thousands of accounts (`device_accounts` 3,547,
   `device_tx_1h` 132): 14.8 % of legitimate transfers were flagged. A missing device or IP is now unknown (NaN)
   and never stored (`pipeline/features.py`); a sequence step without a device is no longer a "device change".
2. **Explanations crashed on unknown values.** `models/explain.py` formatted feature values with `int()`, which
   fails on NaN; with fix 1 in place, 31,628 transfers failed to score. Every reason now handles unknown values
   ("device not reported by the FSP"), with a test over all feature names.
3. **FSPIOP amount format.** The simulator wrote `"500.00"`; FSPIOP amounts allow no trailing zeros, so a real
   switch rejects them. Amounts are now `"500"`.


## Running it on your machine

Needs Docker (about 8 GB free RAM for the switch plus the engine), the engine set up as in the README, and its
Redis on a port other than 6379 (the harness uses 6379).

1. **Configure the switch.** In your `ml-core-test-harness` checkout, use the override from this repo, which
   publishes Kafka (9092), the FSPIOP API (3000) and the admin API (3001), and allows same-FSP ("on-us")
   transfers:
   ```bash
   cp <fraud-engine>/docker/mojaloop-harness.override.yml docker-compose.override.yml
   ```
   On Windows PowerShell: `Copy-Item D:\fraud-engine\docker\mojaloop-harness.override.yml docker-compose.override.yml`
2. **Start the switch** and wait for `ttk-provisioning` to exit with code 0:
   ```bash
   docker compose --profile all-services --profile ttk-provisioning up -d
   ```
3. **Engine settings.** In the engine's `.env`: `TRUST_FSP_EVENT_TIME=true` and `REDIS_URL` on your engine Redis
   port (e.g. `redis://127.0.0.1:6380/0`).
4. **Start the engine from a clean state**, then the consumer (separate terminals):
   ```bash
   python scripts/tasks.py bootstrap
   python scripts/tasks.py run
   python -m pipeline.mojaloop_consumer --bootstrap localhost:9092
   ```
   and the dashboard: `cd dashboard && npm run dev` (Windows: `npm.cmd run dev`).
5. **Send the test window through the switch** (days 46–51: 15,182 transfers, 57 frauds; about 20 minutes):
   ```bash
   python -m simulator.mojaloop_sender --from-day 46 --to-day 51 --out docs/live_eval/my_run_switch.json
   ```
   On Docker Desktop (Windows/macOS) the switch reaches the sender's callbacks at `host.docker.internal` (the
   default). On Linux Docker add `--callback-host 172.18.0.1` (the `mojaloop-net` gateway:
   `docker network inspect mojaloop-net -f '{{(index .IPAM.Config 0).Gateway}}'`). Windows may ask to allow
   Python through the firewall for port 4041: allow it.
6. **Watch** Live Alerts on the dashboard while it runs: REVIEW/BLOCK alerts appear as frauds pass through the
   switch.
7. **Score the run** against the ground truth:
   ```bash
   python -m scripts.evaluate_live --label my_run --out docs/live_eval/my_run.json
   python -m scripts.live_report docs/live_eval/my_run.json
   ```
   The sender's summary (`my_run_switch.json`) shows how many transfers the switch committed.

To repeat a run, stop the engine, run `python scripts/tasks.py bootstrap` again (it resets state and ground
truth), and start again from step 4.
