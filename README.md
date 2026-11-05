# Real-Time Fraud Detection Engine for DPI Payment Rails

**Bracket 2 · AI-for-DPI · Project 5.** A real-time fraud detection system for open payment networks built
on Mojaloop / UPI-like rails. It scores every transfer before settlement with an ML ensemble (XGBoost + LSTM +
graph analysis), detects fraud rings asynchronously, explains every decision with SHAP, and retrains from operator
feedback behind a promotion gate.

> **Important — synthetic data; live Mojaloop connection available.** The demo traffic and every reported
> metric come from **synthetic Mojaloop-style data**: the simulator emits FSPIOP *transfer-prepare* messages (the
> shape of Mojaloop `topic-transfer-prepare` events). The engine can also read a **real Mojaloop switch**:
> `pipeline/mojaloop_consumer.py` consumes the switch's Kafka topic, decodes real ILP packets and feeds the same
> pipeline (see [docs/mojaloop_integration.md](docs/mojaloop_integration.md)); this was tested against Mojaloop's
> `ml-core-test-harness`, not a production switch. Metrics are **not** a prediction of real-world performance (see
> [docs/model_card.md](docs/model_card.md)).

## What it detects

| Typology | Signals used |
|---|---|
| Identity spoofing / account takeover | fresh device binding (SIM swap), new device/IP, geo jump, night-time, drain amounts, new mule payees, sequence anomalies; device cloning (drift variant) |
| Settlement manipulation | round/structured amounts just below thresholds, timing near settlement cutoffs, reversals, FSP-level round-high-value anomaly z-scores, reciprocal bursts (graph) |
| Merchant fraud rings | dense subgraphs (k-core + Louvain), shared devices/IPs, mutual payer↔merchant flows, fraud-neighbour ratio |
| Transaction laundering | temporal cycles A→B→C→A, fan-out → pass-through → fan-in mule networks, pass-through ratio, inflow/outflow timing |

## Architecture

```mermaid
flowchart LR
  SIM[Simulator<br/>FSPIOP events] -->|Redis Streams<br/>partitioned by payer| W[Scoring workers]
  PSP[PSP / switch] -->|POST /v1/score| API[FastAPI]
  W & API <-->|2 pipelined round trips| R[(Redis<br/>online features<br/>+ graph cache)]
  W & API -->|persist:rows stream| PER[Persister<br/>+ async SHAP] --> PG[(PostgreSQL)]
  PG --> GW[Graph worker<br/>NetworkX rings] -->|node scores| R
  W -->|pub/sub| API -->|WebSocket| UI[React dashboard]
  UI -->|feedback| API --> PG --> RT[Retrainer + gate] --> REG[(Model registry)] -->|hot reload| W & API
```

The full component diagram, latency budget, failure modes and parity design are in
[docs/architecture.md](docs/architecture.md). Every feature is documented in
[docs/feature_dictionary.md](docs/feature_dictionary.md).

**Per-transfer hot path:**
1. One pipelined Redis read: features, graph scores, idempotency.
2. Feature compute with the shared module.
3. XGBoost.
4. ONNX LSTM under a timeout.
5. Cached graph lookup.
6. Stacked logistic meta-learner and decision.
7. SHAP for REVIEW/BLOCK.
8. One pipelined Redis write.

PostgreSQL persistence (a separate persister process) and graph detection run off the hot path. If the sequence model times out or the graph
cache is stale, the transfer is scored by a calibrated fallback combo and marked `degraded=true`.

## Repository layout

```
fraud-engine/
  docker-compose.yml  Makefile  README.md  .env.example  requirements.txt  pyproject.toml
  core/        settings, structured logging, PostgreSQL schema/pool, Redis client
  data_gen/    synthetic Mojaloop-style data generator (entities, typologies, hard negatives, drift)
  pipeline/    Mojaloop adapter + ILP decoder + live Kafka consumer, feature engine + state stores, replay, scorer, stream worker, persister, bootstrap
  models/      training, LSTM/ONNX, ensemble, SHAP explanations, metrics, registry, feedback retraining
  graph/       ring detectors, node scoring, graph worker, ring evaluation, optional Neo4j sink
  api/         FastAPI app (REST + WebSocket), auth, metrics, routes
  dashboard/   React + Vite + TypeScript + Tailwind operator console
  simulator/   Mojaloop-style event simulator (TPS control, duplicates, out-of-order)
  tests/       unit/, integration/ (end-to-end), load/ (benchmark)
  docs/        architecture.md, model_card.md, feature_dictionary.md, results.md, benchmarks.json, feedback_demo.json
  scripts/     tasks.py (task runner), feedback_demo.py, make_report.py
  docker/      Dockerfiles + nginx config
```

## Quick start (Docker)

```bash
cp .env.example .env        # then set OPERATOR_API_KEYS / VITE_API_KEY (generate keys as described in the file)
docker compose --profile demo up -d --build
```

> **Verified** on Windows 11 with Docker Desktop (WSL 2, 16 GB RAM): all containers start, the dashboard works on
> :8080 and the simulator's alerts appear. The first build plus 1M-transaction training took about an hour. It needs
> about 15 GB of free disk where Docker keeps its disk image (Docker Desktop: Settings → Resources → Advanced → Disk
> image location; move it off a nearly full system drive) and about 8–10 GB of memory for Docker during training.
> The stack uses ports 5432, 6379, 8000 and 8080, so stop any local PostgreSQL/Redis/API first.

The one-shot `init` service generates 1M transactions, replays features, trains and registers the models, and
bootstraps Redis/Postgres; this takes about 15–25 minutes on a laptop. Then the API, two scoring workers, the
graph worker, the retrain scheduler, the dashboard and the simulator (200 TPS, looping) start. Set `DEMO_N_TX=200000`
in `.env` for a faster first start.

* Dashboard: <http://localhost:8080> — live alerts, investigation, ring graph, cases, feedback, health
* API docs (OpenAPI): <http://localhost:8000/docs>

`make demo` runs the same command. `make down` stops everything.

## Run without Docker (how it was built and measured)

Requirements: Python 3.11, Node 20+, a Redis ≥ 6.2 and a PostgreSQL ≥ 14 reachable through `REDIS_URL` /
`DATABASE_URL` in `.env`.

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
cd dashboard && npm ci && cd ..
python scripts/tasks.py demo --tps 200                 # prepare + bootstrap + all services + simulator + browser
```

`scripts/tasks.py demo` is what `make local-demo` calls. It is idempotent: data, features and models are reused when
present (`--force` rebuilds them). Individual steps:

```bash
python -m data_gen.generate --n-transactions 1000000 --seed 42 --out data
python -m pipeline.replay --data data --out data/features --snapshot-day 46
python -m models.train --features data/features --data data --register --promote
python -m graph.evaluate --data data --out models/artifacts/graph_eval.json
python -m pipeline.bootstrap --features data/features --data data --reset
uvicorn api.main:app --port 8000
python -m pipeline.worker --partition 0            # one per STREAM_PARTITIONS
python -m pipeline.persister
python -m graph.worker
python -m models.retrain --schedule
python -m simulator.simulator --tps 200 --loop
cd dashboard && npm run dev                          # http://localhost:5173
```

## Live Mojaloop connection

`python -m pipeline.mojaloop_consumer` reads a Mojaloop switch's Kafka topic `topic-transfer-prepare` (own consumer
group, so the switch is unaffected) and feeds the scoring pipeline; with Docker, add
`-f docker-compose.mojaloop.yml` to join the switch's `mojaloop-net` network. Tested against Mojaloop's
`ml-core-test-harness`: real P2P transfers were decoded (including the binary ILP packet), scored in 4.5–9.4 ms
and shown via the API. Setup, message format and results: [docs/mojaloop_integration.md](docs/mojaloop_integration.md).

## Demo scripts, tests and benchmarks

These need the system running for `bench` and `feedback-demo`.

```bash
python -m pytest tests/unit                           # unit + train/serve parity (needs Redis for parity tests)
python -m pytest tests/integration -s                 # end-to-end on isolated Redis DB 15 / database fraud_test
python -m tests.load.bench                            # 200 TPS x 120 s, 1000 TPS x 60 s stress, sync API -> docs/benchmarks.json
python -m scripts.feedback_demo                       # replay feedback window, label alerts, retrain, before/after
python -m scripts.make_report                         # regenerate docs/results.md from the run artifacts
python -m ruff check . && (cd dashboard && npm run build)
```

## API

All endpoints except `/healthz` and `/metrics` require `X-API-Key: <operator key>` (WebSocket: `?api_key=`).

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/score` | Pre-settlement scoring → `{transfer_id, risk_score, decision, top_reasons, model_version, latency_ms, degraded, …}` |
| POST | `/v1/score/fspiop` | Same, accepting a Mojaloop transfer-prepare message or a bare `POST /transfers` body |
| GET | `/v1/alerts` | Alerts with filters (`decision`, `fraud_type`, `fsp`, `status`, `min_score`, paging) |
| GET | `/v1/transactions/{id}` | Details, payer/payee history, linked entities, rings, feedback, cases |
| GET | `/v1/transactions/{id}/explanation` | SHAP explanation (stored, or computed on demand for ALLOW) |
| GET | `/v1/rings`, `/v1/rings/{id}` | Detected fraud rings with evidence and edges |
| POST/GET/PATCH | `/v1/cases`, `/v1/cases/{id}` | Case management with notes and audit trail |
| POST/GET | `/v1/feedback` | True fraud / false positive / missed fraud (false negative) |
| GET | `/v1/models` | Registry, production version, retraining runs |
| POST | `/v1/models/retrain` | Trigger feedback retraining (separate process, gated) |
| POST | `/v1/models/{version}/promote`, `/v1/models/rollback` | Promotion / rollback (workers hot-reload) |
| GET | `/v1/metrics`, `/metrics` | Throughput, latency percentiles, alert rate, PSI drift, feedback (JSON / Prometheus) |
| GET | `/healthz` | Redis, PostgreSQL, model status |
| WS | `/ws/alerts` | Live alert + ring stream |

## Configuration reference (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Online feature store, graph cache, streams, pub/sub |
| `DATABASE_URL` | `postgresql://fraud:fraud@localhost:5432/fraud` | Transactions, alerts, rings, cases, feedback, registry |
| `OPERATOR_API_KEYS` | *(none)* | `operator:key,operator2:key2` — no default secrets |
| `VITE_API_KEY` | *(none)* | Key used by the dashboard build |
| `STREAM_PARTITIONS` | 2 | Stream partitions = scoring worker processes |
| `SEQUENCE_TIMEOUT_MS` | 25 | Sequence-model deadline before falling back |
| `GRAPH_INTERVAL_S` / `GRAPH_WINDOW_HOURS` / `GRAPH_STALE_AFTER_S` | 20 / 24 / 300 | Graph worker cadence, window, staleness for degraded mode |
| `GRAPH_BACKEND` | `networkx` | `neo4j` additionally mirrors rings to Neo4j (`NEO4J_URI/USER/PASSWORD`, needs `pip install neo4j`) |
| `MODEL_POLL_INTERVAL_S` | 5 | Hot-reload polling of `model:current` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` (`kafka:29092` in the Mojaloop overlay) | Live Mojaloop switch's Kafka |
| `MOJALOOP_PREPARE_TOPIC` / `MOJALOOP_CONSUMER_GROUP` / `MOJALOOP_OFFSET_RESET` | `topic-transfer-prepare` / `fraud-engine` / `latest` | Topic, consumer group, start position of a new group |
| `RETRAIN_INTERVAL_HOURS` / `RETRAIN_MIN_FEEDBACK` / `RETRAIN_TOLERANCE` | 24 / 50 / 0.01 | Scheduled retraining and promotion gate |
| `SETTLEMENT_CUTOFF_HOURS` | `[0,6,12,18]` | Settlement-window cutoffs (UTC) |
| `SHAP_SYNC_FOR_ALLOW` | false | Compute ALLOW explanations synchronously |
| `DEMO_N_TX`, `SIM_TPS` | 1000000, 200 | Docker demo dataset size and simulator rate |

## Measured results (from executed runs; full tables in [docs/results.md](docs/results.md))

All numbers below come from runs on this machine (Windows 11 laptop, Intel 12th-gen mobile CPU, 7.6 GB RAM, Python
3.11, Redis 7.0 inside WSL2, PostgreSQL 16, 4 scoring-worker partitions) on synthetic data.

**Model, held-out future window (days 46–60, ground truth).** The full ensemble reached:

| PR-AUC | ROC-AUC | Recall @1% FPR | BLOCK precision / recall | REVIEW precision / recall |
|---|---|---|---|---|
| 0.8926 | 0.9924 | 0.9357 | 0.7006 / 0.8677 | 0.2782 / 0.9258 |

The ablation (XGB only → + LSTM → + graph) is in the model card. Against the observed noisy labels, PR-AUC was
0.8321.

| Fraud type | Recall @ REVIEW | Recall @ BLOCK |
|---|---|---|
| Identity spoofing | 0.8827 | 0.7908 |
| Merchant fraud ring | 0.8524 | 0.8143 |
| Settlement manipulation | 1.0000 | 1.0000 |
| Transaction laundering | 0.9754 | 0.8893 |

**Graph ring detection vs planted rings.**

| Ring score threshold | Precision | Recall |
|---|---|---|
| ≥ 0.5 | 0.7647 | 0.6509 |
| ≥ 0.7 | 0.9508 | 0.5377 |

**Latency (final benchmark, `docs/benchmarks.json`).**

| Phase | Scoring p50 / p95 / p99 | End-to-end p99 | Errors |
|---|---|---|---|
| Stream, 200 TPS × 120 s (24,014 events) | 3.57 / 5.31 / **10.96 ms** | 42.6 ms | 0 |
| Stress, 1000 TPS offered × 60 s | 5.03 / 8.96 / 15.44 ms | 47 s (queueing) | 0 |
| Sync `POST /v1/score` over HTTP (2,000 requests) | server 3.69 / 5.17 / 7.12 ms | client p99 12.65 ms | 0 |

* **Stress phase.** On this laptop the 4 workers saturate at **≈ 540–620 TPS**. At 1000 TPS offered the backlog
  grows, so end-to-end latency reflects queueing, while per-transfer scoring stays < 16 ms p99.
* **Per-stage breakdown.** See [docs/architecture.md](docs/architecture.md) (latency budget table) and
  `docs/latency_breakdown.json`.
* **Run-to-run variance.** An immediately preceding full run measured 4.85 ms scoring p99 and 9.2 ms end-to-end p99
  at 200 TPS. Its JSON was overwritten by the final run, so only the final run's numbers are claimed above.

**Feedback loop (v1 → v2).** Operators labelled 1,192 alerts and reported 26 missed frauds, all through the API.
Retraining was gated on the untouched holdout (days 53–60) and the candidate was promoted. The workers hot-reloaded
v2 without restart.

| Holdout metric | v1 | v2 |
|---|---|---|
| PR-AUC (observed labels) | 0.8162 | 0.8652 |
| PR-AUC (truth) | 0.8960 | 0.9441 |
| BLOCK precision | 0.584 | 0.886 |

## Definition of Done — evidence

| # | Requirement | Evidence |
|---|---|---|
| 1 | Scores every transaction flowing through a Mojaloop-style flow within 100 ms; p50/p95/p99 reported | `tests/load/bench.py` → `docs/benchmarks.json` (200 TPS: scoring p99 10.96 ms, end-to-end p99 42.6 ms; HTTP sync p99 7.12 ms server / 12.65 ms client). `tests/integration/test_end_to_end.py` asserts p99 < 100 ms and that every streamed transfer is stored exactly once (with 2% duplicate deliveries and 1% out-of-order events) |
| 2 | ML ensemble + graph analysis catching fraud rings | `models/train.py` metrics and ablation (`models/artifacts/v1/metrics.json`); `graph/evaluate.py` → `models/artifacts/graph_eval.json`; `tests/unit/test_graph_detectors.py` (cycles, cliques, mule fan-in/out, shared devices, reciprocal bursts) |
| 3 | Explainable alerts (SHAP per scored transaction) | REVIEW/BLOCK explanations computed synchronously; ALLOW explanations asynchronously (persister) or on demand. `test_explanation_is_exactly_additive_in_logit_space`; the integration test checks 5 plain-language reasons; screenshot `docs/screenshots/investigation_shap.jpg` |
| 4 | Improves through operator feedback | `scripts/feedback_demo.py` → `docs/feedback_demo.json` (v1 → v2 promoted, holdout PR-AUC 0.8162 → 0.8652); the integration test exercises feedback → retrain → promote → hot reload |
| — | Train/serve parity | `tests/unit/test_feature_parity.py` (identical features from the offline replay and the Redis serving path) |
| — | Tests / lint / build | `docs/test_results.txt` (75 passed: 74 unit + 1 end-to-end integration); `ruff check .` clean; `npm run build` (tsc + vite) clean |
| — | Dashboard | Live alerts via WebSocket (`docs/screenshots/live_alerts.jpg`), investigation with SHAP chart + history timeline, ring graph (react-force-graph), cases with audit trail, feedback & model registry, system health (throughput, p50/p95/p99, alert rate, PSI drift, degraded mode, persister/graph status) |

## Deviations from the brief (and why)

| Brief | What was built | Reason |
|---|---|---|
| Redpanda/Kafka | **Redis Streams** (consumer groups, payer-hash partitions, pending-entry recovery, DLQ) | Redis was already required for the feature store; the build machine has 7.6 GB RAM and no Docker. The transport is isolated in `pipeline/worker.py` / `simulator/simulator.py` |
| Live Mojaloop | Live connection via `pipeline/mojaloop_consumer.py` (Kafka `topic-transfer-prepare`, real ILP decoding) and a labelled fraud replay through Mojaloop's `ml-core-test-harness`: 15,182 transfers, all committed, 53 of 57 frauds caught, both natively and in Docker ([docs/mojaloop_fraud_test.md](docs/mojaloop_fraud_test.md)). The headline model metrics above come from the offline synthetic evaluation | No production switch or real labelled transfers available |
| Neo4j | NetworkX (default, fully working); optional Neo4j mirror behind `GRAPH_BACKEND=neo4j` | No Neo4j server available. **The Neo4j sink was not executed in any run** |
| `docker compose up` verified from a fresh clone | Verified on Windows 11 / Docker Desktop: images build, `init` trains on 1M transactions, all services and the dashboard run | — (measured results were produced natively; see "Run without Docker") |
| `make` targets | Makefile is a thin wrapper over `scripts/tasks.py` and the module commands that were actually run | `make` is not installed on the Windows build machine |
| Isotonic/Platt calibration | Logistic stacking = Platt-style calibration | Keeps explanations exactly additive; isotonic would break additivity |

## Known limitations

* **Synthetic data.** Metrics are optimistic relative to real fraud (see the model card).
  Settlement-manipulation recall is 1.0, which says more about the generator than about real-world difficulty.
* **Thresholds drift.** On the future window the realised FPR exceeded the calibration budget (REVIEW 0.71% vs
  0.5%, BLOCK 0.11% vs 0.05%). Thresholds need periodic re-validation.
* **Graph signal is small.** It adds little to transaction-level PR-AUC; stealth merchant rings using only
  established accounts are mostly missed (5/10 found at score ≥ 0.5).
* **Graph refresh cadence.** The graph worker is wall-clock based (every 20 s over a 24 h event window). The
  training replay uses an event-time cadence (every 3 event-hours), a small train/serve difference for graph
  features only.
* **Single-node deployment.** One Redis, one PostgreSQL. Capacity is ~540–620 TPS on this laptop with 4 workers;
  scale horizontally by adding partitions/workers.
* **Retraining scope.** Retraining reuses the parent's sequence model and retrains XGBoost + meta-learner + typology.
  The LSTM is retrained only by a full `models.train` run.
* **Windows/WSL quirk.** Connecting to Redis via `localhost` costs ~2 s per new connection because of IPv6 fallback
  to WSL's forwarder. Use `127.0.0.1` (the default in `.env.example`).
* **Time zone.** Time-of-day features use UTC event time; a real deployment should use the switch's local timezone.

## Suggested next steps

1. Run the live Mojaloop consumer in shadow mode on real traffic, label alerts and retrain on real transfers; ask
   FSPs to send device/IP/geo signals in the transfer `extensionList`.
2. Add cohort-level fairness monitoring (FPR by FSP, region, account age, device-sharing cohort) and random audits
   of ALLOWed traffic to counter feedback selection bias.
3. Make the graph worker incremental (streaming edge updates) and feed ring membership into the sequence model.
4. Run periodic threshold re-calibration on fresh labelled windows; make thresholds a per-FSP policy.
5. Replace static API keys with an IdP (OIDC/JWT with roles); add rate limiting.
6. Load-test on Linux/Docker with Kafka/Redpanda and multiple Redis shards to establish production capacity.
