# Project status and handoff notes

Last updated: 2026-10-30. Read this first when picking the project up again.

## Requirement status (brief: AI-5, Bracket 2, Project 5)

Every item of the brief and the Definition of Done is built and demonstrated:

| Requirement | Evidence |
|---|---|
| Real-time scoring before settlement, < 100 ms | Live Mojaloop run: 15,182/15,182 transfers scored, p99 18.3 ms (laptop) |
| 15+ features | 53 tabular + 7 graph + sequence features (`docs/feature_dictionary.md`) |
| XGBoost + LSTM + graph ensemble | `models/`, `graph/` |
| Fraud rings (NetworkX: dense subgraphs, cycles) | Cycle and mule-network rings detected from Mojaloop traffic (days 46–51 replay) |
| Dashboard: alerts, investigation, cases | 6 views; cases with audit trail tested |
| SHAP for every transaction | REVIEW/BLOCK synchronous, ALLOW asynchronous |
| Feedback loop retrains | `scripts/feedback_demo.py`: v1 → v2, PR-AUC 0.816 → 0.865 |
| Synthetic but realistic data | `data_gen/` (the brief asks for synthetic data; PaySim is not required) |
| Mojaloop (DPG) | Live integration + full fraud test: `docs/mojaloop_integration.md`, `docs/mojaloop_fraud_test.md` |

Laptop Mojaloop fraud test (days 46–51, 200k-transaction model): 53/57 frauds caught (93.0%), 89 false alarms
(0.6% of legitimate), PR-AUC 0.850, ROC-AUC 0.994, block precision 80.8%, latency p50/p99 7.25/18.29 ms.
Server run of the same test: `docs/live_eval/mojaloop_complete.json`.

## Next: Docker deployment (planned for the next session)

* **Branch `docker-fixes`** (not merged): adds `.dockerignore` and builds the dashboard with `npm run build`.
  Verified on a Linux test machine: both images build and `docker compose --profile demo up` runs all 11
  containers (dashboard on :8080, API, WebSocket alerts, simulator) with no errors. Not yet verified there:
  the Debian `apt-get` step and the PyTorch CPU-index install / in-container training (blocked by that
  machine's network). **Verified on the Windows laptop (2026-10-30):** full build, 1M-transaction training in
  `init`, all containers up, dashboard on :8080 working.
* **Laptop facts (Windows 11, 16 GB RAM, Docker Desktop on WSL 2):**
  * Docker's disk image is on **`D:\DockerData`** (moved after the first build filled C: — do not move it back).
  * `%USERPROFILE%\.wslconfig` caps WSL memory (`memory=8GB`; raise to ~10GB for the Docker build/training,
    when Mojaloop is not running, then `wsl --shutdown`).
  * Set `$env:OPENBLAS_NUM_THREADS="1"` in each PowerShell window before starting Python (avoids
    "OpenBLAS memory allocation failed").
  * Docker stack and the 4-window setup share ports 5432/6379/8000: never run both at once
    (`docker stop fraud-pg fraud-redis` before `docker compose up`).
  * Recommended training size for Docker: default 1,000,000 (`DEMO_N_TX` unset); needs ~15 GB disk on D:.
* **Docker + Mojaloop verified (2026-10-30):** engine in Docker with `docker-compose.mojaloop.yml`, sender from
  the venv replaying the Docker dataset: 14,824/14,824 transfers scored, 49/60 frauds caught, p99 19.65 ms
  (`docs/mojaloop_fraud_test.md`, "Running the engine in Docker instead").
* **Deployment plan:** Docker and Docker + Mojaloop verified → next: public access. With Mojaloop, the
  recommended option is the laptop plus a Cloudflare Tunnel to :8080 (free; the switch needs ~5 GB RAM and its
  images are x86, so Oracle's free ARM VM likely cannot run it). A permanent alternative is an x86 VM with
  ~16 GB RAM (student credits), with only port 8080 open (close 5432/6379/6380/8000 in the cloud firewall).

## The 4-window procedure (laptop, working)

Repo `D:\fraud-engine` (branch `main`), venv `.venv`; Mojaloop harness `D:\ml-core-test-harness` with
`docker/mojaloop-harness.override.yml` copied in as `docker-compose.override.yml`. Containers `fraud-pg`
(5432) and `fraud-redis` (**6380**); `.env` has `REDIS_URL=redis://127.0.0.1:6380/0`, and
`TRUST_FSP_EVENT_TIME=true` only during Mojaloop replays. Steps: `docs/mojaloop_fraud_test.md`.

## Pending small items

1. Push the laptop result files `docs/live_eval/laptop.json` and `docs/live_eval/laptop_switch.json`
   (the laptop has no GitHub push credentials yet; install GitHub Desktop or run `gh auth login`).
2. README: test count is 72 (says 39); "Live Mojaloop" row and Docker/PaySim notes are out of date.
3. Case title stays "Investigation of 0 alert(s)" after alerts are added.
4. Dashboard footer always says "not a live Mojaloop switch".
5. Plain FSPIOP without device extensions needs a model trained without device features (not needed for the
   synthetic-data presentation).

## Conventions

* Commits and PRs are authored as Lakshith9 and must not mention AI tools.
* Work goes on a branch with a pull request; the owner merges.
