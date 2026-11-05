# Deployment roadmap

Last updated: 2026-11-05. Where the project stands and the steps to put the dashboard online safely.

## Where it stands

* The Docker stack (`docker compose --profile demo up -d --build`) builds and runs end to end: 11 containers,
  ~200 TPS from the simulator, p99 scoring latency ~12 ms, all components healthy.
* Security fixes (branch `security-fixes`): operators sign in on the dashboard with their own key (none is
  built into the bundle), keys never travel in URLs, Redis / PostgreSQL / API are published on 127.0.0.1 only,
  nginx sends CSP and security headers, vulnerable dependencies are upgraded (pip-audit and npm audit clean).
* Only port **8080** (nginx: dashboard + `/v1`, `/ws`, `/healthz`) is meant to be reachable from outside.
* Tests: 79 passed (78 unit + 1 end-to-end). The end-to-end p99 < 100 ms check can fail on a small machine
  that is running the full stack and the simulator at the same time (seen: 162 ms on a 7.6 GB laptop).

## Phase 0: merge and clean up (before any deployment)

1. Open and merge the pull requests: `security-fixes`, then `simplify`.
2. **Rotate the operator keys.** The old dashboard embedded a key in its JavaScript and the WebSocket wrote it to
   `data/logs/api.log` and the nginx log. Generate new ones
   (`python -c "import secrets; print(secrets.token_urlsafe(24))"`), put them in `OPERATOR_API_KEYS`, and delete
   old logs (`data/logs/`, and `docker compose logs` history by recreating the containers).
3. Set a strong `POSTGRES_PASSWORD` in `.env` (the compose default is `fraud`). On an existing volume, change it
   with `ALTER USER fraud PASSWORD '...'` first, since the variable is only read when the volume is created.
4. Update `docs/PROJECT_STATUS.md` (it still lists `docker-fixes` and the repo-tidy items as open).

## Phase 1: hardening that should land before the dashboard is public

| Item | Why | How |
|---|---|---|
| HTTPS | The API key is sent on every request | Cloudflare Tunnel terminates TLS (option A), or Caddy / nginx + Let's Encrypt on a VM (option B) |
| Rate limiting | Slows key guessing and request floods on `/v1` and `/ws` | nginx `limit_req_zone` + `limit_req` on `/v1/` and `/ws/` |
| Non-root containers | A compromised API process should not be root in its container | `USER` in `docker/Dockerfile.app`; `chown` the existing `fraud-data` / `fraud-models` volumes once |
| Log size limits | Container logs grow without bound | `logging: {driver: json-file, options: {max-size: 10m, max-file: "3"}}` in `x-app` |
| Pinned base images | Reproducible, patchable builds | Pin `python:3.11-slim`, `node:20-alpine`, `nginx:1.27-alpine`, `redis:7.2-alpine`, `postgres:16-alpine` by digest; rebuild monthly |
| Operator roles | Every key can promote / roll back models and trigger retraining | Add an admin role for `/v1/models/*` (e.g. `OPERATOR_ADMINS=alice`) |
| CI | Nothing runs automatically today | GitHub Actions: `ruff check`, unit tests, `npm run build`, `pip-audit`, `npm audit` on each PR |

## Phase 2: choose where it runs

| | A. Laptop + Cloudflare Tunnel | B. x86 cloud VM (~16 GB RAM) |
|---|---|---|
| Cost | Free | Student credits |
| Uptime | Only while the laptop is on | Permanent |
| Capacity | This 7.6 GB PC runs the stack (~1.4 GB) but not Mojaloop alongside it | Stack + Mojaloop switch (~5 GB) + 1M-transaction training |
| Exposure | No open inbound port; the tunnel reaches `localhost:8080` | Firewall: only 443 (or 8080) open; 5432 / 6379 / 6380 / 8000 closed |

Recommendation: **A for the presentation / demo now**, **B if the dashboard has to stay online**. Oracle's free
ARM VM is unlikely to work with Mojaloop (its images are x86).

## Phase 3: deploy

Option A (laptop):
1. Start the stack: `docker compose --profile demo up -d` (Docker Desktop running).
2. Install `cloudflared`, then `cloudflared tunnel --url http://localhost:8080` for a temporary URL, or a named
   tunnel on your own domain for a stable one. Optionally put Cloudflare Access (email login) in front.
3. Share the URL; each operator signs in with their own key.

Option B (VM):
1. Install Docker Engine + Compose plugin; clone the repo (branch `main` after Phase 0).
2. `cp .env.example .env`; set `OPERATOR_API_KEYS`, `POSTGRES_PASSWORD`; leave `DEMO_N_TX` unset (1M) with 16 GB.
3. `docker compose --profile demo up -d --build` (first start trains the models; about an hour).
4. TLS reverse proxy (Caddy is one line: `your.domain { reverse_proxy localhost:8080 }`), or a tunnel as in A.
5. Cloud firewall: allow 443 only (and SSH from your IP).

## Phase 4: verify and operate

* Smoke test after each deploy: `GET /healthz` is `ok`; sign-in rejects a wrong key and accepts a real one; the
  live stream shows "open"; System Health p99 stays well under 100 ms.
* Monitoring: an uptime check on `https://<host>/healthz`; Prometheus can scrape `/metrics` on `127.0.0.1:8000`.
* Backups: `docker run --rm -v fraud-engine_pg-data:/v -v $PWD:/b alpine tar czf /b/pg-data.tgz -C /v .` (and
  the same for `fraud-models`), daily.
* Updates: `git pull && docker compose --profile demo up -d --build`; models and data persist in the volumes.
* Retraining runs on its schedule (`RETRAIN_INTERVAL_HOURS`); check Feedback & Models for promotions.

## Phase 5 (optional): live Mojaloop

Needs option B (or a machine with ~16 GB RAM). Start `ml-core-test-harness`, then
`docker compose -f docker-compose.yml -f docker-compose.mojaloop.yml up -d --build`; steps and expected results
are in `docs/mojaloop_fraud_test.md`. Plain FSPIOP without device extensions still needs a model trained
without device features (`docs/PROJECT_STATUS.md`, pending item 5).
