# Thin wrappers. Docker targets need Docker Compose v2; local targets need Python 3.11 + Node 20,
# a reachable Redis and PostgreSQL (see README "Run without Docker").
PY ?= python

.PHONY: up demo down logs prepare bootstrap run local-demo test test-unit test-integration bench lint feedback-demo dashboard

up:                 ## start the full stack (first run generates data + trains, ~15-25 min for 1M tx)
	docker compose up -d --build

demo:               ## full stack + live simulator; then open http://localhost:8080
	docker compose --profile demo up -d --build
	@echo "Dashboard: http://localhost:8080   API docs: http://localhost:8000/docs"

down:
	docker compose --profile demo down

logs:
	docker compose logs -f --tail=100

prepare:            ## generate data, replay features, train, evaluate graph (local)
	$(PY) scripts/tasks.py prepare

bootstrap:
	$(PY) scripts/tasks.py bootstrap

run:                ## start API + workers + graph worker + retrain scheduler (local)
	$(PY) scripts/tasks.py run

local-demo:         ## everything locally incl. dashboard dev server + simulator, opens the browser
	$(PY) scripts/tasks.py demo

test:
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest tests/unit

test-integration:
	$(PY) -m pytest tests/integration -s

bench:              ## needs the system running (make run / make up)
	$(PY) -m tests.load.bench

feedback-demo:      ## needs the system running; replays the feedback window, labels alerts, retrains
	$(PY) -m scripts.feedback_demo

lint:
	$(PY) -m ruff check .
	cd dashboard && npm run typecheck

dashboard:
	cd dashboard && npm run build
