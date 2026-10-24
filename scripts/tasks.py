"""Cross-platform task runner (the Makefile and Docker entrypoints call into this).

python scripts/tasks.py prepare [--n-tx N]   generate data -> replay features -> train -> graph eval (idempotent)
python scripts/tasks.py bootstrap            load the online state into Redis/Postgres, promote model
python scripts/tasks.py run                  start API + scoring workers + graph worker + retrain scheduler
python scripts/tasks.py demo [--tps 200]     prepare + bootstrap + run + dashboard + simulator (+ open browser)
python scripts/tasks.py test | lint | bench
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = sys.executable


def sh(args: list[str], **kw) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True, **kw)


def spawn(name: str, args: list[str], logs: Path) -> subprocess.Popen:
    logs.mkdir(parents=True, exist_ok=True)
    f = open(logs / f"{name}.log", "ab")
    print(f"  starting {name}: {' '.join(args)}  (log: data/logs/{name}.log)", flush=True)
    return subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def prepare(n_tx: int, force: bool = False) -> None:
    from core.config import get_settings

    s = get_settings()
    data = s.data_dir
    if force or not (data / "transactions.parquet").exists():
        sh([PY, "-m", "data_gen.generate", "--n-transactions", str(n_tx), "--out", str(data)])
    if force or not (data / "features" / "features.parquet").exists():
        sh([PY, "-m", "pipeline.replay", "--data", str(data), "--out", str(data / "features"), "--snapshot-day", "46"])
    has_model = s.artifacts_dir.exists() and any((p / "bundle.json").exists() for p in s.artifacts_dir.iterdir())
    if force or not has_model:
        sh(
            [
                PY,
                "-m",
                "models.train",
                "--features",
                str(data / "features"),
                "--data",
                str(data),
                "--register",
                "--promote",
            ]
        )
    if force or not (s.artifacts_dir / "graph_eval.json").exists():
        sh([PY, "-m", "graph.evaluate", "--data", str(data), "--out", str(s.artifacts_dir / "graph_eval.json")])


def bootstrap() -> None:
    from core.config import get_settings

    s = get_settings()
    sh(
        [
            PY,
            "-m",
            "pipeline.bootstrap",
            "--features",
            str(s.data_dir / "features"),
            "--data",
            str(s.data_dir),
            "--reset",
        ]
    )


def services(with_dashboard: bool) -> list[subprocess.Popen]:
    from core.config import get_settings

    s = get_settings()
    logs = s.data_dir / "logs"
    procs = [spawn("api", [PY, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"], logs)]
    for p in range(s.stream_partitions):
        procs.append(spawn(f"worker-{p}", [PY, "-m", "pipeline.worker", "--partition", str(p)], logs))
    procs.append(spawn("graph-worker", [PY, "-m", "graph.worker"], logs))
    procs.append(spawn("persister", [PY, "-m", "pipeline.persister"], logs))
    procs.append(spawn("retrainer", [PY, "-m", "models.retrain", "--schedule"], logs))
    if with_dashboard:
        npm = shutil.which("npm") or "npm"
        dash = ROOT / "dashboard"
        if not (dash / "node_modules").exists():
            subprocess.run([npm, "ci"], cwd=dash, check=True, shell=os.name == "nt")
        f = open(logs / "dashboard.log", "ab")
        procs.append(
            subprocess.Popen([npm, "run", "dev"], cwd=dash, stdout=f, stderr=subprocess.STDOUT, shell=os.name == "nt")
        )
        print("  starting dashboard: http://localhost:5173", flush=True)
    return procs


def wait_api(timeout=120) -> None:
    import httpx

    t = time.time()
    while time.time() - t < timeout:
        try:
            if httpx.get("http://127.0.0.1:8000/healthz", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit("API did not become healthy (see data/logs/api.log)")


def stop_all(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            else:
                p.send_signal(signal.SIGTERM)
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "bootstrap", "run", "demo", "test", "lint", "bench", "feedback-demo"])
    ap.add_argument("--n-tx", type=int, default=int(os.environ.get("DEMO_N_TX", "1000000")))
    ap.add_argument("--tps", type=float, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a, rest = ap.parse_known_args(argv)
    if a.cmd == "prepare":
        prepare(a.n_tx, a.force)
    elif a.cmd == "bootstrap":
        bootstrap()
    elif a.cmd == "test":
        sh([PY, "-m", "pytest", *rest])
    elif a.cmd == "lint":
        sh([PY, "-m", "ruff", "check", "."])
        subprocess.run(["npm", "run", "typecheck"], cwd=ROOT / "dashboard", check=True, shell=os.name == "nt")
    elif a.cmd == "bench":
        sh([PY, "-m", "tests.load.bench", *rest])
    elif a.cmd == "feedback-demo":
        sh([PY, "-m", "scripts.feedback_demo", *rest])
    elif a.cmd in ("run", "demo"):
        if a.cmd == "demo":
            prepare(a.n_tx, a.force)
            bootstrap()
        procs = services(with_dashboard=(a.cmd == "demo"))
        try:
            wait_api()
            print("API healthy: http://localhost:8000/docs", flush=True)
            if a.cmd == "demo":
                from core.config import get_settings

                logs = get_settings().data_dir / "logs"
                procs.append(spawn("simulator", [PY, "-m", "simulator.simulator", "--tps", str(a.tps), "--loop"], logs))
                if not a.no_browser:
                    time.sleep(4)
                    webbrowser.open("http://localhost:5173/alerts")
                print(
                    json.dumps(
                        {
                            "dashboard": "http://localhost:5173",
                            "api_docs": "http://localhost:8000/docs",
                            "logs": "data/logs/",
                        },
                        indent=1,
                    )
                )
            print("Running. Press Ctrl+C to stop.", flush=True)
            while True:
                time.sleep(1)
                dead = [p for p in procs if p.poll() is not None]
                for p in dead:
                    print(f"process {p.args} exited with {p.returncode}", flush=True)
                    procs.remove(p)
        except KeyboardInterrupt:
            pass
        finally:
            stop_all(procs)


if __name__ == "__main__":
    main()
