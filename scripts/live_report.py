"""Side-by-side markdown table of live evaluation runs (JSON files written by ``scripts.evaluate_live``).

Usage: python -m scripts.live_report docs/live_eval/direct.json docs/live_eval/mojaloop_device.json ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def render(runs: list[dict]) -> str:
    alerts = [r["alerts_review_or_block"] for r in runs]
    blocks = [r["blocks"] for r in runs]
    rows = [
        ("Transfers scored", [f"{r['transfers_scored']:,} / {r['transfers_expected']:,}" for r in runs]),
        ("Frauds in window", [str(r["frauds"]) for r in runs]),
        ("PR-AUC", [str(r["pr_auc"]) for r in runs]),
        ("ROC-AUC", [str(r["roc_auc"]) for r in runs]),
        ("Recall at 1% false-positive rate", [_pct(r["recall_at_1pct_fpr"]) for r in runs]),
        ("Alerts (REVIEW + BLOCK)", [str(a["flagged"]) for a in alerts]),
        ("Frauds caught (REVIEW or BLOCK)", [f"{a['frauds_caught']} ({_pct(a['recall'])})" for a in alerts]),
        ("False alarms", [f"{a['false_alarms']} ({_pct(a['false_alarm_rate'])} of legit)" for a in alerts]),
        ("Alert precision", [_pct(a["precision"]) for a in alerts]),
        ("Blocked frauds", [f"{b['frauds_caught']} ({_pct(b['recall'])})" for b in blocks]),
        ("Block precision", [_pct(b["precision"]) for b in blocks]),
    ]
    types = sorted({t for r in runs for t in r["per_fraud_type"]})
    for t in types:
        cells = []
        for r in runs:
            v = r["per_fraud_type"].get(t)
            cells.append(f"{v['caught']}/{v['frauds']} ({_pct(v['recall'])})" if v else "–")
        rows.append((f"Caught: {t.replace('_', ' ')}", cells))
    rows.append(
        (
            "Scoring latency p50 / p99",
            [f"{r['scoring_latency_ms']['p50']} / {r['scoring_latency_ms']['p99']} ms" for r in runs],
        )
    )
    head = "| | " + " | ".join(r["label"] for r in runs) + " |"
    sep = "|---|" + "---|" * len(runs)
    return "\n".join([head, sep] + [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args(argv)
    print(render([orjson.loads(f.read_bytes()) for f in a.files]))


if __name__ == "__main__":
    main()
