"""Render docs/results.md from the JSON artifacts produced by actual runs (no hand-typed numbers).

Inputs: models/artifacts/<production version>/metrics.json, models/artifacts/graph_eval.json,
        docs/benchmarks.json, docs/feedback_demo.json, data/manifest.json
Usage:  python -m scripts.make_report [--version v1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import ROOT, get_settings


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f4(x):
    return "—" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)


def model_tables(m: dict) -> list[str]:
    out = []
    t = m["test"]
    out.append("### Held-out future window (days 46–60, ground-truth labels)\n")
    out.append(
        "| Model (ablation) | PR-AUC | ROC-AUC | Recall @1% FPR | Recall @0.1% FPR | REVIEW precision | "
        "REVIEW recall | BLOCK precision | BLOCK recall | Alerts/day | Brier |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    names = {
        "xgb": "XGBoost only",
        "xgb_seq": "+ sequence (LSTM)",
        "xgb_graph": "+ graph",
        "full": "**Full ensemble** (XGB + LSTM + graph)",
    }
    for k in ("xgb", "xgb_seq", "xgb_graph", "full"):
        r = t[k]
        out.append(
            f"| {names[k]} | {f4(r['pr_auc'])} | {f4(r['roc_auc'])} | {f4(r['recall_at_1pct_fpr'])} | "
            f"{f4(r['recall_at_0.1pct_fpr'])} | {f4(r['review']['precision'])} | {f4(r['review']['recall'])} | "
            f"{f4(r['block']['precision'])} | {f4(r['block']['recall'])} | {r['review'].get('alerts_per_day')} | "
            f"{r['brier']} |"
        )
    out.append("\nSame window scored against the *observed* (noisy) labels the operator would actually have:\n")
    out.append("| Model | PR-AUC (observed) | ROC-AUC (observed) | Recall @1% FPR (observed) |")
    out.append("|---|---|---|---|")
    for k in ("xgb", "xgb_seq", "xgb_graph", "full"):
        o = t[k]["observed_labels"]
        out.append(f"| {names[k]} | {f4(o['pr_auc'])} | {f4(o['roc_auc'])} | {f4(o['recall_at_1pct_fpr'])} |")
    out.append("\n### Per fraud type (full ensemble, held-out window)\n")
    out.append("| Fraud type | Fraud transfers | Recall @ REVIEW | Recall @ BLOCK |")
    out.append("|---|---|---|---|")
    for k, v in t["full"]["per_fraud_type"].items():
        out.append(f"| {k} | {v['n']} | {f4(v['recall_review'])} | {f4(v['recall_block'])} |")
    if t.get("drift_variants"):
        d = t["drift_variants"]
        out.append(
            f"| *drift variants (day ≥ 48: device cloning / slow layering)* | {d['n']} | "
            f"{f4(d['recall_review_full'])} | — |"
        )
    if m.get("typology_accuracy_test") is not None:
        out.append(f"\nTypology classifier accuracy on held-out fraud transfers: **{m['typology_accuracy_test']:.4f}**")
    th = t["full"]
    out.append(
        f"\nOperating thresholds (chosen on the calibration window for an FPR budget of 0.5% REVIEW / 0.05% "
        f"BLOCK): REVIEW ≥ {th['review']['threshold']}, BLOCK ≥ {th['block']['threshold']}; realised FPR on "
        f"the held-out window: REVIEW {th['review']['fpr']}, BLOCK {th['block']['fpr']}."
    )
    return out


def graph_tables(g: dict) -> list[str]:
    out = [
        "| Ring score ≥ | Detected rings | Planted rings | Ring precision | Ring recall | "
        "Laundering recall | Merchant-ring recall | Settlement recall | ATO cases touched |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in g["thresholds"]:
        r = t["recall_by_typology"]

        def rr(k, r=r):
            return f"{r[k]['matched']}/{r[k]['planted']}" if k in r else "—"

        out.append(
            f"| {t['min_score']} | {t['detected_rings']} | {t['planted_rings']} | {f4(t['ring_precision'])} | "
            f"{f4(t['ring_recall'])} | {rr('transaction_laundering')} | {rr('merchant_fraud_ring')} | "
            f"{rr('settlement_manipulation')} | {t['ato_cases_touched']} |"
        )
    tm = g["timing"]
    out.append(
        f"\nDetection run time per 24 h window ({tm['runs']} windows): p50 {tm['run_ms_p50']:.0f} ms, "
        f"p95 {tm['run_ms_p95']:.0f} ms, max {tm['run_ms_max']:.0f} ms."
    )
    t5 = next((t for t in g["thresholds"] if t["min_score"] == 0.5), None)
    if t5:
        out.append(
            "\nPrecision by detector at score ≥ 0.5: "
            + ", ".join(f"{k} {v['matched']}/{v['detected']}" for k, v in t5["precision_by_detector"].items())
        )
    return out


def bench_tables(b: dict) -> list[str]:
    out = [
        f"Machine: {b['machine']['platform']}, {b['machine']['processor']}, Python {b['machine']['python']}, "
        f"{b['machine']['partitions']} scoring-worker partitions (started {b['started']}).\n",
        "| Phase | Events | Processing TPS | p50 ms | p95 ms | p99 ms | max ms | E2E p99 ms | Errors | p99 < 100 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in b["phases"]:
        if "scoring_latency_ms" in p:
            lat, e2e = p["scoring_latency_ms"], p["e2e_latency_ms"]
            out.append(
                f"| {p['phase']} | {p['processed_events']:,} | {p['processing_tps']} | {lat['p50']} | {lat['p95']} | "
                f"{lat['p99']} | {lat['max']} | {e2e.get('p99')} | {p['errors']} ({p['error_rate']:.4%}) | "
                f"{'yes' if p['p99_under_100ms'] else 'NO'} |"
            )
        else:
            c, s = p["client_latency_ms"], p["server_scoring_latency_ms"]
            out.append(
                f"| {p['phase']} — server-side scoring | {p['requests']:,} | — | {s['p50']} | {s['p95']} | "
                f"{s['p99']} | {s['max']} | — | {p['errors']} | {'yes' if s['p99'] < 100 else 'NO'} |"
            )
            out.append(
                f"| {p['phase']} — client round trip | {p['requests']:,} | — | {c['p50']} | {c['p95']} | {c['p99']} | "
                f"{c['max']} | — | {p['errors']} | {'yes' if c['p99'] < 100 else 'NO'} |"
            )
    return out


def feedback_tables(f: dict) -> list[str]:
    r = f["retrain"]
    lw = f["live_window"]
    out = [
        f"Live feedback window: {lw['scored_transfers']:,} transfers scored through the stream, "
        f"{lw['alerts_reviewed']:,} alerts reviewed by simulated operators, {lw['missed_frauds_reported']} of "
        f"{lw['missed_frauds_in_window']} missed frauds reported as false negatives. Feedback by kind: "
        f"{lw['feedback_by_kind']}.\n",
        f"Retraining run #{r['run_id']}: {r['base_version']} → candidate {r['candidate_version']} using "
        f"{r['n_feedback']} feedback labels. **Promoted: {r['promoted']}** — gate: {r['reason']}.\n",
        f"Holdout window days {r['holdout_days'][0]}–{r['holdout_days'][1]} (never used for training or feedback):\n",
        "| Metric | Before | After |",
        "|---|---|---|",
    ]
    b, a = r["before"], r["after"]
    for k, label in (
        ("pr_auc_observed", "PR-AUC (observed labels, gate metric)"),
        ("pr_auc_truth", "PR-AUC (truth)"),
        ("roc_auc", "ROC-AUC (truth)"),
        ("recall_at_1pct_fpr", "Recall @1% FPR (truth)"),
        ("review_precision", "REVIEW precision"),
        ("review_recall", "REVIEW recall"),
        ("block_precision", "BLOCK precision"),
        ("block_recall", "BLOCK recall"),
        ("drift_variant_recall_review", "Drift-variant recall @ REVIEW"),
    ):
        out.append(f"| {label} | {f4(b.get(k))} | {f4(a.get(k))} |")
    for t in sorted(set(b["per_fraud_type_recall_review"]) | set(a["per_fraud_type_recall_review"])):
        out.append(
            f"| Recall @ REVIEW — {t} | {f4(b['per_fraud_type_recall_review'].get(t))} | "
            f"{f4(a['per_fraud_type_recall_review'].get(t))} |"
        )
    out.append(
        f"\nProduction model {f['model_before']} → {f['model_after']}; scoring workers hot-reloaded to "
        f"{f['workers_model_after']} without restart."
    )
    return out


def main(argv=None):
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    a = ap.parse_args(argv)
    s = get_settings()
    version = a.version
    if version is None:
        cands = [p for p in s.artifacts_dir.iterdir() if (p / "metrics.json").exists()]
        version = min(cands, key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 1e9).name
    parts = [
        "# Measured results\n",
        "_Generated by `python -m scripts.make_report` from run artifacts. "
        "All numbers come from executed runs on synthetic data._\n",
    ]
    man = load(s.data_dir / "manifest.json")
    if man:
        parts.append(
            f"Dataset: {man['n_transactions']:,} transfers, {man['n_accounts']:,} accounts, fraud rate "
            f"{man['fraud_rate']:.4%} ({man['fraud_rows']:,} fraud transfers in {man['n_cases']} cases; "
            f"by type {man['by_typology']}), data hash `{man['data_hash']}`.\n"
        )
    m = load(s.artifacts_dir / version / "metrics.json")
    if m:
        parts.append(f"## Model metrics — {version}\n")
        parts += model_tables(m)
    g = load(s.artifacts_dir / "graph_eval.json")
    if g:
        parts.append("\n## Graph ring detection vs planted rings\n")
        parts += graph_tables(g)
    b = load(ROOT / "docs" / "benchmarks.json")
    if b:
        parts.append("\n## Latency / load benchmark\n")
        parts += bench_tables(b)
    f = load(ROOT / "docs" / "feedback_demo.json")
    if f:
        parts.append("\n## Feedback loop (before / after retraining)\n")
        parts += feedback_tables(f)
    (ROOT / "docs" / "results.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print((ROOT / "docs" / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
