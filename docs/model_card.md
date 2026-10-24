# Model card — DPI real-time fraud ensemble (v1, and the feedback-retrained v2)

> **All data is synthetic.** The models were trained and evaluated only on Mojaloop-style data produced by
> `data_gen/generate.py`. The numbers below describe how well the system separates the fraud *the generator plants*
> from the legitimate behaviour *the generator simulates*. They are **not** an estimate of performance on a real
> switch, where fraud is more varied, adaptive and under-labelled. Every number below is copied from
> [`docs/results.md`](results.md), which `scripts/make_report.py` renders from the artifacts of the runs.

## Intended use

* Pre-settlement risk scoring of interoperable P2P/P2M transfers on a Mojaloop / UPI-like switch, producing
  ALLOW / REVIEW / BLOCK decisions with operator-facing reasons.
* Decision support for fraud operations teams. BLOCK is meant for automated holds subject to review, not
  irreversible account actions.
* **Not intended for:** credit decisions, profiling of individuals, or deployment on real traffic without
  retraining and re-validating on that institution's own labelled data.

## Data

| Item | Value |
|---|---|
| Generator | `data_gen/generate.py`, seed 42, data hash `36a56b1e02663366` |
| Volume | 1,001,316 transfers, 43,600 accounts, 12 FSPs, 60 days of event time |
| Fraud rate | 0.313 % (3,131 transfers in 260 planted cases) |
| Typologies | identity spoofing / ATO (750), settlement manipulation (608), merchant fraud rings (1,010), transaction laundering (763) |
| Realism devices | log-normal amounts, daily/weekly seasonality, salary days, festival sales, repeat payees, households sharing devices, CGNAT IPs, travel and IP-geolocation errors, phone changes and app re-installs (fresh device bindings), legit refunds and technical reversals, B2B round payments near settlement cutoffs, family relays (pass-through), large one-off transfers to new payees |
| Adversarial realism | 30 % of fraud cases are *stealth* variants that mimic customers (cloned devices, victim location, everyday amounts, recruited established mule accounts, slower layering); concept drift from day 48 (device-cloning ATO, slow laundering hops) |
| Label noise | 3 % of fraud transfers unlabelled (unreported), 0.01 % of legit transfers mislabelled as fraud. Training uses the *observed* noisy labels; evaluation reports both observed and ground-truth labels |
| Fraud registry | 60 % of fraud beneficiaries are reported with a log-normal delay (median ≈ 36 h); reports feed counterparty risk only after their report time |

**Time-based split** (event time; no random shuffling): train days 0–36 · valid_a 36–41 (XGBoost
hyper-parameters / early stopping, LSTM epoch selection) · valid_b 41–46 (meta-learner = calibration, and
thresholds) · **test 46–60 (held-out future window)**. For the feedback demo, 46–53 is the live feedback window and
**53–60 is a holdout** that neither training nor feedback ever touches.

**Leakage controls:**
* Features are computed by event-time replay, reading state *before* the current transfer is applied.
* Out-of-order events cannot see the future.
* Graph scores come from windows that end before the scored event.
* Fraud reports are applied only at their report time.
* No feature uses a label.

## Features

53 tabular features, 7 graph features, and a 16-step × 12-feature sequence. All are defined in
[feature_dictionary.md](feature_dictionary.md) and computed by the single shared module `pipeline/features.py`.
Train/serve parity is enforced by `tests/unit/test_feature_parity.py`.

## Models

| Component | Details |
|---|---|
| XGBoost (tabular) | `binary:logistic`, hist; grid of 5 configurations scored by PR-AUC on valid_a; `scale_pos_weight = √(neg/pos)` or `neg/pos`; early stopping; selected depth 6, η 0.05, min_child_weight 5, `scale_pos_weight = neg/pos`, 384 trees |
| Sequence | 1-layer LSTM (48 hidden) over the payer's last 15 transfers + current, plus a skip connection of the current step; weighted BCE with negative subsampling; exported to ONNX (max \|ONNX − PyTorch\| difference 1.9e-6; LSTM validation PR-AUC 0.6813 on its own) and served with ONNX Runtime, batch 1 |
| Graph | Five NetworkX detectors (temporal cycles, k-core/Louvain dense subgraphs, fan-out→pass-through→fan-in mule networks, shared-device clusters, reciprocal settlement bursts) producing ring scores; 7 node-level graph features |
| Meta-learner | Logistic regression on `[xgb_margin, seq_logit, graph features]` fitted on valid_b. It is both the stacker and the probability calibrator (Platt-style), which keeps explanations exactly additive. Four combos (`xgb`, `xgb_seq`, `xgb_graph`, `full`) are fitted for graceful degradation |
| Thresholds | REVIEW at FPR ≤ 0.5 %, BLOCK at FPR ≤ 0.05 % of legitimate traffic, chosen on valid_b |
| Typology | XGBoost multiclass on fraud rows, used to label alerts with the likely fraud type |

## Results (held-out future window, days 46–60)

| Model | PR-AUC | ROC-AUC | Recall @1 % FPR | BLOCK precision / recall | REVIEW precision / recall |
|---|---|---|---|---|---|
| XGBoost only | 0.8899 | 0.9913 | 0.9333 | 0.7003 / 0.8752 | 0.2767 / 0.9234 |
| + sequence | 0.8922 | 0.9925 | 0.9370 | 0.7168 / 0.8665 | 0.2803 / 0.9271 |
| + graph | 0.8905 | 0.9911 | 0.9333 | 0.6924 / 0.8764 | 0.2722 / 0.9234 |
| **Full ensemble** | **0.8926** | **0.9924** | **0.9357** | 0.7006 / 0.8677 | 0.2782 / 0.9258 |

Against the observed noisy labels the full ensemble reaches PR-AUC 0.8321 and recall@1 % FPR 0.8947.

Per typology (full ensemble, REVIEW / BLOCK recall):

| Typology | REVIEW recall | BLOCK recall |
|---|---|---|
| Identity spoofing | 0.8827 | 0.7908 |
| Merchant fraud ring | 0.8524 | 0.8143 |
| Settlement manipulation | 1.0000 | 1.0000 |
| Transaction laundering | 0.9754 | 0.8893 |
| Drift variants (REVIEW) | 0.9283 | — |

The typology classifier reaches 0.9876 accuracy on held-out fraud transfers.

**Ablation reading.** The sequence and graph components give small, consistent gains in PR-AUC (+0.0027 together)
and recall@1 % FPR. Most of the signal is already captured by the rich tabular features, many of which are
graph-like (degrees, device/IP sharing, pass-through). Ring detection's main value here is operator-facing: ring
evidence in explanations and investigations. Its incremental contribution to the transfer-level score is modest on
this data.

**Threshold transfer.** The FPR budgets were met on the calibration window. On the future window the realised FPR
was 0.71 % at REVIEW (budget 0.5 %) and 0.11 % at BLOCK (budget 0.05 %). The test window contains festival days and
drift, so thresholds should be re-validated periodically. The PSI monitor and the retraining gate exist for this.

**Graph ring detection** vs the planted collusive rings (106 rings): precision 0.7647 / recall 0.6509 at ring score
≥ 0.5, and 0.9508 / 0.5377 at ≥ 0.7. Laundering networks and settlement bursts are found most reliably. *Stealth*
merchant rings that use only established accounts and their own phones are mostly missed: only 5 of 10 merchant
rings are found at ≥ 0.5.

## Feedback retraining (v1 → v2)

Simulated operators reviewed all 1,192 alerts in the live feedback window (441 true positives, 751 false positives)
and reported 26 of 46 missed frauds. Retraining used those 1,218 labels, overriding the original labels. The
candidate was gated on the untouched holdout (days 53–60) and **promoted**:

| Holdout metric | v1 | v2 |
|---|---|---|
| PR-AUC (observed labels — gate) | 0.8162 | 0.8652 |
| PR-AUC (truth) | 0.8960 | 0.9441 |
| Recall @1 % FPR (truth) | 0.9627 | 0.9783 |
| BLOCK precision | 0.5840 | 0.8862 |
| REVIEW precision | 0.2089 | 0.4058 |
| Drift-variant recall @ REVIEW | 0.9324 | 0.9459 |

Most of the gain is precision: false-positive feedback teaches the model what the new, legitimate post-drift
traffic looks like.

## Explainability

Every REVIEW/BLOCK decision carries its top 5 contributions, computed synchronously. They are exact additive
contributions in the ensemble's log-odds:
* SHAP TreeExplainer values for each tabular feature, scaled by the meta weight,
* the sequence-model term,
* per-graph-feature terms with ring id, type, size and cycle length.

Each contribution is rendered as plain-language text, for example "amount 8.2× this payer's 30-day average" or
"part of a 6-node circular flow". ALLOW explanations are computed asynchronously by the persister or on demand.

**Known library issue.** SHAP 0.51 reports `expected_value = 0` for XGBoost 3.x models. The base value is therefore
taken from XGBoost's own TreeSHAP bias term, and additivity is verified whenever a model loads.

## Limitations, bias and ethics

* **Synthetic only.**
  * The ring structures, typologies and hard negatives are designed by us, so performance is optimistic relative to
    real fraud.
  * Typology parameters can leak into what the model learns; for example, settlement-manipulation recall is 1.0
    because even its stealth variant uses rogue-FSP business accounts that behave unlike normal businesses.
* **Fairness proxies.**
  * Features such as `geo_distance_km`, city-level geo, device sharing, IP sharing and account age can correlate
    with protected or vulnerable groups: shared family phones, migrant workers, new-to-banking users, people in
    regions with CGNAT or poor IP geolocation.
  * Before real use, measure false-positive rates by region, FSP, account age and device-sharing cohort, and keep
    human review in the loop for BLOCK.
* **Thin-file customers** (cold start) are scored mostly from transfer-level and counterparty features. New users
  may see more REVIEWs.
* **Feedback loops** can entrench bias: only alerted transfers get labelled. The demo adds false-negative reports
  to counter this, but real deployments need random audits of ALLOWed traffic.
* **Label noise.** Training uses noisy observed labels, and gate decisions use observed labels on the holdout.
* **Privacy.** The system processes payment metadata (aliases, device ids, IPs, geo). A deployment needs data
  minimisation, retention limits and lawful-basis review. Simulated aliases and IPs here are random.
