# Feature dictionary

All features are computed by **one module**, `pipeline/features.py` (`FeatureEngine`), for both offline training
(event-time replay through `InMemoryStore`) and online serving (`RedisStore`). Every window is in **event time**
(`ts` of the transfer, epoch ms), so a replay at any speed produces the same values. Parity is enforced by
`tests/unit/test_feature_parity.py` (identical output from both stores on 6,000 events including fraud reports and
graph refreshes, plus a check that `pipeline/replay.py` output equals the online path).

State is read **before** the current transfer is applied (no self-leakage), and a transfer only ever sees events
with `ts <= its own ts` (out-of-order arrivals are ignored for history-based windows). No feature uses a label.
`NaN` marks "not computable" (e.g. cold-start payer) and is handled natively by XGBoost.

Notation: *P* = payer account, *Q* = payee account, *D* = device id, *I* = IP, *F* = payer FSP.
"History" = the payer's last 50 transfers kept in Redis list `ph:<P>`.

## Tabular features (XGBoost) — 54

| # | Feature | Definition | Window | Rationale |
|---|---|---|---|---|
| 1 | `amount_log` | log(1 + amount) | — | Fraud drains / layering amounts are larger than everyday spend |
| 2 | `amount_zscore_payer` | (amount − mean) / (std + 1) over the payer's history | 30 d, ≥3 tx else NaN | Amount deviation vs the payer's own behaviour |
| 3 | `amount_ratio_mean_30d` | amount / mean(history amounts) | 30 d | "8.2× this payer's 30-day average" (ATO drains) |
| 4 | `amount_ratio_max_30d` | amount / max(history amounts) | 30 d | New maximum spend |
| 5 | `payer_hist_count_30d` | number of payer transfers in history | 30 d (cap 50) | Cold-start / thin-file payers |
| 6 | `is_round_amount` | round_level ≥ 2 (multiple of 100) | — | Structuring uses round values |
| 7 | `round_level` | largest k ≤ 5 such that amount is a multiple of 10^k | — | Strength of roundness |
| 8 | `near_threshold` | amount in [9k,10k), [45k,50k) or [90k,100k) | — | Structuring just below reporting/limit thresholds |
| 9–11 | `payer_tx_1m/1h/24h` | count of payer transfers | 1 min / 1 h / 24 h | Transaction velocity bursts |
| 12 | `payer_amt_1h_log` | log(1 + Σ payer amounts) | 1 h | Value velocity (draining) |
| 13–14 | `payee_tx_1h/24h` | count of transfers received by payee (zset `zt:q:<Q>`) | 1 h / 24 h | Mule / collector inflow bursts |
| 15–16 | `device_tx_1h/24h` | count of transfers from the device (zset `zt:d:<D>`) | 1 h / 24 h | Device farms |
| 17 | `new_payee` | payee never paid by this payer before | since bootstrap | Payee-alias change / new beneficiary |
| 18 | `payer_out_degree` | distinct payees ever paid by P | since bootstrap | Graph degree: thin vs established payer |
| 19 | `payee_in_degree` | distinct payers that paid Q | since bootstrap | Merchants vs fresh mules |
| 20 | `payee_out_degree` | distinct payees Q has paid | since bootstrap | Pass-through accounts pay onwards |
| 21 | `payee_age_hours_log` | log(1 + hours since Q first seen) | since bootstrap | Newly created beneficiary accounts |
| 22 | `payer_age_hours_log` | log(1 + hours since P first seen) | since bootstrap | Newly created payer accounts |
| 23 | `payee_report_count` | fraud reports against Q (registry feed + operator-confirmed fraud) | reports with `report_ts ≤ ts` | Known-bad counterparties |
| 24 | `payer_report_count` | fraud reports against P | same | Known-bad payer |
| 25 | `counterparty_risk_score` | 1 − e^(−reports_Q) · (1 − min(1, 5·reversal_rate_Q)) · (1 − 0.5·[Q younger than 72 h]) | as above | Composite counterparty risk (no graph input, so the XGB-only ablation stays graph-free) |
| 26 | `hour_of_day` | UTC hour of event time | — | Time-of-day pattern |
| 27 | `is_night` | hour < 5 | — | Night-time ATO |
| 28 | `hour_deviation` | circular distance (h) from the payer's mean hour | history | Unusual time for this payer |
| 29 | `is_weekend` | Saturday/Sunday | — | Day-of-week pattern |
| 30 | `dow_deviation` | 1 − share of payer history on the same weekday type | history | Unusual day for this payer |
| 31 | `secs_since_last_tx_log` | log(1 + seconds since payer's previous transfer) | history | Rapid sequences |
| 32 | `device_accounts` | distinct payer accounts seen on D (incl. current) | since bootstrap | Device sharing / farms |
| 33 | `ip_accounts` | distinct payer accounts seen on I (incl. current) | since bootstrap | IP sharing (CGNAT is a hard negative) |
| 34 | `new_device_for_payer` | P never used D before | since bootstrap | SIM-swap / device change |
| 35 | `payer_device_count` | distinct devices used by P (incl. current) | since bootstrap | Credential sharing |
| 36 | `device_binding_age_hours_log` | log(1 + hours since the device was bound to the account), from the `deviceBoundAt` extension | — (NaN if absent) | Fresh SIM-swap / re-binding (app re-installs are hard negatives) |
| 37 | `geo_distance_km` | haversine distance from P's previous transfer location | history | Geo mismatch |
| 38 | `geo_speed_kmh` | distance / elapsed time (cap 2000) | history | Impossible travel |
| 39 | `payee_reversal_rate` | reversals+refunds issued by Q / (transfers received by Q + 1) | since bootstrap | Merchant reversal abuse |
| 40 | `payer_reversal_rate` | reversals issued by P / (P's in + out + 1) | since bootstrap | Settlement manipulation reversals |
| 41 | `is_reversal` | scenario = REVERSAL (Mojaloop REFUND/subScenario REVERSAL) | — | Reversal events (technical reversals are hard negatives) |
| 42 | `is_refund` | scenario = REFUND | — | Merchant refunds |
| 43 | `inflow_24h_log` | log(1 + Σ amounts received by P) (zset `inf:<P>`) | 24 h | Money in before money out |
| 44 | `pass_through_ratio_24h` | (P's outflow + this amount) / P's inflow, cap 5; 0 if no inflow | 24 h | Mule pass-through (family relays are hard negatives) |
| 45 | `mins_since_last_inflow_log` | log(1 + minutes since P last received money) | 24 h | Rapid in-and-out |
| 46 | `mins_to_settlement_cutoff` | minutes to the next settlement-window cutoff (config `SETTLEMENT_CUTOFF_HOURS`) | — | Settlement-window timing |
| 47 | `near_settlement_cutoff` | `mins_to_settlement_cutoff` ≤ 15 | — | Settlement manipulation |
| 48 | `fsp_rhv_zscore` | z-score of the payer FSP's round-high-value transfer count in the current 30-min bucket vs the previous 48 buckets (0 if the transfer is not round-high-value) | 24 h of 30-min buckets | FSP-level anomaly |
| 49 | `fsp_volume_zscore` | same for total FSP volume | 24 h | FSP-level anomaly |
| 50 | `channel_code` | APP 0, QR 1, WEB 2, USSD 3, other 4 | — | Channel risk |
| 51 | `payee_is_merchant` | payee type MERCHANT (FSPIOP `merchantClassificationCode` present) | — | P2M vs P2P |
| 52 | `payer_is_business` | payer is BUSINESS/MERCHANT | — | B2B / merchant payouts |
| 53 | `cross_fsp` | payer FSP ≠ payee FSP | — | Interoperable transfer |

`TABULAR_FEATURES` in `pipeline/features.py` is the authoritative ordered list (53 entries).

## Graph features (graph component of the ensemble) — 7

Produced asynchronously by the graph worker (`graph/`) on the trailing 24 h event window and cached in Redis hashes
`g:a:<account>` / `g:d:<device>`; the hot path only performs lookups (same pipeline round trip as the other reads).

| Feature | Definition | Rationale |
|---|---|---|
| `payer_ring_score` | max score of detected rings containing P (0 if none) | Ring membership |
| `payee_ring_score` | same for Q | Merchant/mule rings |
| `device_ring_score` | same for D (shared-device / dense-subgraph rings) | Device farms |
| `payer_fraud_neighbor_ratio` | share of P's window counterparties that are flagged (reported, or ring members with score ≥ 0.5) | Guilt by association |
| `payee_fraud_neighbor_ratio` | same for Q ("merchant fraud-neighbour ratio") | Merchant fraud-neighbour ratio |
| `max_ring_size` | max ring size (accounts) across P, Q, D; fed to the meta-learner as log(1+x) | Ring scale |
| `in_cycle` | P or Q is on a detected circular flow | Circular laundering |

## Sequence features (LSTM input) — 16 steps × 12

Built from the same payer history list (last 15 transfers + the current one, chronological, left-padded with zeros).

| Feature | Definition |
|---|---|
| `amt_log` | log(1+amount)/12 |
| `amt_vs_median` | log((amount+1)/(median of the sequence+1)), clipped to ±3 |
| `hour_sin`, `hour_cos` | cyclical hour of day |
| `gap_log` | log(1 + minutes since previous step)/10 |
| `new_payee_in_seq` | payee not seen earlier in the sequence |
| `device_change` | device differs from previous step |
| `geo_jump_log` | log(1 + km from previous step)/8 |
| `round_flag` | multiple of 100 |
| `payee_merchant` | payee is a merchant |
| `is_current` | 1 for the transfer being scored |
| `present` | 1 for real steps, 0 for padding |
