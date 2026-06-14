# CICIoT2023 IoT Intrusion Detection — Full Project Report

_Master record of the whole project: data, feature engineering, every model, the
extractor, the two-stage live system, and all findings — good and bad. Honest
throughout._

---

## 0. TL;DR

- **Goal:** detect attacks in live network flows on a Raspberry Pi, from a
  pcap → CICFlowMeter-style extractor, using a lightweight model.
- **The deployable system is TWO-STAGE:**
  1. **Benign anomaly gate** (Isolation Forest, 50 trees) — "is this normal for
     *this* network?" — **calibrated on-site**.
  2. **Attack-type classifier** (DecisionTree, attacks-only, **28 fine-grained
     classes** — the DoS/DDoS floods merged by protocol into
     `UDP_Flood/SYN_Flood/TCP_Flood/HTTP_Flood`, everything else kept distinct) —
     labels what the gate flags.
- **Why two stages:** a supervised "benign" class does **not** generalise across
  networks (cross-capture benign 14%), but an on-site-calibrated anomaly gate
  reaches **~95% benign / ~92% attack**.
- **Why the protocol merge:** the DoS-vs-DDoS split did **not** generalise across
  capture sessions (DoS-UDP recall 0.79 within-capture → **0.11** cross-capture);
  merging each DoS flood into its matching DDoS flood by protocol (and
  SynonymousIP→SYN_Flood) fixes it — held-out DoS sessions now recall **95–99.5%**
  as their merged class. Non-flood DDoS subtypes (ICMP/PSHACK/RSTFIN/fragmentation)
  stay distinct because they generalise fine.
- **Why not the obvious model:** the published-CSV model scores 0.97 on the
  dataset but **~6% on live traffic** because its #1 feature (`IAT`) is an
  **unreproducible capture artifact**. We retrain on our own extractor's
  features instead ("Path A").
- **Final deployable numbers:** Stage-2 (28 classes) **0.952 acc / 0.749
  macro-F1** (merged floods 0.99, Mirai 1.0, MITM 0.98, fragmentation 0.97–0.99,
  Recon/Spoofing 0.53–0.84, Web 0.15–0.47); gate ~95% on-site. Latency 4.8 ms.
- **Hard limit:** application-layer attacks (Web, brute force) are weakly
  detectable from flow features — they need payload inspection.
- **Artifacts:** `artifacts/models/pathA/`.

---

## 1. Goal & constraints
Classify live flows (tcpdump → extractor → 46 CICIoT2023 features) into attack
types on a Pi 4 (≤10 MB, target <5 ms/flow). Train on PC, infer on Pi.

## 2. Dataset & sampling
CICIoT2023: 169 CSVs, ~13 GB, 46.7 M rows, 46 features + label, 34 classes,
heavily imbalanced. For the original model: classes >1 M rows down-sampled to
100 k → `artifacts/sampled.parquet` (5.76 M rows).

## 3. Feature engineering
27 engineered features from the raw 46 (flag ratios, rates, size dispersion,
protocol breadth, statistical consistency) — all per-flow, reproducible on the
Pi. `src/feature_engineering.py` (training) and `src/fast_transform.py` (numpy,
bit-identical, ~18× faster). Final list = **73 features** (§11). Engineered
`log_iat` was the original model's #4 feature.

## 4. Model selection (original, published CSVs)
LightGBM **collapses at 34-class** (~0.25; binary 0.99, 8-class 0.98) — a
multiclass instability, diagnosed not assumed. XGBoost (0.95, 27 min) and HistGBDT
(0.75) also underperformed. **DecisionTree won** on macro-F1 + size and became
the base model.

Original model on the published dialect: 34-class **0.970 / 0.881**, 8-class
**0.981 / 0.871** — real on the dataset, but see §6, they do not transfer live.

## 5. The live-extraction pivot
We built a pcap → 46-feature extractor (`src/extractor/pcap_to_features.py`,
dpkt, official CIC methodology). Feeding real extracted features to the original
model scored **6%**; correcting only `IAT` jumped it to **99%** → `IAT` is the
sole blocker.

## 6. The `IAT` artifact (central bad finding)
Published `IAT` is near-constant **~8.3×10⁷ across the whole dataset** (per-class
means within ~1.1 %) yet is the #1 feature — an artifact of CIC's **unreleased**
production extractor, **not a real inter-arrival time**. The released code's
formula yields ~10⁻⁴ s (~10 orders off); every variant we tested missed (closest
21×). So the celebrated 0.97 model **cannot work live**. (Removing `IAT` also
collapses the rare app-layer classes → they were identified *via* the artifact.)

## 7. Extractor correctness
Calibrated to the published dialect where possible (cumulative-bytes
Header_Length, per-flow IAT, 10 MB-chunk reset, per-flow Rate, global windowing).
**Quantified: 85 % of important-feature cells match the published dataset within
tolerance** (best classes 92–100 %). Not bit-perfect (impossible), but the
meaningful features are right, and for Path A only **self-consistency** matters.

## 8. Path A — retrain on extractor features
Train *and* serve on our extractor → zero skew, no artifact. We extracted **all
34 classes**, then **multi-session** (101 pcaps), de-duplicated by **content
fingerprint** (each unique session counted once).

| Model | Single-session | **Multi-session (1.29 M rows)** |
|---|---|---|
| 34-class acc / macro-F1 | 0.768 / 0.702 | **0.859 / 0.715** |
| 8-class macro-F1 | 0.760 | 0.763 |
| binary acc | 0.926 | 0.972 |

Multi-session lifted the data-starved classes (Web 0.51→0.62, BruteForce
0.44→0.54, Spoofing 0.71→0.89).

## 9. Cross-capture generalization (the tests that matter)
Within-capture F1 **overstates** generalization — so we tested leave-one-session-out.

**Benign:** train on 2 captures, test on a held-out 3rd → **14 % benign recall**
(86 % false alarm). Multi-session did *not* fix it; same-capture benign F1 even
dropped (0.82→0.65). **Benign is open-ended and network-specific** — it cannot be
learned as a fixed class.

**Attacks (leave-one-session-out, 18 multi-session classes):**
- **11 generalize perfectly** (cross ≈ within ≈ 1.0): all Mirai, most DDoS
  floods, all fragmentation.
- **The DoS family does NOT:** DoS-UDP **0.79 → 0.11**, DoS-SYN/TCP roughly halve
  — because DoS vs DDoS is flow-indistinguishable; the split was session
  memorization.

## 10. The two fixes (validated)

**(A) Protocol-wise flood merge (final scheme).** Each DoS flood is folded into
its matching DDoS flood **by protocol** — `DoS-X_Flood + DDoS-X_Flood → X_Flood`
for X∈{UDP, SYN, TCP, HTTP} — and `DDoS-SynonymousIP_Flood` (a spoofed SYN flood,
not separable) → `SYN_Flood`. All other classes (Mirai, the non-flood DDoS
subtypes ICMP/PSHACK/RSTFIN/fragmentation/SlowLoris, Recon, Spoofing, Web,
BruteForce) stay at full 34-class granularity → **28 attack classes**.

Leave-one-session-out recall on the merged floods (was 0.11–0.70 as separate
classes): DoS-UDP→UDP_Flood **98.1%**, DoS-SYN→SYN_Flood **98.9%**,
DoS-TCP→TCP_Flood **99.5%**, DoS-HTTP→HTTP_Flood **94.8%**, DDoS-SYN **99.7%**,
DDoS-SynonymousIP→SYN_Flood **99.9%**. Merged-flood F1: UDP 0.995, TCP 0.998,
SYN 0.995, HTTP 0.976. The non-flood DDoS subtypes were left distinct because
they already generalise (cross ≈ within ≈ 1.0).

**(B) Two-stage architecture for benign.** Instead of a supervised benign class,
a **one-class Isolation Forest gate** learns "normal for this network":

| Scenario | Benign accepted | Attacks flagged |
|---|---|---|
| Supervised benign (baseline) | 14 % | — |
| Anomaly gate, CROSS-NETWORK (generic) | **52 %** | 92 % |
| Anomaly gate, **ON-SITE** (calibrated) | **94–96 %** | **~92 %** |

No model choice fixes benign (DecisionTree 17.7 %, RandomForest 14.0 %) — it is
structural. On-site calibration is the answer.

## 11. Final feature list (73)
All 73 (46 raw + 27 engineered), no pruning →
`artifacts/models/pathA/feature_list.json`.

**Raw (46):** flow_duration, Header_Length, Protocol Type, Duration, Rate, Srate,
Drate, fin/syn/rst/psh/ack/ece/cwr_flag_number, ack/syn/fin/urg/rst_count, HTTP,
HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IPv, LLC, Tot sum,
Min, Max, AVG, Std, Tot size, IAT, Number, Magnitue, Radius, Covariance, Variance,
Weight.

**Engineered (27):** flag_number_sum, flag_count_sum, syn_ack_ratio, rst_ratio,
fin_ratio, urg_ratio, syn_no_ack, rst_fin_combo, log_rate, rate_dst_ratio,
rate_per_byte, bytes_per_duration, rate_iat_consistency, size_range, size_cv,
min_max_ratio, header_to_size, avg_pkt_size, size_per_count, l7_proto_count,
transport_count, tcp_no_l7, udp_no_l7, variance_to_avg, radius_to_magnitude,
duration_ratio, log_iat.

> `IAT` is in the list, but it is our **real** per-flow inter-arrival, not the
> published artifact. The anomaly gate additionally signed-log-compresses and
> StandardScaler-normalises the features (a scaler IS needed for Stage 1, unlike
> the trees).

## 12. The deployable two-stage model (`src/deploy/predict.py`)
```
raw 46 features -> build_matrix(df, feature_list)   # +27 engineered, NaN/inf->0
  STAGE 1 gate:  signed_log -> scaler -> IsolationForest(50)  -> normal | anomalous
  if normal    -> "BenignTraffic"
  if anomalous -> STAGE 2: DecisionTree -> one of 28 attack classes
                           (UDP/SYN/TCP/HTTP_Flood, DDoS-* subtypes, Mirai-*,
                            Recon-*, MITM/DNS spoofing, Web-*, DictionaryBruteForce)
```
```python
ids = IDSModel()                       # 28-class attacks-only Stage-2 + benign gate
ids.calibrate_benign(local_benign_df)  # ON-SITE: tune gate to this network (run once)
ids.predict(flow_df)                   # 'BenignTraffic' or an attack class
ids.predict_detailed(flow_df)          # is_anomaly / attack_type / label
```

## 13. Honest per-class deployment table (Stage-2, 28 classes)
Grouped here by deployment confidence; full per-class numbers in §13.1.

| Group (classes) | F1 range | Cross-capture | Deploy verdict |
|---|---|---|---|
| **Merged floods** UDP/SYN/TCP/HTTP_Flood | 0.97–0.995 | ✅ 95–99.9% held-out | trustworthy |
| **Mirai** ×3, **DDoS-ICMP_Flood/RSTFIN/PSHACK** | 0.99–1.0 | ✅ ~1.0 | trustworthy |
| **Fragmentation** (ACK/UDP/ICMP), **MITM-ArpSpoofing** | 0.97–0.99 | ✅ | trustworthy |
| **DDoS-SlowLoris, Recon-HostDiscovery** | 0.82–0.84 | ✅/⚠️ | good |
| **DNS_Spoofing, VulnerabilityScan** | 0.70–0.73 | ⚠️ | moderate |
| **DictionaryBruteForce, Recon-Port/OSScan** | 0.54–0.61 | ⚠️ scan subtypes confuse | moderate |
| **Web** (XSS/SQLi/Backdoor/Upload/CmdInj/BrowserHijack), **Recon-PingSweep** | 0.15–0.47 | ❌ slip the gate | **weak — needs payload inspection** |
| **Benign** | (gate) | ❌ cross-network; ✅ on-site ~95% | **calibrate on-site** |

Stage-2 overall: **accuracy 0.952, macro-F1 0.749, weighted-F1 0.956**
(held-out, 250,792 attack flows). Reproduce with
`src/extractor/train_proto_merged.py`.

### 13.1 Full per-class precision / recall / F1 (28 classes, held-out)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Mirai-greeth_flood | 0.9995 | 0.9992 | 0.9994 | 12,385 |
| Mirai-greip_flood | 0.9996 | 0.9988 | 0.9992 | 13,256 |
| DDoS-ICMP_Flood | 0.9996 | 0.9980 | 0.9988 | 17,999 |
| Mirai-udpplain | 0.9971 | 0.9974 | 0.9972 | 12,287 |
| TCP_Flood | 0.9998 | 0.9938 | 0.9968 | 29,992 |
| DDoS-RSTFINFlood | 0.9998 | 0.9930 | 0.9964 | 17,997 |
| DDoS-UDP_Fragmentation | 0.9981 | 0.9941 | 0.9961 | 6,315 |
| DDoS-PSHACK_Flood | 0.9992 | 0.9928 | 0.9960 | 17,998 |
| UDP_Flood | 0.9993 | 0.9911 | 0.9952 | 26,997 |
| SYN_Flood | 0.9993 | 0.9902 | 0.9947 | 38,996 |
| DDoS-ACK_Fragmentation | 0.9933 | 0.9869 | 0.9901 | 9,780 |
| HTTP_Flood | 0.9879 | 0.9638 | 0.9757 | 8,504 |
| MITM-ArpSpoofing | 0.9869 | 0.9643 | 0.9755 | 5,997 |
| DDoS-ICMP_Fragmentation | 0.9892 | 0.9541 | 0.9713 | 6,423 |
| Recon-HostDiscovery | 0.8472 | 0.8282 | 0.8376 | 2,986 |
| DDoS-SlowLoris | 0.8167 | 0.8014 | 0.8090 | 2,986 |
| DNS_Spoofing | 0.7951 | 0.6878 | 0.7376 | 2,944 |
| VulnerabilityScan | 0.7318 | 0.6403 | 0.6830 | 2,975 |
| DictionaryBruteForce | 0.6443 | 0.5817 | 0.6114 | 2,613 |
| Recon-PortScan | 0.5959 | 0.5877 | 0.5918 | 2,966 |
| Recon-OSScan | 0.6086 | 0.5279 | 0.5654 | 2,978 |
| BrowserHijacking | 0.3801 | 0.5384 | 0.4456 | 1,172 |
| SqlInjection | 0.3933 | 0.5024 | 0.4412 | 1,049 |
| CommandInjection | 0.3588 | 0.4214 | 0.3876 | 1,082 |
| Recon-PingSweep | 0.2204 | 0.5111 | 0.3080 | 452 |
| Backdoor_Malware | 0.2279 | 0.3851 | 0.2864 | 644 |
| XSS | 0.1783 | 0.3147 | 0.2277 | 769 |
| Uploading_Attack | 0.0981 | 0.3640 | 0.1545 | 250 |
| **Macro avg** | 0.7445 | 0.7682 | **0.7489** | — |
| **Weighted avg** | 0.9605 | 0.9524 | **0.9559** | — |

### 13.2 Oversampling the weak classes — tested and ruled out
We tested interpolation-based oversampling (SMOTE, Borderline-SMOTE, ADASYN — no
generative/GAN synthesis) on the 6 weak Web classes, vs the current
`class_weight='balanced'` baseline (`src/extractor/oversample_compare.py`):

| | Baseline | SMOTE | Borderline | ADASYN |
|---|---|---|---|---|
| Weak-class mean F1 | 0.318 | 0.327 | **0.329** | 0.326 |
| Strong-class mean F1 | 0.865 | 0.867 | 0.864 | 0.866 |

**Result:** boundary-focused methods do **not** hurt the strong classes, but
improve the weak ones only **+0.01 F1** (negligible, inconsistent per class).
**Reason:** the weak classes' problem is *feature-space inseparability*, not
imbalance — interpolating between minority samples that already overlap benign
adds no separating signal. **Conclusion: keep `class_weight`; the fix is payload
features, not resampling.**

## 14. Deployment artifacts (`artifacts/models/pathA/`)
| File | Size | Purpose |
|---|---|---|
| `pathA_dtree_attack.joblib` | 1.46 MB | **Stage-2 (deployable): 28 attack classes, protocol-merged floods** |
| `label_encoder_attack.joblib` | <1 KB | index → attack class name |
| `anomaly_gate.joblib` | ~0.2 MB | **Stage-1 gate** (scaler + 50-tree IsolationForest) |
| `feature_list.json` | 1 KB | the 73-feature order |
| `pathA_dtree_attackcat.joblib` (6 cats) / `8class` / `34class` (+ encoders) | — | alternative granularities |
| `DEPLOYMENT_manifest.json` | — | pipeline + caveat |

> `predict.py` loads the 28-class model by default (`model="attack"`); pass
> `model="attackcat"` for the 6 broad categories.

- **No scaler for Stage 2** (tree, scale-invariant); the **gate has its own scaler**.
- **Pi also needs the code:** `config.py`, `fast_transform.py`,
  `feature_engineering.py`, `deploy/predict.py`.
- ⚠️ `artifacts/models/*.joblib` (outside `pathA/`) are the artifact-dependent
  ORIGINAL model — **do not deploy.**

**Latency (50-tree gate, PC):** single-flow **4.8 ms**, batch 256 = 5.7 ms,
batch 5000 = 4.76 µs/flow. On a Pi (~3–5×) batched is fine; single-flow ~15–24 ms.

## 15. Deployment recipe
1. Copy the code modules + `artifacts/models/pathA/` to the Pi.
2. `ids = IDSModel()`.
3. **On-site once:** capture a clean benign baseline on the target network →
   `ids.calibrate_benign(local_benign_df)` (→ ~95 % benign).
4. Live: `ids.predict(flow_df)`.
5. Treat **Web / brute-force** as out of scope for flow features (payload
   inspection is a separate system).

## 16. Findings summary
**Good:** feature engineering helps; DecisionTree is tiny/fast/strong;
extractor matches the dialect 85 %; Path A is genuinely deployable; multi-session
helped rare classes; fingerprint dedup guarantees no duplicate sessions; the
protocol-wise flood merge fixes DoS generalization (held-out DoS sessions
95–99.9 %); the on-site anomaly gate fixes benign (~95 %).

**Bad / limits:** the published `IAT` artifact makes the 0.97 CSV model
non-deployable (~6 % live); Web/brute-force are weakly detectable from flow
features (fundamental); benign is network-specific (must calibrate on-site);
DoS-vs-DDoS and Recon scan subtypes are flow-indistinguishable (merge / use
categories); within-capture F1 overstates generalization — always test
cross-capture.

## 16.1 Remaining shortcomings & solutions

| # | Shortcoming | Why | Solution |
|---|---|---|---|
| 1 | **App-layer attacks weak** (Web 0.15–0.47, brute force ~0.6) | flow stats don't carry the signal; resampling ruled out (§13.2) | add **payload / DPI features** (HTTP/URI/SQL-keyword n-grams, TLS JA3/SNI) and/or a **signature engine** (Suricata/Snort) alongside the ML — defense in depth |
| 2 | **No multi-flow / host context** (forced DoS↔DDoS merge, scan breadth) | single-flow model can't see fan-out (# sources, # dst ports/IPs) | add a **host/time-window aggregation feature layer** (per-source-IP stats over a sliding window) → recovers DoS-vs-DDoS and lifts Recon |
| 3 | **Recon scan subtypes confuse** (OSScan/PortScan/PingSweep 0.31–0.59) | flow-indistinguishable | merge to a single `Recon` class (as done for floods) **or** add scan-breadth features (item 2) |
| 4 | **Benign needs on-site calibration & drifts** | benign is network-specific and changes over time | gate is calibrated on-site (done); add **periodic re-calibration** + **feature-drift monitoring** to catch concept drift |
| 5 | **Calibration-data poisoning** | a benign baseline containing hidden attacks teaches the gate to accept them | capture during a known-quiet window **and** pre-filter the baseline with the attack classifier before fitting the gate |
| 6 | **Extractor-consistency dependence** | Path A needs the *same* extractor at train & serve | freeze one extractor spec; add a **live feature-distribution monitor** that alerts on drift vs training stats |
| 7 | **No confidence / abstention** | model emits a hard label even when unsure (low-precision weak classes → false positives) | use `predict_proba` + the gate's anomaly **score** with operator-tuned thresholds → emit "suspicious / low-confidence" instead of a wrong label |
| 8 | **Single-flow Pi latency ~15–24 ms** (gate) | 50-tree IsolationForest fixed cost | **batch** flows per time window (11.6 µs/flow batched), or drop the gate to ~30 trees |
| 9 | **Some classes have one capture session** (DNS_Spoofing, several Recon/Web) | within-capture F1 may be optimistic | collect **≥2 sessions per class** (different times/networks) to validate & harden generalization |
| 10 | **No adversarial robustness** | trained on natural attack tools; evasion (padding/timing) possible | adversarial testing/training + defense-in-depth (signatures + anomaly gate); open research problem |

## 17. Key files
```
src/feature_engineering.py / fast_transform.py   features (train / inference)
src/extractor/pcap_to_features.py                pcap -> 46 features (calibrated)
src/extractor/extract_all.py                     multi-session extract + fingerprint dedup
src/extractor/cross_capture_attacks.py           leave-one-session-out attack test
src/extractor/anomaly_gate_test.py               Isolation Forest validation
src/extractor/finalize_deployment.py             trains category Stage-2 + re-saves gate
src/extractor/train_proto_merged.py              trains the FINAL 28-class Stage-2 + cross-capture
src/extractor/oversample_compare.py              SMOTE/Borderline/ADASYN comparison (negative result)
src/deploy/predict.py                            TWO-STAGE inference + on-site calibration
artifacts/pcap_features/                         per-class extractor features (+cache/, manifest)
artifacts/models/pathA/                          DEPLOYABLE: Stage-2 + gate + encoders + feature_list
docs/PROJECT_REPORT.md                           this document
```
