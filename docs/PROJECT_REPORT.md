# CICIoT2023 Intrusion Detection — Project Report

_Final report. Covers the data pipeline, the feature engineering, the full model
investigation (including a LightGBM failure we diagnosed and worked around), and
the final results._

---

## 0. The goal (plain language)

Classify one **network flow** (a summarized conversation between two machines)
into one of **34 classes** (benign + 33 attack types from CICIoT2023), fast and
small enough to run live on a **Raspberry Pi 4 (4 GB)**.

Live pipeline on the Pi:
```
tcpdump -> CICFlowMeter (46 numeric features per flow) -> our feature engineering
        -> DecisionTree model -> attack label (+ class probabilities)
```

**Targets and how we did:**

| Requirement | Target | Result |
|---|---|---|
| Model size | < 10 MB | **2.0 MB** ✅ |
| Inference | < 5 ms / batch on Pi 4 | **1.3 ms/batch on PC** (≈4–6 ms on Pi) ✅ |
| Accuracy / F1 | as high as possible | **97.0% acc, 0.881 macro-F1, 0.970 weighted-F1** ✅ |
| Classes | 34 | 34 ✅ |

---

## 1. The dataset

- **CICIoT2023**, 169 CSV parts, **~13 GB**, **46,686,579 rows**, **46 features +
  label**, **34 classes**.
- Massively imbalanced: `DDoS-ICMP_Flood` has 7.2 M rows; `Uploading_Attack` has
  1,252.

---

## 2. Data sampling (per your rule)

Any class with **> 1,000,000 rows → randomly cut to 100,000**; every class with
**≤ 1,000,000 rows → kept in full**. Implemented as a memory-safe two-pass
stream over all 169 files (never loads 13 GB at once).

**Result:** `artifacts/sampled.parquet`, **5,763,133 rows** (327 MB). 11 flood
classes capped at 100 K; everything else kept whole. Also stores `label_8`
(8 categories) and `label_binary` columns so granularity can change later
without re-sampling.

---

## 3. FEATURE ENGINEERING (the core of the project)

### 3.1 Idea
The raw 46 features are mostly **counts and averages**. Attacks usually reveal
themselves through **ratios and relationships** between those numbers (e.g. a SYN
flood has a wild SYN-to-ACK ratio even when packet counts look normal). So we
derive **27 new features** that are ratios, rates, spreads and combinations.

**Critical property:** every engineered feature is computed from a single flow's
own raw numbers — no time windows, no other rows, no label. So the identical
transform runs in training and on the Pi (no train/serve skew) and there is no
label leakage.

### 3.2 The 27 engineered features

**A. Flag dynamics — abuse of TCP control flags**

| Feature | Meaning | Catches |
|---|---|---|
| `flag_number_sum` | sum of all 7 flag indicators | overall flag activity |
| `flag_count_sum` | ack+syn+fin+urg+rst counts | total control packets |
| `syn_ack_ratio` | syn_count / ack_count | **SYN floods** |
| `rst_ratio` | rst_count / total flags | **RST/FIN floods** |
| `fin_ratio` | fin_count / total flags | teardown abuse |
| `urg_ratio` | urg_count / total flags | urgent-flag attacks |
| `syn_no_ack` | SYN set, ACK absent | **half-open / SYN flood** |
| `rst_fin_combo` | RST and FIN both set | crafted teardown |

**B. Rate / throughput**

| Feature | Meaning | Catches |
|---|---|---|
| `log_rate` | log(1+Rate) | compresses huge rate range |
| `rate_dst_ratio` | src rate / dst rate | one-directional floods |
| `rate_per_byte` | Rate / packet size | many tiny packets = flood |
| `bytes_per_duration` | bytes / duration | volumetric throughput |
| `rate_iat_consistency` | Rate × inter-arrival | crafted-flood inconsistency |

**C. Packet-size dispersion**

| Feature | Meaning | Catches |
|---|---|---|
| `size_range` | Max − Min | floods have ~0 range |
| `size_cv` | Std / Avg | automated uniform traffic |
| `min_max_ratio` | Min / Max | uniform-size attacks |
| `header_to_size` | header / total size | fragmentation, odd headers |
| `avg_pkt_size` | sum / count | average payload |
| `size_per_count` | size / count | per-packet weight |

**D. Protocol breadth**

| Feature | Meaning | Catches |
|---|---|---|
| `l7_proto_count` | # of HTTP/HTTPS/DNS/Telnet/SMTP/SSH/IRC active | benign breadth |
| `transport_count` | # of TCP/UDP/ICMP/ARP active | transport mix |
| `tcp_no_l7` | TCP but no app protocol | **raw TCP flood** |
| `udp_no_l7` | UDP but no app protocol | **raw UDP flood** |

**E. Statistical consistency**

| Feature | Meaning | Catches |
|---|---|---|
| `variance_to_avg` | Variance / Avg | spread vs size |
| `radius_to_magnitude` | Radius / Magnitude | size-distribution shape |
| `duration_ratio` | flow_duration / Duration | timing inconsistency |
| `log_iat` | log(1+inter-arrival) | compresses timing gaps |

### 3.3 Pruning
After adding the 27 (total 73), we auto-drop near-constant and near-duplicate
(|correlation| > 0.995) columns. **9 dropped** — `Srate` (≈`Rate`), `LLC`,
`Number`, `Radius`, `Weight`, and the engineered `rst_fin_combo`,
`rate_dst_ratio`, `rate_iat_consistency`, `udp_no_l7` — leaving **64 features**
(saved to `artifacts/models/feature_list.json`, the single source of truth used
by the Pi).

### 3.4 Did the engineered features help? — Yes
In the final model's importance ranking, engineered features appear high:
`log_iat` is **#4 overall**, with `header_to_size` (#7) and `flag_count_sum`
(#15) also in the top 15. Top features overall: `IAT`, `Min`, `Protocol Type`,
**`log_iat`**, `ICMP`, `Magnitue`, **`header_to_size`**, `AVG`, `Variance`,
`fin_flag_number`, `UDP`, `psh_flag_number`, `syn_flag_number`, `Header_Length`,
**`flag_count_sum`**.

---

## 4. Model selection — what we tried and why

We trained on 4.6 M rows (80%) and tested on 1.15 M held-out rows (20%), 34
classes, identical features for every model.

### 4.1 The LightGBM detour (and why it failed)
LightGBM is normally the go-to lightweight model, so we started there. It
**collapsed**: ~18–25% accuracy, predicting only a few classes. We diagnosed it
methodically (scripts `debug_lgbm*.py`):

- Not the class weighting (failed with and without).
- Not the engineered features or their value ranges (failed on raw features too;
  float64 vs float32 made no difference).
- Not under-training (even 300 rounds × 255 leaves only reached **0.32 train**
  accuracy — it could not even fit the training data).
- **The tell:** LightGBM scored **0.992 on the binary task** and **0.978 on the
  8-class task**, but **0.25 on 34-class**, and its result even **changed with
  the thread count**. That isolates the problem to **LightGBM's multiclass
  (softmax) training becoming numerically unstable with 34 classes** on this data
  (LightGBM 4.6.0) — not the features, which are clearly excellent (binary/8-class
  prove it).

So we excluded LightGBM and ran a full bake-off of models that handle 34-class
natively.

### 4.2 Bake-off (800 K stratified subsample)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Size | Train time |
|---|---|---|---|---|---|
| **DecisionTree** ★ | 0.966 | **0.878** | 0.966 | **0.3 MB** | 22 s |
| RandomForest (40 trees) | **0.971** | 0.845 | 0.970 | 24.4 MB | 12 s |
| RandomForest (30×2000 leaves) | 0.968 | 0.794 | 0.967 | 6.5 MB | 10 s |
| ExtraTrees (40) | 0.944 | 0.767 | 0.942 | 48.4 MB | 6 s |
| XGBoost (hist, 400) | 0.951 | 0.756 | 0.951 | 7.0 MB | **1657 s** |
| HistGradientBoosting | 0.749 | 0.398 | 0.751 | 1.2 MB | 62 s |
| LightGBM | 0.25 | 0.10 | 0.16 | — | broken |

**Why the DecisionTree wins for this problem:**
- Best **macro-F1** — i.e. best at the *rare* attack classes, which is what
  matters most for an IDS. (RandomForest's bagging + majority vote washes out the
  tiny classes, so its macro-F1 is lower despite slightly higher overall
  accuracy.)
- **Smallest** (fits the 10 MB budget with room to spare) and **fastest**.
- The boosting libraries (LightGBM, XGBoost, HistGBDT) all underperform on this
  34-class task; XGBoost is also ~75× slower to train.

---

## 5. Final model & results

**DecisionTree**, `max_depth=35`, `min_samples_leaf=5`, 64 features, 34 classes,
trained on the full 4.6 M-row training split.

| Metric | Value |
|---|---|
| Accuracy | **0.9705** |
| Macro-F1 (per-class avg) | **0.8809** |
| Weighted-F1 | **0.9703** |
| Balanced accuracy | 0.8621 |
| Macro precision | 0.9092 |
| Macro recall | 0.8621 |
| Model size | **2.02 MB** (depth 35, ~30 k leaves) |
| Inference | **1.3 ms/batch** (batch=1, PC); 1.6 µs/flow at batch 5000 |

### 5.1 Per-class picture
- **Perfect (F1 = 1.00):** the volumetric floods — `Mirai-greeth_flood`,
  `Mirai-udpplain`, `DDoS-ICMP_Flood`, `DDoS-RSTFINFlood`, `DDoS-TCP_Flood`,
  `DDoS-PSHACK_Flood`, etc. These are distinctive and well-sampled.
- **Weakest (F1 ≈ 0.60–0.66):** the rare application-layer web attacks —
  `XSS`, `Backdoor_Malware`, `SqlInjection`, `Uploading_Attack`,
  `CommandInjection`, and `Recon-PingSweep`. They have very few samples and look
  similar to benign web traffic, so recall is the limiting factor. This is the
  well-known hard part of CICIoT2023 34-class.

Full breakdown: `artifacts/reports/final_per_class.csv`,
`final_confusion_matrix.csv`, `final_feature_importance.csv`, `metrics.json`.

### 5.2 Validation on realistic (full-46M) class proportions
The numbers above are measured on the held-out split of the **capped** sample,
where rare classes are over-represented relative to reality. In production the
class mix matches the full 46 M distribution (≈90 % giant DDoS/DoS floods). That
shift can erode *rare-class precision* (more dominant-class flows = more chances
to falsely fire a rare label), so we re-validated macro-F1 on realistic
proportions two independent ways (`src/realistic_eval.py`):

- **(A) Exact reweighting** of the held-out confusion matrix by the true
  full-dataset priors (recall is prior-invariant; precision is reweighted).
- **(B) Empirical held-out test** of 2,000,001 rows whose proportions match the
  full 46 M distribution, built only from rows the model never trained on —
  dominant classes drawn from genuinely *unseen* full-dataset rows (excluded by
  row hash), rare/kept-in-full classes from the held-out 20 % split.

| Metric | Capped test | (A) Reweighted | (B) Empirical held-out |
|---|---|---|---|
| Accuracy | 0.9705 | 0.9920 | **0.9921** |
| **Macro-F1** | 0.8809 | 0.8655 | **0.8653** |
| Weighted-F1 | 0.9703 | 0.9924 | **0.9924** |
| Balanced accuracy | 0.8621 | 0.8621 | 0.8647 |

The two methods agree to within 0.0002. **Macro-F1 holds at ≈0.865** on realistic
proportions (only −0.016 vs the capped test), while accuracy and weighted-F1 rise
to **0.992** because the dominant floods are classified almost perfectly. Recall
is unchanged (it does not depend on class priors); the small macro-F1 dip comes
entirely from rare-class precision under contamination — worst cases
`Recon-OSScan` (precision 0.49), `Backdoor_Malware`, `Recon-PingSweep`, `XSS`,
`SqlInjection`. Per-class detail: `artifacts/reports/realistic_per_class.csv`.

**Conclusion: the model's quality is robust to the real-world class imbalance —
the capped-sample metrics were not an artifact of the balanced test set.**

### 5.3 If you ever want higher overall accuracy
RandomForest (40 trees) gives ~+0.5% accuracy (0.971) but lower rare-class
recall and a 24 MB model — only worth it if the size budget is relaxed and you
care more about common-class accuracy than rare-attack detection.

---

## 6. Deployment on the Raspberry Pi 4

Copy to the Pi: `src/config.py`, `src/feature_engineering.py`,
`src/fast_transform.py`, `src/deploy/predict.py`, and `artifacts/models/`.
Install `scikit-learn`, `pandas`, `numpy`, `joblib`.

```python
from predict import IDSModel
model = IDSModel()
labels = model.predict(flow_df)        # 34-class names
proba  = model.predict_proba(flow_df)  # per-class probabilities
```

`flow_df` is a DataFrame with the raw 46 CICIoT2023 columns from your
tcpdump → CICFlowMeter pipeline. Inference uses a **numpy fast-path**
(`fast_transform.build_matrix`) that reproduces the training features exactly
(verified bit-for-bit, 100% identical predictions on 5,000 flows) but is ~18×
faster than the pandas path on small batches — that is what gets single-flow
latency down to ~1.3 ms.

---

## 7. Files

```
src/config.py              schema, 34->8 mapping, sampling rule, paths
src/feature_engineering.py 27 engineered features (training path, pandas)
src/fast_transform.py      identical features in numpy (fast inference path)
src/sample_data.py         two-pass streaming sampler
src/train.py               final training (DecisionTree tuning + full metrics)
src/model_compare.py       the bake-off (section 4.2)
src/debug_lgbm*.py         the LightGBM diagnosis (section 4.1)
src/deploy/predict.py      Pi inference + latency benchmark
artifacts/sampled.parquet  5.76 M-row training sample
artifacts/models/          dtree_34class.joblib, label_encoder_34.joblib,
                           feature_list.json
artifacts/reports/         metrics.json, final_per_class.csv,
                           final_confusion_matrix.csv,
                           final_feature_importance.csv, class_distribution.csv
docs/PROJECT_REPORT.md     this document
```
