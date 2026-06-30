# CICIoT2023 Lightweight IDS (Model A)

A 34-class intrusion-detection model for the **CICIoT2023** dataset, built to run
live on a **Raspberry Pi 4**. Classifies each network flow as benign or one of 33
attack types.

```
tcpdump -> CICFlowMeter (46 features) -> feature engineering -> DecisionTree -> attack label
```

## Results (held-out test, 34 classes)

| Metric | Capped test | Realistic 46M proportions* |
|---|---|---|
| Accuracy | **97.0%** | **99.2%** |
| Macro-F1 | **0.881** | **0.865** |
| Weighted-F1 | **0.970** | **0.992** |
| Model size | **2.0 MB** (budget: 10 MB) | — |
| Inference | **1.3 ms/batch** on PC (≈4–6 ms on Pi 4; budget 5 ms) | — |

\*Validated on a 2M-row held-out test matching the full 46M class distribution
(and cross-checked by exact confusion-matrix reweighting) — macro-F1 holds; see
[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) §5.2.

Model: a tuned **DecisionTree** (`max_depth=35, min_samples_leaf=5`, 64 features).
It beat RandomForest, ExtraTrees, XGBoost and HistGradientBoosting on macro-F1 +
size; LightGBM was excluded after its multiclass training was found to be broken
on this 34-class task (works on binary/8-class, collapses at 34). Full story,
feature-engineering details, and the model bake-off are in
[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

## Run it

```powershell
pip install -r requirements.txt                 # setup (scikit-learn pinned to 1.8.0)
python src/sample_data.py                       # build artifacts/sampled.parquet (~5.76M rows)
python src/train.py                             # train + evaluate -> artifacts/models, artifacts/reports
python src/deploy/predict.py artifacts/bench_flows.csv   # latency benchmark
```

## Deploy on the Pi

The **deployable** system is the two-stage **Path A** model (a benign anomaly
gate calibrated on-site + a 28-class attack classifier) — it avoids the `IAT`
artifact that makes the single-stage model non-portable. Full instructions:
[`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md).

Copy `src/config.py`, `src/feature_engineering.py`, `src/fast_transform.py`,
`src/extractor/pcap_to_features.py`, `src/deploy/predict.py`,
`src/deploy/ids_runner.py`, and `artifacts/models/pathA/` to the Pi, then
`pip install -r requirements.txt` (`scikit-learn` is pinned to 1.8.0 so the saved
models unpickle).

```python
from predict import IDSModel
ids = IDSModel(models_dir="models/pathA")   # 28 attack classes + benign gate
ids.calibrate_benign(local_benign_df)       # ON-SITE: tune the gate to this network
labels = ids.predict(flow_df)               # 'BenignTraffic' or an attack name
```

Live capture loop: `sudo python3 ids_runner.py --iface eth0` (tcpdump → extract →
score → `alerts.csv`). View alerts on a monitor with the Streamlit dashboard:
`streamlit run dashboard.py`.

## Test without live traffic

No IoT traffic handy? Generate a synthetic capture and run the full pipeline:

```powershell
python tests/make_test_pcap.py      # -> tests/sample_traffic.pcap (benign + floods)
python tests/test_pipeline.py       # pcap -> features -> two-stage IDS -> labels
```

See [`tests/README.md`](tests/README.md) for replaying real attack pcaps too.

## Layout

```
requirements.txt              pinned deps (grouped: runtime / capture / dashboard / training)
src/
├── config.py                 core config: features, class maps, paths
├── feature_engineering.py    27 engineered features (training path)
├── fast_transform.py         same features in numpy (fast inference)
├── sample_data.py            two-pass streaming sampler (per-class cap rule)
├── train.py / train_8class.py   final training + metrics
├── model_compare.py, debug_lgbm*.py, experiment.py, realistic_eval.py
├── extractor/
│   ├── pcap_to_features.py   dpkt pcap -> 46 CICIoT2023 features (live capture)
│   ├── extract_all.py        multi-session extraction + dedup
│   ├── train_proto_merged.py trains the 28-class Stage-2 + cross-capture test
│   ├── save_anomaly_gate.py  trains/saves the Stage-1 benign gate
│   └── *_test.py / *_compare.py  cross-capture & oversampling analysis
└── deploy/
    ├── predict.py            two-stage IDSModel (gate + attack classifier)
    ├── ids_runner.py         live capture loop (tcpdump) + --replay test mode
    ├── dashboard.py          Streamlit live-alert dashboard
    ├── suricata_collector.py Suricata eve.json -> alerts.csv bridge (Phase 1 DPI)
    └── suricata/             local.rules (payload/Web DPI) + disable.conf (ruleset trim)
tests/
├── make_test_pcap.py         synthetic pcap generator (no live traffic needed)
├── test_pipeline.py          end-to-end smoke test
└── README.md                 how to test without real traffic
artifacts/
├── models/pathA/             DEPLOYABLE: 28-class model, gate, encoders, feature_list
├── reports/                  metrics, confusion matrices, feature importances
└── bench_flows.csv           small sample for the latency benchmark
docs/
├── MODEL_OVERVIEW.md         current model: data, 62 features, algorithm, technique
├── DEPLOYMENT_GUIDE.md       Pi setup + flat model + monitoring (no Suricata)
├── DATA_COLLECTION.md        hping/nmap capture cmds, tcpdump, pcap -> ids_runner
├── PROJECT_REPORT.md         full report (data, model arc, honest limits)
├── CAPTURE_PCAPS.md          how to capture IP + WiFi pcaps from the office net
└── SURICATA_SETUP.md         optional/parked: Suricata DPI (not in the pipeline)
```
