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
python src/sample_data.py                       # build artifacts/sampled.parquet (~5.76M rows)
python src/train.py                             # train + evaluate -> artifacts/models, artifacts/reports
python src/deploy/predict.py artifacts/bench_flows.csv   # latency benchmark
```

## Deploy on the Pi

Copy `src/config.py`, `src/feature_engineering.py`, `src/fast_transform.py`,
`src/deploy/predict.py`, and `artifacts/models/` to the Pi; install
`scikit-learn pandas numpy joblib`.

```python
from predict import IDSModel
model = IDSModel()
labels = model.predict(flow_df)        # flow_df = raw 46 CICIoT2023 columns
proba  = model.predict_proba(flow_df)
```

The Pi uses a numpy fast-path (`fast_transform.py`) that reproduces the training
features bit-for-bit while keeping single-flow latency ~1.3 ms.

## Layout

- `src/feature_engineering.py` — 27 engineered features (training path)
- `src/fast_transform.py` — same features in numpy (fast inference)
- `src/sample_data.py` — two-pass streaming sampler (per-class cap rule)
- `src/train.py` — final training + metrics
- `src/model_compare.py`, `src/debug_lgbm*.py` — model bake-off & LightGBM diagnosis
- `artifacts/` — sampled data, models, reports
- `docs/PROJECT_REPORT.md` — full report
